#!/usr/bin/env python3

import logging
from libs.workflow_stage import WorkflowStage

class WorkflowGenerator:
    """
    Generates Nextflow workflow code from JSON workflow definition
    """
    
    def __init__(self, workflow_json):
        """
        Constructor
        
        Args:
            workflow_json (dict): Map containing workflow definition
        """
        self.dependencies = workflow_json.get('dependencies', {})
        self.stages = {}
        self.autogen_stages = {}
        
        # Create WorkflowStage objects
        for stage_data in workflow_json.get('stages', []):
            self.stages[stage_data.get('name')] = WorkflowStage(stage_data)
        
        # Validate dependencies
        self.validate_dependencies()
    
    def validate_dependencies(self):
        """
        Validates stage dependencies
        
        Raises:
            Exception: If dependencies are invalid
        """
        # Check that all dependencies exist as stages
        for stage_name, dep_list in self.dependencies.items():
            if stage_name not in self.stages:
                raise Exception(f"Stage {stage_name} referenced in dependencies but not defined")
            
            for dep_name in dep_list:
                if dep_name not in self.stages:
                    raise Exception(f"Dependency {dep_name} for stage {stage_name} not defined")
        
        # Validate stage types
        valid_types = ['csvloader', 'pathloader', 'view', 'process', 'cstloader', 'dependentpathloader']
        for name, stage in self.stages.items():
            if stage.type not in valid_types:
                raise Exception(f"Invalid stage type '{stage.type}' for stage {name}. "
                                f"Type must be one of: {', '.join(valid_types)}")
    
    def generate(self):
        """
        Generates the complete Nextflow workflow code
        
        Returns:
            str: Generated Nextflow code
        """
        code = []
        processes_to_generate = set()
        
        # Generate header
        code.append("#!/usr/bin/env nextflow")
        
        # First identify all processes in the dependency list that need to be generated
        for stage_name, dep_list in self.dependencies.items():
            # Add the stage itself if it's a process
            if stage_name in self.stages and self.stages[stage_name].is_process_stage():
                processes_to_generate.add(stage_name)
            
            # Check dependencies as well
            for dep_name in dep_list:
                if dep_name in self.stages and self.stages[dep_name].is_process_stage():
                    processes_to_generate.add(dep_name)
        
        # Identify multi-input processes and prepare their modified definitions
        multi_input_processes = set()
        
        for process_name in processes_to_generate:
            stage = self.stages[process_name]
            
            if stage.is_process_stage() and len(stage.inputs) >= 2:
                multi_input_processes.add(process_name)
                
                # Pre-process this multi-input stage to create its definition
                if stage.script.get('operator') == 'combine' and len(stage.inputs) == 2:
                    self.prepare_direct_combine_process_definition(process_name, stage)
                else:
                    self.prepare_multi_input_process_definition(process_name, stage)
        
        # Add process definitions for all processes
        for process_name in processes_to_generate:
            if process_name not in multi_input_processes:
                # Regular process - add standard definition
                code.append(self.stages[process_name].generate_process_stage())
            else:
                # Multi-input process - use modified definition from autogen_stages
                process_key = f"{process_name}_combined"
                direct_combine_key = f"{process_name}_direct_combine"
                
                if (process_key in self.autogen_stages and 
                    'process_code' in self.autogen_stages[process_key]):
                    code.append(self.autogen_stages[process_key]['process_code'])
                elif (direct_combine_key in self.autogen_stages and 
                      'process_code' in self.autogen_stages[direct_combine_key]):
                    code.append(self.autogen_stages[direct_combine_key]['process_code'])
                else:
                    # Fallback - add original process definition
                    code.append(self.stages[process_name].generate_process_stage())
        
        # Add other auto-generated process definitions (combine processes, etc.)
        for name, stage_info in self.autogen_stages.items():
            if ('process_code' in stage_info and 
                not name.endswith("_combined") and 
                not name.endswith("_direct_combine")):
                code.append(stage_info['process_code'])
        
        # Generate the workflow
        code.append("""
// Auto-generated workflow
workflow {
""")
        
        # Generate workflow body
        code.append(self.generate_workflow_body())
        
        # Close workflow
        code.append("""
}
""")
        
        return "\n".join(code)
    
    def generate_workflow_body(self):
        """
        Generates the workflow body in dependency order
        
        Returns:
            str: Generated workflow body code
        """
        executed = set()
        stages_in_dependencies = set()
        
        # Collect all stages from dependencies
        for stage_name, dep_list in self.dependencies.items():
            stages_in_dependencies.add(stage_name)
            stages_in_dependencies.update(dep_list)
        
        # Add auto-generated stages to the list
        for name in self.autogen_stages:
            if not name.endswith("_combined") and not name.endswith("_direct_combine"):
                stages_in_dependencies.add(name)
        
        workflow_code = []
        
        while len(executed) < len(stages_in_dependencies):
            progress_made = False
            
            for stage_name in stages_in_dependencies:
                # Skip if already executed
                if stage_name in executed:
                    continue
                
                # Handle auto-generated stages
                if stage_name in self.autogen_stages:
                    stage_info = self.autogen_stages[stage_name]
                    parent_stages = stage_info.get('dependencies', [])
                    
                    # Check if all dependencies are executed
                    if all(parent in executed for parent in parent_stages):
                        if 'execution_code' in stage_info:
                            workflow_code.append(stage_info['execution_code'])
                        executed.add(stage_name)
                        progress_made = True
                    continue
                
                # Skip if stage doesn't exist (could be a non-existent dependency)
                if stage_name not in self.stages:
                    continue
                
                stage = self.stages[stage_name]
                
                # Case 1: Initial stages (no dependencies) or dependencies are already executed
                if (stage_name in self.dependencies and 
                    (not self.dependencies[stage_name] or 
                     all(dep in executed for dep in self.dependencies[stage_name]))):
                    
                    if stage.is_csv_stage():
                        workflow_code.append(stage.generate_csv_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_path_stage():
                        workflow_code.append(stage.generate_path_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_cst_stage():
                        workflow_code.append(stage.generate_cst_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_dependent_path_stage():
                        # Set up the stage before code generation
                        self.prepare_dependent_path_stage(stage_name, stage)
                        # Generate the code
                        workflow_code.append(stage.generate_dependent_path_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_process_stage():
                        workflow_code.append(self.generate_process_stage_execution(stage_name, stage))
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_view_stage():
                        # Get the parent stage for the view
                        parent_stage_name = self.dependencies[stage_name][0]
                        parent_stage = self.stages[parent_stage_name]
                        
                        # Generate view code
                        workflow_code.append(stage.generate_view_stage(parent_stage))
                        executed.add(stage_name)
                        progress_made = True
                
                # Case 2: Input stages (only in dependency values)
                elif (stage_name not in self.dependencies and 
                      any(stage_name in dep_list for dep_list in self.dependencies.values())):
                    
                    if stage.is_csv_stage():
                        workflow_code.append(stage.generate_csv_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_path_stage():
                        workflow_code.append(stage.generate_path_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
                    elif stage.is_cst_stage():
                        workflow_code.append(stage.generate_cst_loading_stage())
                        executed.add(stage_name)
                        progress_made = True
            
            # If no progress was made in this iteration, we might have a circular dependency
            if not progress_made:
                raise Exception("Error: Circular dependency detected or unable to resolve dependencies")
        
        return "\n".join(workflow_code)
    
    def get_input_channel(self, stage_name):
        """
        Gets the input channel for a stage
        
        Args:
            stage_name (str): Stage name
            
        Returns:
            str: Name of input channel or None if not found
        """
        dep_list = self.dependencies.get(stage_name, [])
        if not dep_list:
            return None
        
        parent_stage_name = dep_list[0]
        parent_stage = self.stages.get(parent_stage_name)
        
        if not parent_stage or not parent_stage.outputs:
            return None
            
        return parent_stage.outputs[0]['name']
    
    def generate_output_format(self, outputs):
        """
        Helper method to generate process output format string
        
        Args:
            outputs (list): List of output configurations
            
        Returns:
            str: Generated output format string
        """
        # Group outputs by mode (separate or tuple)
        separate_outputs = [o for o in outputs if o.get('mode') != 'tuple']
        tuple_outputs = [o for o in outputs if o.get('mode') == 'tuple']
        
        # Group tuple outputs by name
        tuple_output_groups = {}
        for output in tuple_outputs:
            name = output.get('name')
            if name not in tuple_output_groups:
                tuple_output_groups[name] = []
            tuple_output_groups[name].append(output)
        
        output_formats = []
        
        # Process separate outputs - each becomes its own output line
        for output in separate_outputs:
            output_formats.append(self.format_output(output))
        
        # Process tuple output groups - each group becomes ONE output line
        for _, group_outputs in tuple_output_groups.items():
            tuple_components = []
            
            for output in group_outputs:
                if output.get('type') == "path":
                    # Use the first field as the path variable name
                    path_var = output.get('fields', [output.get('value', '').replace(r'[^a-zA-Z0-9_]', '_')])[0] \
                        if output.get('fields') else output.get('value', '').replace(r'[^a-zA-Z0-9_]', '_')
                    tuple_components.append(f"path('{output['value']}')")
                else:
                    for field in output.get('fields', []):
                        tuple_components.append(f"val({field})")
            
            # Only add 'tuple' keyword if more than one component or if explicitly specified
            if (len(tuple_components) > 1 or 
                (len(group_outputs) == 1 and len(group_outputs[0].get('fields', [])) > 1)):
                output_formats.append("tuple " + ", ".join(tuple_components))
            elif len(tuple_components) == 1:
                output_formats.append(tuple_components[0])
        
        return "\n    ".join(output_formats)
    
    def format_output(self, output):
        """
        Helper method to format a single output
        
        Args:
            output (dict): Output configuration
            
        Returns:
            str: Formatted output string
        """
        if output.get('type') == "path":
            # Use the first field as the path variable name
            path_var = output.get('fields', [output.get('value', '').replace(r'[^a-zA-Z0-9_]', '_')])[0] \
                if output.get('fields') else output.get('value', '').replace(r'[^a-zA-Z0-9_]', '_')
            return f"path('{output['value']}')"
        else:
            output_field_formats = [f"val({field})" for field in output.get('fields', [])]
            format_str = ", ".join(output_field_formats)
            if len(output.get('fields', [])) > 1:
                format_str = "tuple " + format_str
            return format_str
    
    def collect_tuple_input_fields(self, input_data, parent_stage):
        """
        Helper method to collect fields information for a tuple input
        
        Args:
            input_data (dict): Input configuration
            parent_stage (WorkflowStage): Parent stage
            
        Returns:
            dict: Map with ordered fields and their types
        """
        ordered_fields = []
        field_types = {}
        
        parent_output_name = input_data.get('output_name')
        
        # Find the matching output in the parent stage
        parent_output = None
        if parent_output_name:
            for output in parent_stage.outputs:
                if output.get('name') == parent_output_name and output.get('mode') == 'tuple':
                    parent_output = output
                    break
        
        if not parent_output:
            # If no specific output name was given or not found, use the first tuple output
            for output in parent_stage.outputs:
                if output.get('mode') == 'tuple':
                    parent_output = output
                    break
        
        if parent_output:
            # Collect all fields from all outputs with the same name, preserving order
            for output in parent_stage.outputs:
                if output.get('name') == parent_output.get('name') and output.get('mode') == 'tuple':
                    for field in output.get('fields', []):
                        if field in input_data.get('fields', []):
                            ordered_fields.append(field)
                            # Set the field type based on the output type
                            field_types[field] = 'path' if output.get('type') == 'path' else 'val'
            
            # Add any remaining fields that weren't found in parent
            for field in input_data.get('fields', []):
                if field not in ordered_fields:
                    ordered_fields.append(field)
                    field_types[field] = input_data.get('field_types', {}).get(field, 'val')
        else:
            # Fallback if no parent output found - use the fields as specified
            for field in input_data.get('fields', []):
                ordered_fields.append(field)
                field_types[field] = input_data.get('field_types', {}).get(field, 'val')
        
        return {'ordered_fields': ordered_fields, 'field_types': field_types}
    
    def generate_input_format(self, inputs):
        """
        Helper method to generate input format string
        
        Args:
            inputs (list): List of input configurations
            
        Returns:
            str: Generated input format string
        """
        input_formats = []
        for input_data in inputs:
            if input_data.get('type') == "path":
                # For path type, use the field name
                if not input_data.get('fields') or len(input_data.get('fields', [])) == 0:
                    # If there's only one path input, use "script" as default name
                    if sum(1 for i in inputs if i.get('type') == "path") == 1:
                        input_formats.append("path(script)")
                    else:
                        raise Exception("Path input requires a field name in 'fields' attribute when multiple path inputs are used")
                else:
                    input_formats.append(f"path({input_data['fields'][0]})")
            elif input_data.get('type') == "tuple":
                # For tuple inputs, handle all fields as part of one tuple
                field_formats = []
                for field in input_data.get('fields', []):
                    field_type = input_data.get('field_types', {}).get(field, 'val')
                    if field_type == "path":
                        field_formats.append(f"path({field})")
                    else:
                        field_formats.append(f"val({field})")
                input_formats.append("tuple " + ", ".join(field_formats))
            else:
                # For regular values, use tuple with val()
                field_formats = [f"val({field})" for field in input_data.get('fields', [])]
                input_formats.append("tuple " + ", ".join(field_formats))
        
        return "\n    ".join(input_formats)
    
    def prepare_process_definition(self, name, stage, input_fields, field_types):
        """
        Helper method to prepare process definition
        
        Args:
            name (str): Process name
            stage (WorkflowStage): Process stage
            input_fields (list): All fields for input tuple
            field_types (dict): Types for input fields
            
        Returns:
            str: Generated process definition code
        """
        # Generate input format with type-aware field formatting
        input_format_parts = []
        for field in input_fields:
            field_type = field_types.get(field, 'val')
            if field_type == "path":
                input_format_parts.append(f"path({field})")
            else:
                input_format_parts.append(f"val({field})")
                
        input_format = "tuple " + ", ".join(input_format_parts)
        
        # Generate output format
        output_formats_str = self.generate_output_format(stage.outputs)

        # Add tag directive if specified
        tag_directive = ""
        if stage.script.get('tag'):
            tag_directive = f"    tag \"${{{stage.script['tag']}}}\"\n"

        # Add errorStrategy directive if specified
        errorStrategy_directive = ""
        if stage.script.get('errorStrategy'):
            errorStrategy_directive = f"    errorStrategy '{stage.script['errorStrategy']}'\n"
        
        # Add container directive if specified
        container_directive = ""
        if stage.script.get('container'):
            container_directive = f"    container = '{stage.script['container']}'\n"
        
        # Add publishDir directive if publish attribute is specified
        publish_directive = ""
        if stage.publish:
            publish_directive = f"    publishDir \"{stage.publish}\", mode: 'copy'\n"
        
        # Shell commands
        shell_commands = "\n    ".join(stage.script.get('shell_commands', []))
        
        # Generate the process definition
        return f"""
// Modified process definition for stage: {name}
process {stage.name} {{
{tag_directive}{errorStrategy_directive}{container_directive}{publish_directive}
    input:
    {input_format}

    output:
    {output_formats_str}

    exec:
    {stage.script.get('exec', '')}

    script:
    \"\"\"
    {shell_commands}
    \"\"\"
}}
"""
    
    def get_parent_output_channel(self, input_data, parent_stage):
        """
        Helper method to get parent output channel
        
        Args:
            input_data (dict): Input configuration
            parent_stage (WorkflowStage): Parent stage
            
        Returns:
            dict: Map with channel name and whether it's a tuple output
        """
        # Get the output channel from parent stage
        parent_channel = None
        parent_output = None
        is_tuple_output = False
        
        if (input_data.get('output_name') and 
            any(o.get('name') == input_data['output_name'] for o in parent_stage.outputs)):
            # Use the specified output by name if it exists
            for output in parent_stage.outputs:
                if output.get('name') == input_data['output_name']:
                    parent_output = output
                    parent_channel = output.get('name')
                    is_tuple_output = output.get('mode') == "tuple"
                    break
        else:
            # Default to first output
            if parent_stage.outputs:
                parent_output = parent_stage.outputs[0]
                parent_channel = parent_output.get('name')
                is_tuple_output = parent_output.get('mode') == "tuple"
        
        return {
            'channel': parent_channel, 
            'is_tuple': is_tuple_output, 
            'output': parent_output
        }
    
    def prepare_multi_input_process_definition(self, name, stage):
        """
        Prepares the process definition for a multi-input process
        This doesn't generate the execution code, just the process definition
        
        Args:
            name (str): Process name
            stage (WorkflowStage): Process stage
        """
        # Collect all field information for inputs, considering tuple types and preserving order
        all_fields = []
        field_types = {}
        
        for input_data in stage.inputs:
            if input_data.get('type') == "tuple":
                # For tuple inputs, we need to preserve the same order as the parent output channel
                parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
                if parent_stage_name in self.stages:
                    parent_stage = self.stages[parent_stage_name]
                    
                    result = self.collect_tuple_input_fields(input_data, parent_stage)
                    
                    # Add fields and types to the overall collections
                    for field in result['ordered_fields']:
                        all_fields.append(field)
                        field_types[field] = result['field_types'].get(field)
            else:
                # For regular inputs
                for field in input_data.get('fields', []):
                    all_fields.append(field)
                    field_types[field] = input_data.get('type', 'val')
        
        # Generate the process definition
        process_code = self.prepare_process_definition(name, stage, all_fields, field_types)
        
        # Store this modified process definition
        self.autogen_stages[f"{stage.name}_combined"] = {
            'type': "process_combined",
            'process_code': process_code,
            'dependencies': []
        }
    
    def prepare_direct_combine_process_definition(self, name, stage):
        """
        Prepares the process definition for a process with direct combine operator
        This doesn't generate the execution code, just the process definition
        
        Args:
            name (str): Process name
            stage (WorkflowStage): Process stage
        """
        # Reuse the same implementation as multi-input process
        self.prepare_multi_input_process_definition(name, stage)
        
        # Just update the key in autogen_stages
        process_code = self.autogen_stages[f"{stage.name}_combined"]['process_code']
        self.autogen_stages.pop(f"{stage.name}_combined")
        
        self.autogen_stages[f"{stage.name}_direct_combine"] = {
            'type': "process_direct_combine",
            'process_code': process_code,
            'dependencies': []
        }
    
    def generate_process_stage_execution(self, name, stage):
        """
        Generates process stage execution code
        
        Args:
            name (str): Stage name
            stage (WorkflowStage): Stage object
            
        Returns:
            str: Generated execution code
        """
        # If the process has 2+ inputs, we'll create an auto-generated combine stage
        if len(stage.inputs) >= 2:
            return self.generate_multi_input_process_execution(name, stage)
        
        # Handle single input process
        input_channels = []
        
        # Detect if we have multiple inputs from the same parent stage
        input_count_by_parent = {}
        
        # Count inputs per parent stage
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            input_count_by_parent[parent_stage_name] = input_count_by_parent.get(parent_stage_name, 0) + 1
        
        # Process each input
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            parent_stage = self.stages.get(parent_stage_name)
            
            # If parent stage doesn't exist, use a default channel name
            if parent_stage is None:
                logging.warning(f"Parent stage {parent_stage_name} not found for input in stage {name}. Using default channel name.")
                input_channels.append(parent_stage_name)
                continue
            
            # Get parent output channel info
            parent_info = self.get_parent_output_channel(input_data, parent_stage)
            parent_channel = parent_info['channel']
            is_tuple_output = parent_info['is_tuple']
            
            # Check if this is a tuple input
            is_tuple_input = input_data.get('type') == "tuple"
            has_multiple_inputs_from_same_parent = input_count_by_parent.get(parent_stage_name, 0) > 1
            
            # Use the parent channel directly if:
            # 1. It's a path input, or
            # 2. It's a tuple input, or
            # 3. The parent output is a tuple, or
            # 4. There are multiple inputs from this same parent (mixed path/value)
            if (input_data.get('type') == "path" or is_tuple_input or 
                is_tuple_output or has_multiple_inputs_from_same_parent):
                # Use the parent channel directly
                input_channels.append(parent_channel)
            else:
                # For regular channel inputs with field selection
                source_fields = self.get_source_fields(parent_stage)
                selected_fields = input_data.get('fields', [])
                
                # Process channel based on if field selection is needed
                if (len(selected_fields) != len(source_fields) or 
                    not all(f in source_fields for f in selected_fields)):
                    input_channels.append(stage.generate_field_selection(parent_channel, selected_fields, source_fields))
                else:
                    input_channels.append(parent_channel)
        
        channel_execution = f"{stage.name}({', '.join(input_channels)})"
        
        # Generate output assignment
        return self.generate_output_assignment(name, stage, channel_execution)
    
    def generate_output_assignment(self, name, stage, channel_execution):
        """
        Helper method to generate output assignment code
        
        Args:
            name (str): Stage name
            stage (WorkflowStage): Stage object
            channel_execution (str): Channel execution expression
            
        Returns:
            str: Generated output assignment code
        """
        # Group outputs by mode (separate or tuple)
        separate_outputs = [o for o in stage.outputs if o.get('mode') != 'tuple']
        tuple_outputs = [o for o in stage.outputs if o.get('mode') == 'tuple']
        
        # Group tuple outputs by name
        tuple_output_groups = {}
        for output in tuple_outputs:
            output_name = output.get('name')
            if output_name not in tuple_output_groups:
                tuple_output_groups[output_name] = []
            tuple_output_groups[output_name].append(output)
        
        # Create a list of all output channel names (one per separate output and one per tuple group)
        output_channel_names = []
        
        # Add separate output names
        for output in separate_outputs:
            output_channel_names.append(output.get('name'))
        
        # Add tuple group names (one name per group)
        for channel_name in tuple_output_groups:
            output_channel_names.append(channel_name)
        
        # Multiple outputs (separate or tuple groups)
        if len(output_channel_names) > 1:
            output_assignment = f"({', '.join(output_channel_names)})"
            
            return f"""
    // Execute {name} with multiple outputs
    {output_assignment} = {channel_execution}
"""
        # Single output (either separate or in a tuple)
        elif len(output_channel_names) == 1:
            return f"""
    // Execute {name}
    {output_channel_names[0]} = {channel_execution}
"""
        else:
            return f"""
    // Execute {name} (no outputs)
    {channel_execution}
"""
    
    def generate_multi_input_process_execution(self, name, stage):
        """
        Generates execution code for a process with multiple inputs
        
        Args:
            name (str): Stage name
            stage (WorkflowStage): Stage object
            
        Returns:
            str: Generated execution code
        """
        # For processes with the combine operator specified, use direct combine operator
        if stage.script.get('operator') == 'combine' and len(stage.inputs) == 2:
            return self.generate_direct_combine_process_execution(name, stage)
        
        # For all other multi-input processes, create a combined input using chain of combine operations
        
        # Detect if we have multiple inputs from the same parent stage
        input_count_by_parent = {}
        
        # Count inputs per parent stage
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            input_count_by_parent[parent_stage_name] = input_count_by_parent.get(parent_stage_name, 0) + 1
        
        # Collect all input channels and their field information
        input_channels_info = []
        
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            parent_stage = self.stages.get(parent_stage_name)
            
            # Handle case where parentStage might be null (e.g., for CSV loader inputs)
            if parent_stage is None:
                logging.warning(f"Parent stage {parent_stage_name} not found for input in stage {name}. Using input fields as is.")
                
                # Create a default channel and fields based on the input definition
                input_channels_info.append({
                    'channel': parent_stage_name,  # Fallback channel name
                    'fields': input_data.get('fields', [])
                })
                continue
            
            # Get parent output channel info
            parent_info = self.get_parent_output_channel(input_data, parent_stage)
            parent_channel = parent_info['channel']
            is_tuple_output = parent_info['is_tuple']
            
            # Check if this is a tuple input
            is_tuple_input = input_data.get('type') == "tuple"
            has_multiple_inputs_from_same_parent = input_count_by_parent.get(parent_stage_name, 0) > 1
            
            processed_channel = None
            
            # Use the parent channel directly if:
            # 1. It's a path input, or
            # 2. It's a tuple input, or
            # 3. The parent output is a tuple, or
            # 4. There are multiple inputs from this same parent (mixed path/value)
            if (input_data.get('type') == "path" or is_tuple_input or 
                is_tuple_output or has_multiple_inputs_from_same_parent):
                # Use the parent channel directly
                processed_channel = parent_channel
            else:
                # For regular channel inputs
                source_fields = self.get_source_fields(parent_stage)
                selected_fields = input_data.get('fields', [])
                
                # Process channel based on if field selection is needed
                if (len(selected_fields) != len(source_fields) or 
                    not all(f in source_fields for f in selected_fields)):
                    processed_channel = stage.generate_field_selection(parent_channel, selected_fields, source_fields)
                else:
                    processed_channel = parent_channel
            
            input_channels_info.append({
                'channel': processed_channel,
                'fields': input_data.get('fields', [])
            })
        
        # Make sure we have at least one input channel
        if not input_channels_info:
            raise Exception(f"No valid input channels found for stage {name}")
        
        # Use a chain of combine operations to merge all inputs
        combined_channel = input_channels_info[0]['channel']
        
        # Generate the combination of all inputs using sequential combine operations
        for i in range(1, len(input_channels_info)):
            current_input = input_channels_info[i]
            
            # Define the combine operation
            combined_channel = f"{combined_channel}.combine({current_input['channel']})"
        
        # Generate output assignment
        return self.generate_output_assignment(name, stage, f"{stage.name}({combined_channel})")
    
    def generate_direct_combine_process_execution(self, name, stage):
        """
        Generates execution code for a process with direct combine operator
        
        Args:
            name (str): Stage name
            stage (WorkflowStage): Stage object
            
        Returns:
            str: Generated execution code
        """
        input_channels = []
        
        # Detect if we have multiple inputs from the same parent stage
        input_count_by_parent = {}
        
        # Count inputs per parent stage
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            input_count_by_parent[parent_stage_name] = input_count_by_parent.get(parent_stage_name, 0) + 1
        
        # Process each input to get the appropriate channels
        for input_data in stage.inputs:
            # Get parent stage
            parent_stage_name = input_data.get('from') or self.dependencies.get(name, [])[0]
            parent_stage = self.stages.get(parent_stage_name)
            
            # Handle case where parentStage might be null (e.g., for CSV loader inputs)
            if parent_stage is None:
                logging.warning(f"Parent stage {parent_stage_name} not found for input in stage {name}. Using input fields as is.")
                
                # Use a default channel name based on parent stage name
                input_channels.append(parent_stage_name)
                continue
            
            # Get parent output channel info
            parent_info = self.get_parent_output_channel(input_data, parent_stage)
            parent_channel = parent_info['channel']
            is_tuple_output = parent_info['is_tuple']
            
            # Check if this is a tuple input
            is_tuple_input = input_data.get('type') == "tuple"
            has_multiple_inputs_from_same_parent = input_count_by_parent.get(parent_stage_name, 0) > 1
            
            # Use the parent channel directly if:
            # 1. It's a path input, or
            # 2. It's a tuple input, or
            # 3. The parent output is a tuple, or
            # 4. There are multiple inputs from this same parent (mixed path/value)
            if (input_data.get('type') == "path" or is_tuple_input or 
                is_tuple_output or has_multiple_inputs_from_same_parent):
                # Use the parent channel directly
                input_channels.append(parent_channel)
            else:
                # For regular channel inputs
                source_fields = self.get_source_fields(parent_stage)
                selected_fields = input_data.get('fields', [])
                
                # Create field selection if needed
                if (len(selected_fields) != len(source_fields) or 
                    not all(f in source_fields for f in selected_fields)):
                    input_channels.append(stage.generate_field_selection(parent_channel, selected_fields, source_fields))
                else:
                    input_channels.append(parent_channel)
        
        # Make sure we have at least two input channels for the combine operation
        if len(input_channels) < 2:
            raise Exception(f"Need at least two input channels for combine operation in stage {name}")
        
        # Generate the combine operation
        channel_combination = f"{input_channels[0]}.combine({input_channels[1]})"
        
        # Generate output assignment
        return self.generate_output_assignment(name, stage, f"{stage.name}({channel_combination})")
    
    def get_source_fields(self, parent_stage):
        """
        Helper method to get source fields from a parent stage
        
        Args:
            parent_stage (WorkflowStage): Parent stage
            
        Returns:
            list: List of field names
        """
        if not parent_stage:
            return []
        
        # For other stage types, collect all fields from all outputs
        all_fields = set()
        
        for output in parent_stage.outputs:
            if output.get('fields'):
                all_fields.update(output.get('fields', []))
        
        return list(all_fields)
    
    def prepare_dependent_path_stage(self, stage_name, stage):
        """
        Prepares a dependent path stage by updating its input configuration
        with the actual parent channel name and validating the field exists in parent outputs
        
        Args:
            stage_name (str): Stage name
            stage (WorkflowStage): Stage object
        """
        input_data = stage.inputs[0]
        
        # Get the parent stage name from dependencies
        parent_stage_name = self.dependencies.get(stage_name, [])[0]
        parent_stage = self.stages.get(parent_stage_name)
        parent_output_channel = parent_stage.outputs[0].get('name')
        
        # Update the 'from' in the input to be the actual parent output channel name
        input_data['from'] = parent_output_channel
        
        # If a field is specified, validate it exists in the parent stage's outputs
        if input_data.get('field'):
            field_exists = False
            field_index = -1
            
            # Check if the field exists in the parent stage's outputs and get its index
            parent_fields = parent_stage.outputs[0].get('fields', [])
            if parent_fields and input_data.get('field') in parent_fields:
                field_exists = True
                field_index = parent_fields.index(input_data.get('field'))
            
            if not field_exists:
                raise Exception(f"Field '{input_data.get('field')}' specified in dependent pathloader stage '{stage_name}' "
                              f"does not exist in parent stage '{parent_stage_name}' outputs. "
                              f"Available fields: {parent_fields}")
            
            # Store the field index for use in code generation
            input_data['field_index'] = field_index
            
        elif parent_stage.outputs[0].get('fields') and parent_stage.outputs[0].get('fields'):
            # If no field is specified but parent has fields, use the first field by default
            parent_fields = parent_stage.outputs[0]['fields']
            logging.warning(f"No field specified for dependent pathloader stage '{stage_name}'. "
                          f"Using first available field '{parent_fields[0]}' from parent stage.")
            input_data['field'] = parent_fields[0]
            input_data['field_index'] = 0