#!/usr/bin/env python3

class WorkflowStage:
    """
    Represents a stage in the workflow.
    Handles stage-specific operations and code generation.
    """
    
    def __init__(self, stage_data):
        """
        Constructor
        
        Args:
            stage_data (dict): Map containing stage configuration
        """
        self.name = stage_data.get('name')
        self.type = stage_data.get('type')
        self.inputs = stage_data.get('inputs', [])
        self.outputs = stage_data.get('outputs', [])
        self.script = stage_data.get('script', {})
        self.view_format = stage_data.get('view_format')
        self.publish = stage_data.get('publish')
    
    def is_csv_stage(self):
        """Check if this is a CSV loading stage"""
        return self.type == 'csvloader'
    
    def is_view_stage(self):
        """Check if this is a view stage"""
        return self.type == 'view'
    
    def is_process_stage(self):
        """Check if this is a process stage"""
        return self.type == 'process'
    
    def is_path_stage(self):
        """Check if this is a path loading stage"""
        return self.type == 'pathloader'
    
    def is_cst_stage(self):
        """Check if this is a comma-separated text loading stage"""
        return self.type == 'cstloader'
    
    def is_dependent_path_stage(self):
        """Check if this is a dependent path loading stage"""
        return self.type == 'dependentpathloader'
    
    def generate_csv_loading_stage(self):
        """
        Generates code for CSV loading stage
        
        Returns:
            str: Generated Nextflow code
        """
        input_data = self.inputs[0]
        output = self.outputs[0]
        
        field_join = ', row.'.join(output['fields'])
        
        return f"""
    // Execute {self.name}
    {output['name']} = Channel
        .fromPath('{input_data['value']}', checkIfExists: true)
        .splitCsv(header: true)
        .map {{ row -> [
            row.{field_join}
        ]}}
"""
    
    def generate_cst_loading_stage(self):
        """
        Generates code for comma-separated text loading stage
        
        Returns:
            str: Generated Nextflow code
        """
        input_data = self.inputs[0]
        output = self.outputs[0]
        
        return f"""
    // Execute {self.name}
    {output['name']} = Channel
        .fromPath('{input_data['value']}', checkIfExists: true)
        .map {{ file -> 
            file.text.split(',').collect {{ it.trim() }}
        }}
"""
    
    def generate_path_loading_stage(self):
        """
        Generates code for path loading stage
        
        Returns:
            str: Generated Nextflow code
        """
        input_data = self.inputs[0]
        
        return f"""
    // Execute {self.name}
    {self.outputs[0]['name']} = Channel
        .fromPath('{input_data['value']}', checkIfExists: true)
"""
    
    def generate_dependent_path_loading_stage(self):
        """
        Generates code for dependent path loading stage which loads files 
        based on values from another channel
        
        Returns:
            str: Generated Nextflow code
        """
        input_data = self.inputs[0]
        output = self.outputs[0]
        
        # Get the pattern for the file path
        path_pattern = input_data['pattern']
        
        # Get the parent channel reference
        parent_channel_ref = input_data['from']
        
        # Get the field to use from the parent channel
        source_field = input_data['field']
        
        # Get the field index from the parent stage (this should be set during preparation)
        field_index = input_data.get('field_index', 0)
        
        # Replace placeholders in the pattern
        formatted_pattern = path_pattern.replace(f"{{{source_field}}}", f"${{{source_field}}}")
        
        return f"""
    // Execute {self.name} - Dependent Path Loading
    {output['name']} = {parent_channel_ref}
        .map {{ row ->
            // Extract the field value by index
            {source_field} = row[{field_index}]
            return {source_field}
        }}
        .collect()
        .map {{ {source_field}List ->
            {source_field}List.collect {{ {source_field} ->
                // Create file path using the field value
                def filePath = "{formatted_pattern}"
                return tuple({source_field}, file(filePath))
            }}
        }}
        .flatten()
        .buffer(size: 2)
"""
    
    def generate_process_stage(self):
        """
        Generates code for process stage
        
        Returns:
            str: Generated Nextflow code
        """
        # Generate input format
        input_formats = []
        for input_data in self.inputs:
            if input_data.get('type') == "path":
                # Handle path type inputs
                if not input_data.get('fields') or len(input_data.get('fields', [])) == 0:
                    # If there's only one path input, use "script" as default name
                    if sum(1 for i in self.inputs if i.get('type') == "path") == 1:
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
        
        input_format_str = "\n    ".join(input_formats)
        
        # Generate output format
        output_formats = []
        
        # Group outputs by mode (separate or tuple)
        separate_outputs = [o for o in self.outputs if o.get('mode') != 'tuple']
        tuple_outputs = [o for o in self.outputs if o.get('mode') == 'tuple']
        
        # Group tuple outputs by name
        tuple_output_groups = {}
        for output in tuple_outputs:
            name = output.get('name')
            if name not in tuple_output_groups:
                tuple_output_groups[name] = []
            tuple_output_groups[name].append(output)
        
        # Process separate outputs
        for output in separate_outputs:
            if output.get('type') == "path":
                output_formats.append(f"path('{output['value']}')")
            else:
                output_field_formats = [f"val({field})" for field in output.get('fields', [])]
                format_str = ", ".join(output_field_formats)
                if len(output.get('fields', [])) > 1:
                    format_str = "tuple " + format_str
                output_formats.append(format_str)
        
        # Process tuple output groups
        for group_name, group_outputs in tuple_output_groups.items():
            tuple_components = []
            
            for output in group_outputs:
                if output.get('type') == "path":
                    tuple_components.append(f"path('{output['value']}')")
                else:
                    for field in output.get('fields', []):
                        tuple_components.append(f"val({field})")
            
            # Only add 'tuple' keyword if necessary
            if len(tuple_components) > 1 or (len(group_outputs) == 1 and len(group_outputs[0].get('fields', [])) > 1):
                output_formats.append("tuple " + ", ".join(tuple_components))
            elif len(tuple_components) == 1:
                output_formats.append(tuple_components[0])
        
        output_format_str = "\n    ".join(output_formats)
        
        # Modify exec statement
        modified_exec = self.modify_exec_statement(self.script.get('exec', ''), self.inputs)
        
        # Container directive
        container_directive = ""
        if self.script.get('container'):
            container_directive = f"    container = '{self.script['container']}'\n"
        
        # Publish directive
        publish_directive = ""
        if self.publish:
            publish_directive = f"    publishDir \"{self.publish}\", mode: 'copy'\n"
        
        shell_commands = "\n    ".join(self.script.get('shell_commands', []))
        
        return f"""
// Stage: {self.name}
process {self.name} {{
{container_directive}{publish_directive}
    input:
    {input_format_str}

    output:
    {output_format_str}

    exec:
    {modified_exec}

    script:
    \"""
    {shell_commands}
    \"""
}}
"""
    
    def modify_exec_statement(self, exec_statement, inputs):
        """
        Modifies exec statement to handle numeric operations
        
        Args:
            exec_statement (str): Original exec statement
            inputs (list): List of input configurations
            
        Returns:
            str: Modified exec statement
        """
        if not exec_statement:
            return ""
        
        parts = exec_statement.split('=', 1)
        if len(parts) != 2:
            return exec_statement
        
        output_var = parts[0].strip()
        expression = parts[1].strip()
        
        # Convert input variables to numeric (skip path types)
        for input_data in inputs:
            if input_data.get('type') != "path":
                for field in input_data.get('fields', []):
                    if field in expression:
                        expression = expression.replace(field, f"{field}.toDouble()")
        
        return f"{output_var} = {expression}"
    
    def generate_view_stage(self, parent_stage):
        """
        Generates code for view stage
        
        Args:
            parent_stage (WorkflowStage): Parent stage that provides input to the view
            
        Returns:
            str: Generated Nextflow code
        """
        view_code = []
        
        # Get all separate outputs from parent stage
        separate_outputs = [o for o in parent_stage.outputs if o.get('mode') != 'tuple']
        
        # Get all tuple output groups from parent stage
        tuple_outputs = [o for o in parent_stage.outputs if o.get('mode') == 'tuple']
        
        # Group tuple outputs by name
        tuple_output_groups = {}
        for output in tuple_outputs:
            name = output.get('name')
            if name not in tuple_output_groups:
                tuple_output_groups[name] = []
            tuple_output_groups[name].append(output)
        
        # If there are multiple outputs, generate view for each
        if len(separate_outputs) + len(tuple_output_groups) > 1:
            # Process separate outputs
            for output in separate_outputs:
                view_code.append(f"""
    // Execute {self.name} - View results for {output['name']}
    {output['name']}.view({self.view_format})
""")
            
            # Process tuple output groups
            for channel_name in tuple_output_groups:
                view_code.append(f"""
    // Execute {self.name} - View results for {channel_name}
    {channel_name}.view({self.view_format})
""")
        
        # Single output
        elif len(separate_outputs) == 1:
            output_name = separate_outputs[0]['name']
            view_code.append(f"""
    // Execute {self.name} - View results
    {output_name}.view({self.view_format})
""")
        
        elif len(tuple_output_groups) == 1:
            output_name = list(tuple_output_groups.keys())[0]
            view_code.append(f"""
    // Execute {self.name} - View results
    {output_name}.view({self.view_format})
""")
        
        return "".join(view_code)
    
    def generate_field_selection(self, input_channel, selected_fields, source_fields):
        """
        Generates field selection code for channel operations
        
        Args:
            input_channel (str): Name of input channel
            selected_fields (list): List of fields to select
            source_fields (list): List of available source fields
            
        Returns:
            str: Generated field selection code
        """
        # If all fields are selected in the same order, no need for mapping
        if selected_fields == source_fields:
            return input_channel
        
        # Get indices of selected fields in source fields
        field_indices = []
        for field in selected_fields:
            index = source_fields.index(field) if field in source_fields else -1
            field_indices.append(index)
        
        # Handle case where some fields are not found in source fields
        if -1 in field_indices:
            raise Exception(f"Selected fields {selected_fields} not found in source fields {source_fields}")
        
        index_selection = ", ".join([f"row[{idx}]" for idx in field_indices])
        return f"{input_channel}.map {{ row -> tuple({index_selection}) }}"