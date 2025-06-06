#!/usr/bin/env python3

import os
import json
import argparse
import logging
from libs.workflow_generator import WorkflowGenerator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Nextflow workflow from JSON definition')
    
    parser.add_argument('--json', type=str, default="configs/synthdist.json",
                        help='JSON configuration file (default: configs/synthdist.json)')
    parser.add_argument('--outfile', type=str, default="run_synthdist.nf",
                        help='Output file for the generated workflow (default: run_synthdist.nf)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log parameters
    logging.info("="*60)
    logging.info("Workflow Generator")
    logging.info("="*60)
    logging.info(f"JSON Config : {args.json}")
    logging.info(f"Output File: {args.outfile}")
    logging.info("="*60)
    
    # Validate input file exists
    if not os.path.exists(args.json):
        error_msg = f"""
        ERROR: JSON specification file not found: {args.json}
        Please provide a valid JSON file using --json parameter
        
        Example usage:
        python generate_workflow.py --json configs/workflow_config.json --outfile output.nf
        """
        logging.error(error_msg)
        exit(1)
    
    try:
        # Parse the workflow JSON
        with open(args.json, 'r') as json_file:
            workflow_json = json.load(json_file)
        
        # Create a WorkflowGenerator instance
        generator = WorkflowGenerator(workflow_json)
        
        # Generate the Nextflow workflow code
        workflow_code = generator.generate()
        
        # Write generated workflow to file
        with open(args.outfile, 'w') as output_file:
            output_file.write(workflow_code)
        
        logging.info("="*60)
        logging.info("Workflow Generation Complete")
        logging.info("="*60)
        logging.info(f"Generated workflow has been written to: {args.outfile}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error generating workflow: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()