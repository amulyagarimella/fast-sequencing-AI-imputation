import os
import json
from glob import glob

def combine_json_files(directory, output_file):
    """
    Combine all JSON files in a directory into a single JSON file with structured format
    """
    # Find all JSON files (excluding .DS_Store)
    json_files = [f for f in glob(os.path.join(directory, '*.json')) if not f.endswith('.DS_Store')]
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
    
    combined_data = {}
    
    # First process parameters.json to get the base structure
    for json_file in json_files:
        if json_file.endswith('parameters.json'):
            with open(json_file, 'r') as f:
                params = json.load(f)
                # Use directory name as key
                dir_path = os.path.dirname(json_file)
                combined_data[dir_path] = params
    
    # Now process metrics files
    for json_file in json_files:
        if 'metrics' in json_file:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                
                # Get the base cool file
                base_file = metrics['base_file']
                
                # For each comparison
                for cool_file, metrics_data in metrics['comparisons'].items():
                    # Get the directory of the cool file
                    cool_dir_path = os.path.dirname(cool_file)
        
                    combined_data[dir_path][cool_dir_path][json_file] = {}
                    combined_data[dir_path][cool_dir_path][json_file][base_file] = metrics_data
    
    # Save combined data
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined JSON files saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine multiple JSON files into one')
    parser.add_argument('--input-dir', required=True, help='Directory containing JSON files')
    parser.add_argument('--output', required=True, help='Output combined JSON file path')
    
    args = parser.parse_args()
    combine_json_files(args.input_dir, args.output)
