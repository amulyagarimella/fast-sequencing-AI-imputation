import os
import json
from glob import glob

def collect_parameters_and_errors(root_dir):
    """Collect all parameters.txt and .err files into a structured JSON"""
    results = {}
    
    # Find all parameters.txt files excluding tmp directories
    param_files = glob(os.path.join(root_dir, '**/parameters.txt'), recursive=True)
    err_files = glob(os.path.join(root_dir, '**/*.err'), recursive=True)
    
    # Filter out files in tmp directories
    param_files = [f for f in param_files if '/tmp/' not in f]
    err_files = [f for f in err_files if '/tmp/' not in f]
    
    # Process parameters files
    for param_file in param_files:
        dir_path = os.path.dirname(param_file)
        results.setdefault(dir_path, {})['parameters'] = read_parameters(param_file)
    
    # Process error files
    for err_file in err_files:
        dir_path = os.path.dirname(err_file)
        results.setdefault(dir_path, {})['gtime'] = read_error_file(err_file)
    
    return results

def read_parameters(param_file):
    """Read parameters.txt file into a dictionary"""
    params = {}
    with open(param_file) as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = value.strip()
    return params

def read_error_file(err_file):
    """Read .err file and parse gtime output into a structured dictionary"""
    gtime_data = {}
    with open(err_file) as f:
        lines = f.readlines()
        # Skip the first line (command line) and process gtime output
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                # Split on exactly three spaces
                if '   ' in line:
                    value, key = line.split('   ', 1)
                    value = value.strip()
                    key = key.strip()
                    try:
                        # Convert numeric values to appropriate types
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
                    gtime_data[key] = value
    return gtime_data

def save_to_json(data, output_path):
    """Save collected data to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect parameters and error files into JSON')
    parser.add_argument('--input-dir', required=True, help='Root directory to search')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    data = collect_parameters_and_errors(args.input_dir)
    save_to_json(data, args.output)
