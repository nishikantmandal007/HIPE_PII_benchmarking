#!/usr/bin/env python3
"""
Comprehensive validation script that validates entity extraction and JSON files for all file types
(docsjson, srtjson, logjson). This script can be run from any directory.
"""

import os
import sys
import json
import argparse

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Validate entity extraction across all file types')
    parser.add_argument('--base-dir', default=os.path.join(os.path.dirname(__file__), 'output'),
                      help='Base output directory containing docsjson, srtjson, and logjson subdirectories')
    parser.add_argument('--json-dir', default=None,
                      help='(Optional) Specific JSON directory to validate. If not provided, all three types will be validated.')
    parser.add_argument('--save-results', action='store_true',
                      help='Save validation results to JSON file')
    parser.add_argument('--results-path', 
                      default=os.path.join(project_root, 'validation', 'results.json'),
                      help='Path to save validation results')
    parser.add_argument('--data-dir',
                      default=os.path.join(project_root, 'HIPE-2022-data'),
                      help='Path to the HIPE-2022 data directory')
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output for debugging')
    
    args = parser.parse_args()
    
    try:
        # Now import the modules (after path setup)
        from DatasetTransformation.dataset_loader import load_hipe_data
        from DatasetTransformation.entity_extraction import extract_text_and_entities
        from DatasetTransformation.entity_validation import run_validation, display_validation_results
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print(f"Current Python path: {sys.path}")
        sys.exit(1)
    
    print("Loading HIPE dataset...")
    if args.verbose:
        print(f"Loading from: {args.data_dir}")
    original_dataset, _ = load_hipe_data(args.data_dir)
    
    print("Extracting text and entities...")
    processed_documents, _ = extract_text_and_entities(original_dataset)
    
    # If specific json-dir is provided, only validate that directory
    if args.json_dir:
        print(f"Running validation against JSON files in {args.json_dir}...")
        if args.verbose:
            print(f"Checking if directory exists: {os.path.exists(args.json_dir)}")
            if os.path.exists(args.json_dir):
                print(f"Directory contents: {os.listdir(args.json_dir)}")
        entity_stats, json_stats = run_validation(original_dataset, processed_documents, args.json_dir)
    else:
        # Otherwise validate all three file types
        print(f"Running validation across all JSON file types in {args.base_dir}...")
        
        # Validate docsjson files
        docsjson_dir = os.path.join(args.base_dir, "docsjson")
        if os.path.exists(docsjson_dir):
            print(f"\nValidating DOCS JSON files in {docsjson_dir}...")
            entity_stats, docsjson_stats = run_validation(original_dataset, processed_documents, docsjson_dir)
        else:
            print(f"\nDOCS JSON directory not found: {docsjson_dir}")
            docsjson_stats = {"error": f"Directory {docsjson_dir} not found"}
            
        # Validate srtjson files
        srtjson_dir = os.path.join(args.base_dir, "srtjson")
        if os.path.exists(srtjson_dir):
            print(f"\nValidating SRT JSON files in {srtjson_dir}...")
            # We reuse entity_stats but get new json stats
            _, srtjson_stats = run_validation(original_dataset, processed_documents, srtjson_dir)
        else:
            print(f"\nSRT JSON directory not found: {srtjson_dir}")
            srtjson_stats = {"error": f"Directory {srtjson_dir} not found"}
        
        # Validate logjson files
        logjson_dir = os.path.join(args.base_dir, "logjson")
        if os.path.exists(logjson_dir):
            print(f"\nValidating LOG JSON files in {logjson_dir}...")
            # We reuse entity_stats but get new json stats
            _, logjson_stats = run_validation(original_dataset, processed_documents, logjson_dir)
        else:
            print(f"\nLOG JSON directory not found: {logjson_dir}")
            logjson_stats = {"error": f"Directory {logjson_dir} not found"}
        
        # Combine the stats for a comprehensive report
        json_stats = {
            "docsjson": docsjson_stats,
            "srtjson": srtjson_stats,
            "logjson": logjson_stats
        }
        
        # Display the combined results
        print("\n==== Combined Validation Results ====")
        display_validation_results(entity_stats, json_stats)
    
    if args.save_results:
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        with open(args.results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'entity_stats': entity_stats,
                'json_stats': json_stats
            }, f, ensure_ascii=False, indent=2)
        print(f"Validation results saved to {args.results_path}")

if __name__ == "__main__":
    main()
