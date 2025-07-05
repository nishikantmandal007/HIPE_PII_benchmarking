import argparse
import os
import sys
import json
import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='HIPE-2022 Dataset Transformation for PII Detection Workflow')
    parser.add_argument('--data_dir', required=True, help='Directory containing HIPE-2022 data files')
    parser.add_argument('--output_dir', default='output', help='Base directory for output files')
    parser.add_argument('--words_per_page', type=int, default=100, help='Number of words per page')
    parser.add_argument('--timestamp', default='2025-07-02 17:54:48', help='Timestamp for output files')
    parser.add_argument('--validate', action='store_true', help='Run validation after processing')
    args = parser.parse_args()
    
    # Import modules here to avoid issues if modules are not in path
    try:
        from dataset_loader import load_hipe_data
        from entity_extraction import extract_text_and_entities, validate_extracted_entities
        from document_segmentation import segment_documents
        from file_generation import generate_files
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all required Python modules are in your path or current directory")
        sys.exit(1)
    
    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist or is not a directory")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp
    timestamp = args.timestamp
    
    # Create log file
    log_path = os.path.join(output_dir, "process_log.txt")
    
    # Setup logging to both console and file
    def log_message(message):
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    
    # Start logging
    log_message(f"=== HIPE-2022 Dataset Transformation ===")
    log_message(f"Started: {timestamp}")
    log_message(f"User: nishikantmandal007")
    log_message(f"Data directory: {os.path.abspath(args.data_dir)}")
    log_message(f"Output directory: {output_dir}")
    log_message(f"Words per page: {args.words_per_page}")
    log_message("")
    
    # Phase 1: Dataset Analysis
    log_message("Phase 1: Loading and analyzing dataset...")
    dataset, dataset_statistics = load_hipe_data(args.data_dir)
    
    if dataset.empty:
        log_message("Error: No data loaded. Please check the data directory path.")
        sys.exit(1)
    
    log_message(f"Loaded dataset with {len(dataset)} tokens across {len(dataset_statistics['language_counts'])} languages")
    log_message(f"Dataset aliases: {', '.join(dataset_statistics['alias_counts'].keys())}")
    
    # Phase 2 & 3: Entity Extraction and Document Segmentation per alias/language/split
    log_message("\nPhase 2 & 3: Extracting text and entities, and segmenting documents...")
    
    total_processed_documents = 0
    total_generated_chunks = 0
    
    # Get unique combinations of alias, language, and split
    unique_combinations = dataset[['dataset_alias', 'language', 'split']].drop_duplicates().values.tolist()
    
    for alias, lang, split in unique_combinations:
        log_message(f"\nProcessing combination: Alias={alias}, Language={lang}, Split={split}")
        
        # Filter dataset for the current combination
        filtered_dataset = dataset[(dataset['dataset_alias'] == alias) & 
                                   (dataset['language'] == lang) & 
                                   (dataset['split'] == split)]
        
        # Extract text and entities for the filtered dataset
        documents, _ = extract_text_and_entities(filtered_dataset)
        total_processed_documents += len(documents)
        log_message(f"  Extracted {len(documents)} documents.")
        
        # Segment documents into chunks
        chunks = segment_documents(documents, args.words_per_page, alias, lang, split)
        total_generated_chunks += len(chunks)
        log_message(f"  Created {len(chunks)} chunks (2 pages each).")
        
        # Phase 4: File Generation for the current combination
        log_message(f"\nPhase 4: Generating files for Alias={alias}, Language={lang}, Split={split}...")
        generate_files(chunks, output_dir)
    
    # Phase 5: Finalizing
    log_message("\nPhase 5: Finalizing...")
    
    # Phase 6: Validation (if enabled)
    if args.validate:
        log_message("\nPhase 6: Validation...")
        log_message("Running validation to compare extracted entities with original dataset...")
        
        # Load the original dataset again to ensure we have all data
        full_dataset = load_hipe_data(args.data_dir)
        
        # Extract all documents with entities in one go for validation
        all_documents, _ = extract_text_and_entities(full_dataset)
        
        # Run comprehensive validation across all file types (docsjson, srtjson, logjson)
        try:
            # Import validation modules
            from entity_validation import run_validation, display_validation_results
            
            # Validate docsjson files
            docsjson_dir = os.path.join(output_dir, "docsjson")
            log_message(f"Validating DOCS JSON files in {docsjson_dir}...")
            entity_stats, docsjson_stats = run_validation(full_dataset, all_documents, docsjson_dir)
            
            # Validate srtjson files
            srtjson_dir = os.path.join(output_dir, "srtjson")
            if os.path.exists(srtjson_dir):
                log_message(f"Validating SRT JSON files in {srtjson_dir}...")
                _, srtjson_stats = run_validation(full_dataset, all_documents, srtjson_dir)
            else:
                log_message(f"SRT JSON directory not found: {srtjson_dir}")
                srtjson_stats = {"error": f"Directory {srtjson_dir} not found"}
            
            # Validate logjson files
            logjson_dir = os.path.join(output_dir, "logjson")
            if os.path.exists(logjson_dir):
                log_message(f"Validating LOG JSON files in {logjson_dir}...")
                _, logjson_stats = run_validation(full_dataset, all_documents, logjson_dir)
            else:
                log_message(f"LOG JSON directory not found: {logjson_dir}")
                logjson_stats = {"error": f"Directory {logjson_dir} not found"}
            
            # Combine the stats for a comprehensive report
            json_stats = {
                "docsjson": docsjson_stats,
                "srtjson": srtjson_stats,
                "logjson": logjson_stats
            }
            
            # Display the combined results
            log_message("\n==== Combined Validation Results ====")
            display_validation_results(entity_stats, json_stats)
            
            # Save validation results
            validation_dir = os.path.join(output_dir, "validation")
            os.makedirs(validation_dir, exist_ok=True)
            
            validation_results_path = os.path.join(validation_dir, "validation_results.json")
            with open(validation_results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'entity_stats': entity_stats,
                    'json_stats': json_stats
                }, f, ensure_ascii=False, indent=2)
            
            log_message(f"Validation results saved to {validation_results_path}")
            
            # Log validation summary
            log_message(f"Original dataset total entities: {entity_stats['original']['total_entities']}")
            log_message(f"Processed dataset total entities: {entity_stats['processed']['total_entities']}")
            entity_diff = entity_stats['processed']['total_entities'] - entity_stats['original']['total_entities']
            log_message(f"Entity count difference: {entity_diff:+d} ({(entity_diff/max(1, entity_stats['original']['total_entities'])*100):.2f}%)")
        
        except Exception as e:
            log_message(f"Error during validation: {str(e)}")
    
    # Final summary
    log_message("\n=== Processing Summary ===")
    log_message(f"Total processed documents: {total_processed_documents}")
    log_message(f"Total generated chunks: {total_generated_chunks}")
    log_message(f"Output files can be found in: {output_dir}")
    log_message(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()