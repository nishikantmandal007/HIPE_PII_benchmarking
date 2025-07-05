import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
import glob

def validate_entity_extraction(original_dataset, processed_documents):
    """
    Compare person entities in original dataset with those in processed documents
    
    Returns:
        dict: Statistics comparing original and processed entities
    """
    # Count entities in original dataset
    original_entities = Counter()
    original_entity_docs = Counter()
    
    # Count by dataset alias and language
    original_by_source = {}
    
    # Group by document
    doc_groups = original_dataset.groupby(['dataset_alias', 'language', 'doc_id'])
    
    print("Analyzing original dataset...")
    for (dataset_alias, lang, doc_id), doc_df in tqdm(doc_groups):
        doc_key = f"{dataset_alias}/{lang}/{doc_id}"
        
        # Determine the correct NE tag column and tags based on dataset alias
        if dataset_alias in ['ajmc', 'hipe2020', 'letemps']:
            ne_tag_col = 'NE-COARSE-LIT'
            b_tag = 'B-pers'
            i_tag = 'I-pers'
        else:  # newseye, sonar, topres19th
            ne_tag_col = 'NE-COARSE-LIT'
            b_tag = 'B-PER'
            i_tag = 'I-PER'
            
        # Extract person entities from document
        current_entity = []
        entities = []
        
        for idx, row in doc_df.iterrows():
            token = row.get('token', '')
            ne_tag = row.get(ne_tag_col, 'O')
            
            if pd.isna(ne_tag) or ne_tag == '_':
                ne_tag = 'O'
                
            if ne_tag.upper() == b_tag.upper():
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    entities.append(entity_text)
                    current_entity = []
                current_entity.append(token)
            elif ne_tag.upper() == i_tag.upper() and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    entities.append(entity_text)
                    current_entity = []
        
        # Add any remaining entity
        if current_entity:
            entity_text = ' '.join(current_entity)
            entities.append(entity_text)
            
        # Count entities in this document
        doc_entities = Counter(entities)
        original_entities.update(doc_entities)
        
        # Count documents containing each entity
        for entity in set(entities):
            original_entity_docs[entity] += 1
            
        # Track by source
        source_key = f"{dataset_alias}/{lang}"
        if source_key not in original_by_source:
            original_by_source[source_key] = {
                'total_entities': 0,
                'unique_entities': 0,
                'documents': set()
            }
        
        original_by_source[source_key]['total_entities'] += len(entities)
        original_by_source[source_key]['unique_entities'] += len(doc_entities)
        original_by_source[source_key]['documents'].add(doc_id)
    
    # Count entities in processed documents
    processed_entities = Counter()
    processed_entity_docs = Counter()
    processed_by_source = {}
    
    print("Analyzing processed documents...")
    for doc in tqdm(processed_documents):
        dataset_alias = doc.get('dataset_alias', 'unknown')
        lang = doc.get('language', 'unknown')
        doc_id = doc.get('doc_id', 'unknown')
        
        doc_key = f"{dataset_alias}/{lang}/{doc_id}"
        source_key = f"{dataset_alias}/{lang}"
        
        entities = [entity['text'] for entity in doc.get('entities', [])]
        
        # Count entities in this document
        doc_entities = Counter(entities)
        processed_entities.update(doc_entities)
        
        # Count documents containing each entity
        for entity in set(entities):
            processed_entity_docs[entity] += 1
            
        # Track by source
        if source_key not in processed_by_source:
            processed_by_source[source_key] = {
                'total_entities': 0,
                'unique_entities': 0,
                'documents': set()
            }
        
        processed_by_source[source_key]['total_entities'] += len(entities)
        processed_by_source[source_key]['unique_entities'] += len(doc_entities)
        processed_by_source[source_key]['documents'].add(doc_id)
    
    # Compute statistics
    stats = {
        'original': {
            'total_entities': sum(original_entities.values()),
            'unique_entities': len(original_entities),
            'entity_docs': sum(original_entity_docs.values())
        },
        'processed': {
            'total_entities': sum(processed_entities.values()),
            'unique_entities': len(processed_entities),
            'entity_docs': sum(processed_entity_docs.values())
        },
        'by_source': {}
    }
    
    # Combine source statistics
    all_sources = set(original_by_source.keys()) | set(processed_by_source.keys())
    for source in all_sources:
        orig = original_by_source.get(source, {'total_entities': 0, 'unique_entities': 0, 'documents': set()})
        proc = processed_by_source.get(source, {'total_entities': 0, 'unique_entities': 0, 'documents': set()})
        
        stats['by_source'][source] = {
            'original': {
                'total_entities': orig['total_entities'],
                'unique_entities': orig['unique_entities'],
                'documents': len(orig['documents'])
            },
            'processed': {
                'total_entities': proc['total_entities'],
                'unique_entities': proc['unique_entities'],
                'documents': len(proc['documents'])
            },
            'diff': {
                'total_entities': proc['total_entities'] - orig['total_entities'],
                'unique_entities': proc['unique_entities'] - orig['unique_entities'],
                'documents': len(proc['documents']) - len(orig['documents'])
            }
        }
    
    # Find entities that were lost or gained
    lost_entities = set(original_entities.keys()) - set(processed_entities.keys())
    gained_entities = set(processed_entities.keys()) - set(original_entities.keys())
    
    stats['lost_entities'] = list(lost_entities)[:100]  # Limit to 100 examples
    stats['gained_entities'] = list(gained_entities)[:100]  # Limit to 100 examples
    
    return stats

def validate_json_files(json_dir, processed_documents):
    """
    Validate that JSON files contain the correct entities for each document
    Returns a dictionary containing validation statistics and issues
    """
    stats = {
        'total_files': 0,
        'files_with_issues': 0,
        'issues': [],
        'missing': [],
        'extra': []
    }
    
    # Create a dictionary from processed_documents for faster lookup
    docs_dict = {}
    total_json_files = 0
    
    for doc in processed_documents:
        # Handle both chunk_id and doc_id formats
        if 'chunk_id' in doc:
            id_part = doc['chunk_id']
        else:
            id_part = doc.get('doc_id', '')
        
        # Use the standardized format as used in file_generation.py
        doc_id = f"{doc['dataset_alias']}_{doc['language']}_{doc.get('split', 'unknown')}_{id_part}"
        docs_dict[doc_id] = doc
        
        # Also store by just dataset/language/id for alternative matching
        alt_key = f"{doc['dataset_alias']}/{doc['language']}/{id_part}"
        docs_dict[alt_key] = doc
    
    print(f"Sample processed document IDs: {list(docs_dict.keys())[:3]}")
    
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Get the document key - try multiple formats
                doc_key = None
                
                # Extract components from the file path
                rel_path = os.path.relpath(file_path, json_dir)
                parts = rel_path.split(os.sep)
                
                # Only process files in the expected structure
                if len(parts) >= 4:  # Should have dataset/lang/split/filename.json
                    dataset_alias = parts[-4] if len(parts) >= 4 else ''
                    language = parts[-3] if len(parts) >= 3 else ''
                    split = parts[-2] if len(parts) >= 2 else ''
                    filename = parts[-1]
                    
                    # Try format 1: standardized ID from filename (remove .json)
                    doc_id = filename[:-5] if filename.endswith('.json') else filename
                    if doc_id in docs_dict:
                        doc_key = doc_id
                    
                    # Try format 2: reconstructed standardized ID
                    if not doc_key:
                        # Remove any additional prefixes from filename
                        base_name = filename.replace('.json', '')
                        expected_prefix = f"{dataset_alias}_{language}_{split}_"
                        if base_name.startswith(expected_prefix):
                            chunk_id = base_name[len(expected_prefix):]
                            doc_key = base_name
                    
                    # Try format 3: full relative path without output dir prefix
                    if not doc_key:
                        clean_path = os.path.join(dataset_alias, language, split, filename)
                        if clean_path in docs_dict:
                            doc_key = clean_path
                
                # If still not found, report issue with debug info
                if not doc_key:
                    stats['issues'].append({
                        'file': rel_path,
                        'error': 'Document not found in processed documents',
                        'attempted_keys': [filename[:-5], os.path.join(dataset_alias, language, split, filename)]
                    })
                    stats['files_with_issues'] += 1
                    continue
                
                # Validate the document content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    processed_doc = docs_dict[doc_key]
                    
                    # Validate entities match between files
                    file_entities = set()
                    if isinstance(doc_data, dict):
                        # Handle different JSON structures
                        if 'page_1' in doc_data:
                            file_entities.update(doc_data['page_1'])
                        if 'page_2' in doc_data:
                            file_entities.update(doc_data['page_2'])
                        if 'entities' in doc_data:
                            if isinstance(doc_data['entities'], list):
                                file_entities.update(doc_data['entities'])
                            elif isinstance(doc_data['entities'], dict):
                                for page_entities in doc_data['entities'].values():
                                    file_entities.update(page_entities)
                    
                    processed_entities = set(processed_doc.get('entities', []))
                    
                    if file_entities != processed_entities:
                        stats['issues'].append({
                            'file': rel_path,
                            'error': 'Entity mismatch',
                            'file_entities': list(file_entities),
                            'processed_entities': list(processed_entities)
                        })
                        stats['files_with_issues'] += 1
                        
                except Exception as e:
                    stats['issues'].append({
                        'file': rel_path,
                        'error': f'Error processing file: {str(e)}'
                    })
                    stats['files_with_issues'] += 1
                
                stats['total_files'] += 1
    
    return stats

def display_validation_results(entity_stats, json_stats):
    """
    Display validation results in a readable format
    Handles multiple JSON file types (docsjson, srtjson, logjson)
    """
    print("\n==== Entity Extraction Validation ====")
    print(f"Original dataset total entities: {entity_stats['original']['total_entities']}")
    print(f"Original dataset unique entities: {entity_stats['original']['unique_entities']}")
    print(f"Processed dataset total entities: {entity_stats['processed']['total_entities']}")
    print(f"Processed dataset unique entities: {entity_stats['processed']['unique_entities']}")
    
    entity_diff = entity_stats['processed']['total_entities'] - entity_stats['original']['total_entities']
    unique_diff = entity_stats['processed']['unique_entities'] - entity_stats['original']['unique_entities']
    
    print(f"Entity count difference: {entity_diff:+d} ({(entity_diff/max(1, entity_stats['original']['total_entities'])*100):.2f}%)")
    print(f"Unique entity difference: {unique_diff:+d} ({(unique_diff/max(1, entity_stats['original']['unique_entities'])*100):.2f}%)")
    
    print("\n==== Entity Distribution by Source ====")
    for source, stats in entity_stats['by_source'].items():
        print(f"\n{source}:")
        print(f"  Original: {stats['original']['total_entities']} total / {stats['original']['unique_entities']} unique in {stats['original']['documents']} docs")
        print(f"  Processed: {stats['processed']['total_entities']} total / {stats['processed']['unique_entities']} unique in {stats['processed']['documents']} docs")
        print(f"  Difference: {stats['diff']['total_entities']:+d} total / {stats['diff']['unique_entities']:+d} unique / {stats['diff']['documents']:+d} docs")
    
    if entity_stats['lost_entities']:
        print(f"\nSample of lost entities ({len(entity_stats['lost_entities'])} shown):")
        for entity in entity_stats['lost_entities'][:10]:
            print(f"  - '{entity}'")
    
    if entity_stats['gained_entities']:
        print(f"\nSample of gained entities ({len(entity_stats['gained_entities'])} shown):")
        for entity in entity_stats['gained_entities'][:10]:
            print(f"  - '{entity}'")
    
    # Check if json_stats is a combined dictionary with multiple file types
    if isinstance(json_stats, dict) and any(key in json_stats for key in ['docsjson', 'srtjson', 'logjson']):
        # Display results for each JSON file type
        print("\n==== JSON File Validation Summary ====")
        for json_type, stats in json_stats.items():
            if isinstance(stats, dict) and not stats.get('error'):
                print(f"\n--- {json_type.upper()} Files ---")
                total_files = stats.get('total_files', 0)
                files_with_issues = stats.get('files_with_issues', 0)
                print(f"Total files checked: {total_files}")
                # Avoid division by zero
                if total_files > 0:
                    error_percentage = (files_with_issues / total_files) * 100
                else:
                    error_percentage = 0
                print(f"Files with issues: {files_with_issues} ({error_percentage:.2f}%)")
                
                if stats.get('issues', []):
                    print(f"\nSample of issues ({len(stats['issues'])} shown):")
                    for i, issue in enumerate(stats['issues'][:3]):
                        print(f"\n  Issue {i+1}:")
                        print(f"    File: {issue['file']}")
                        if 'error' in issue:
                            print(f"    Error: {issue['error']}")
                        else:
                            print(f"    Missing entities: {len(issue['missing'])}")
                            if issue['missing']:
                                for entity in issue['missing'][:2]:
                                    print(f"      - '{entity}'")
                                if len(issue['missing']) > 2:
                                    print(f"      - ... and {len(issue['missing']) - 2} more")
                            
                            print(f"    Extra entities: {len(issue['extra'])}")
                            if issue['extra']:
                                for entity in issue['extra'][:2]:
                                    print(f"      - '{entity}'")
                                if len(issue['extra']) > 2:
                                    print(f"      - ... and {len(issue['extra']) - 2} more")
            elif isinstance(stats, dict) and stats.get('error'):
                print(f"\n--- {json_type.upper()} Files ---")
                print(f"  Error: {stats['error']}")
            else:
                print(f"\n--- {json_type.upper()} Files ---")
                print("  No validation data available")
    else:
        # Fall back to the original single JSON file type display
        print("\n==== JSON File Validation ====")
        total_files = json_stats.get('total_files', 0)
        files_with_issues = json_stats.get('files_with_issues', 0)
        print(f"Total JSON files checked: {total_files}")
        if total_files > 0:
            error_percentage = (files_with_issues / total_files) * 100
        else:
            error_percentage = 0
        print(f"Files with issues: {files_with_issues} ({error_percentage:.2f}%)")
        
        if json_stats.get('issues', []):
            print(f"\nSample of issues ({len(json_stats['issues'])} shown):")
            for i, issue in enumerate(json_stats['issues'][:5]):
                print(f"\nIssue {i+1}:")
                print(f"  File: {issue['file']}")
                if 'error' in issue:
                    print(f"  Error: {issue['error']}")
                else:
                    print(f"  Missing entities: {len(issue['missing'])}")
                    if issue['missing']:
                        for entity in issue['missing'][:3]:
                            print(f"    - '{entity}'")
                        if len(issue['missing']) > 3:
                            print(f"    - ... and {len(issue['missing']) - 3} more")
                    
                    print(f"  Extra entities: {len(issue['extra'])}")
                    if issue['extra']:
                        for entity in issue['extra'][:3]:
                            print(f"    - '{entity}'")
                        if len(issue['extra']) > 3:
                            print(f"    - ... and {len(issue['extra']) - 3} more")

def run_validation(original_dataset, processed_documents, json_dir):
    """
    Run all validation steps and display results
    
    Args:
        original_dataset: Original HIPE dataset dataframe
        processed_documents: List of processed documents with entities
        json_dir: Directory containing JSON files to validate
        
    Returns:
        tuple: (entity_stats, json_stats) - Results of the validation
    """
    entity_stats = validate_entity_extraction(original_dataset, processed_documents)
    
    # Handle different JSON output directories
    json_stats = {}
    json_types = ['docsjson', 'logjson', 'srtjson']
    
    for json_type in json_types:
        json_type_dir = os.path.join(os.path.dirname(json_dir.rstrip('/')), json_type)
        if os.path.exists(json_type_dir):
            print(f"\nValidating {json_type.upper()} files in {json_type_dir}...")
            try:
                json_stats[json_type] = validate_json_files(json_type_dir, processed_documents)
            except Exception as e:
                json_stats[json_type] = {'error': str(e)}
    
    # Get the caller's name to determine whether to display results
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name if caller_frame else None
    
    # Only display individual results when not called from multi-file validation processes
    if caller_name not in ['main', 'validate_extracted_entities']:
        display_validation_results(entity_stats, json_stats)
    
    return entity_stats, json_stats
