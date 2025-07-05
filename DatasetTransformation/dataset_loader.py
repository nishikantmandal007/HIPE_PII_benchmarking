import os
import pandas as pd
import glob
from collections import defaultdict

def load_hipe_data(data_dir):
    """
    Load HIPE-2022 dataset with proper path handling and dataset alias awareness
    """
    all_documents = []
    dataset_statistics = {
        'language_counts': defaultdict(int),
        'entity_counts': defaultdict(int),
        'alias_counts': defaultdict(int),
        'alias_language_counts': defaultdict(lambda: defaultdict(int))
    }
    
    # Dataset aliases and their languages
    dataset_aliases = {
        'ajmc': ['de', 'fr', 'en'],
        'hipe2020': ['de', 'fr', 'en'],
        'letemps': ['fr'],
        'topres19th': ['en'],
        'newseye': ['de', 'fi', 'fr', 'sv'],
        'sonar': ['de']
    }
    
    # Ensure data_dir is absolute
    data_dir = os.path.abspath("/home/stark007/MayaDataPrivacy/TESTAPI/HIPE-2022-data/data/v2.1")

    print(f"Looking for data files in: {data_dir}")
    
    # Find all training, dev and test files
    tsv_files = []
    for pattern in ["*train*.tsv", "*dev*.tsv", "*test*.tsv"]:
        search_path = os.path.join(data_dir, "**", pattern)
        found_files = glob.glob(search_path, recursive=True)
        tsv_files.extend(found_files)
        
    if not tsv_files:
        print(f"Warning: No TSV files found in {data_dir}. Please check the path.")
        return pd.DataFrame(), dataset_statistics
        
    print(f"Found {len(tsv_files)} TSV files to process.")
    
    for filepath in sorted(tsv_files):
        try:
            # Skip README files
            if "README" in filepath or "documentation" in filepath:
                continue
            
            # Determine dataset alias from file path
            dataset_alias = None
            for alias in dataset_aliases.keys():
                if alias in filepath:
                    dataset_alias = alias
                    break
            
            if not dataset_alias:
                # Try to infer from directory structure
                path_parts = os.path.normpath(filepath).split(os.sep)
                for part in path_parts:
                    if part in dataset_aliases:
                        dataset_alias = part
                        break
            
            if not dataset_alias:
                print(f"Warning: Could not determine dataset alias for {os.path.basename(filepath)}. Using 'unknown'.")
                dataset_alias = 'unknown'
                
            # Determine language from file path
            language = None
            basename = os.path.basename(filepath)
            # Try multiple patterns for language detection
            for lang in ["de", "en", "fr", "fi", "sv", "nl"]:
                if f"_{lang}_" in basename or f".{lang}." in basename or basename.endswith(f".{lang}.tsv"):
                    language = lang
                    break
            if not language:
                # Try to infer from directory structure
                path_parts = os.path.normpath(filepath).split(os.sep)
                for part in path_parts:
                    if part in ["de", "en", "fr", "fi", "sv", "nl"]:
                        language = part
                        break
            if not language:
                print(f"Warning: Could not determine language for {os.path.basename(filepath)}. Using 'unknown'.")
                language = 'unknown'
            
            if dataset_alias != 'unknown' and language != 'unknown' and language not in dataset_aliases[dataset_alias]:
                print(f"Note: Language {language} not expected for dataset alias {dataset_alias}. Proceeding anyway.")
            
            # Determine split from file path
            split = 'unknown'
            if 'train' in basename:
                split = 'train'
            elif 'dev' in basename:
                split = 'dev'
            elif 'test' in basename:
                split = 'test'
            
            print(f"Processing {os.path.basename(filepath)} (Alias: {dataset_alias}, Language: {language}, Split: {split})")
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            current_doc = []
            doc_id = None
            
            for line in lines:
                line = line.strip()
                
                # Skip comment lines but extract document ID if available
                if line.startswith('#'):
                    if "document_id" in line:
                        doc_id = line.split('=')[1].strip()
                    continue
                
                # Handle document boundaries
                if '-DOCSTART-' in line or not line:
                    if current_doc:
                        # Convert to dataframe and store document
                        doc_df = pd.DataFrame(current_doc, columns=["token", "NE-COARSE-LIT", "NE-FINE-LIT", "NE-FINE-METO", "NE-NESTED"])
                        doc_df['language'] = language
                        doc_df['dataset_alias'] = dataset_alias
                        doc_df['doc_id'] = doc_id
                        doc_df['split'] = split
                        all_documents.append(doc_df)
                        
                        # Update statistics
                        dataset_statistics['language_counts'][language] += len(doc_df)
                        dataset_statistics['alias_counts'][dataset_alias] += len(doc_df)
                        dataset_statistics['alias_language_counts'][dataset_alias][language] += len(doc_df)
                        
                        # Update entity statistics (excluding O tag)
                        for tag in doc_df[doc_df['NE-COARSE-LIT'] != 'O']['NE-COARSE-LIT']:
                            dataset_statistics['entity_counts'][tag] += 1
                        
                        current_doc = []
                        doc_id = None
                    continue
                
                # Process token lines
                parts = line.split('\t')
                if len(parts) >= 5:  # We need at least token and NE tags
                    current_doc.append(parts[:5])  # Token + 4 NE columns
            
            # Process the last document if any
            if current_doc:
                doc_df = pd.DataFrame(current_doc, columns=["token", "NE-COARSE-LIT", "NE-FINE-LIT", "NE-FINE-METO", "NE-NESTED"])
                doc_df['language'] = language
                doc_df['dataset_alias'] = dataset_alias
                doc_df['doc_id'] = doc_id
                doc_df['split'] = split
                all_documents.append(doc_df)
                
                dataset_statistics['language_counts'][language] += len(doc_df)
                dataset_statistics['alias_counts'][dataset_alias] += len(doc_df)
                dataset_statistics['alias_language_counts'][dataset_alias][language] += len(doc_df)
                
                for tag in doc_df[doc_df['NE-COARSE-LIT'] != 'O']['NE-COARSE-LIT']:
                    dataset_statistics['entity_counts'][tag] += 1
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Combine all documents into one dataset
    combined_data = pd.concat(all_documents) if all_documents else pd.DataFrame()
    
    print(f"Successfully loaded {len(combined_data)} tokens from {len(all_documents)} documents")
    
    return combined_data, dataset_statistics

def extract_person_names_per_doc(combined_data):
    """
    Extract person names (PER) for each document (page) from the loaded HIPE data.
    Returns a dict: {doc_id: [list of person names]}
    """
    if combined_data.empty:
        return {}
    
    person_names_by_doc = {}
    for doc_id, doc_df in combined_data.groupby('doc_id'):
        names = []
        current_name = []
        for idx, row in doc_df.iterrows():
            tag = row['NE-COARSE-LIT']
            token = row['token']
            if tag == 'B-PER':
                if current_name:
                    names.append(' '.join(current_name))
                current_name = [token]
            elif tag == 'I-PER' and current_name:
                current_name.append(token)
            else:
                if current_name:
                    names.append(' '.join(current_name))
                    current_name = []
        if current_name:
            names.append(' '.join(current_name))
        person_names_by_doc[doc_id] = names
    return person_names_by_doc

def save_person_names_to_json(person_names_by_doc, output_dir):
    """
    Save person names per document as individual JSON files in the given output directory.
    Each file will be named <doc_id>.json and contain a list of person names.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)
    for doc_id, names in person_names_by_doc.items():
        if not doc_id:
            continue  # skip docs without ID
        out_path = os.path.join(output_dir, f"{doc_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)