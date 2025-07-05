import pandas as pd
from tqdm import tqdm
import os
import json
import re
import nltk
# Fix imports to handle both direct import and package import
try:
    # When run directly from DatasetTransformation directory
    from dataset_loader import load_hipe_data, extract_person_names_per_doc, save_person_names_to_json
except ImportError:
    # When run from parent directory
    from DatasetTransformation.dataset_loader import load_hipe_data, extract_person_names_per_doc, save_person_names_to_json

# Download the sentence tokenizer model if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_text_and_entities(dataset):
    """
    Extract text and person entities from the dataset with dataset alias and language tracking
    """
    documents = []
    all_person_entities = []
    
    # Group by document
    doc_groups = dataset.groupby(['dataset_alias', 'language', 'doc_id'])
    
    for (dataset_alias, lang, doc_id), doc_df_orig in tqdm(doc_groups, desc="Processing documents"):
        
        # Pre-process tokens to handle '¬' character for split words
        rows_list = doc_df_orig.to_dict('records')
        new_rows = []
        i = 0
        while i < len(rows_list):
            row = rows_list[i]
            token = row.get('token', '')

            if token == '¬' and i > 0 and i + 1 < len(rows_list):
                if new_rows:
                    prev_row = new_rows[-1]
                    next_row = rows_list[i+1]
                    
                    # Merge token
                    prev_row['token'] += next_row['token']
                    
                    # The tag of the first part of the word is kept.
                    # This is a reasonable assumption for split words.
                    
                    i += 2  # Skip '¬' and the next token
                    continue
            
            new_rows.append(row)
            i += 1
        
        if not new_rows:
            continue
            
        doc_df = pd.DataFrame(new_rows)
        
        # Store original document ID for consistent referencing
        original_doc_id = doc_id

        doc_text = ""
        person_entities = []
        current_entity = []
        
        for idx, row in doc_df.iterrows():
            token = row.get('token', '')
            
            # Use less aggressive cleaning for tokens
            # Only remove TOKEN artifacts and HTML-like tags
            token = re.sub(r'TOKEN\s*\d*', '', token).strip()
            token = re.sub(r'<.*?>', '', token).strip()

            # Skip empty tokens after cleaning
            if pd.isna(token) or token == '':
                continue
            
            # Add token to document text with a simple space for now
            if doc_text:
                doc_text += ' '
            
            token_start = len(doc_text)
            doc_text += token
            token_end = len(doc_text)
            
        # --- Post-processing doc_text for consistent spacing and cleanup ---
        # Step 1: Less aggressive character filtering - keep more special characters
        # Only remove control characters and truly problematic characters
        allowed_chars = r'[\x00-\x1F\x7F]'  # Less aggressive filtering
        doc_text = re.sub(allowed_chars, '', doc_text)

        # Step 2: Special handling for academic abbreviations
        # Fix common Latin abbreviations
        doc_text = re.sub(r'(\s+i\.\s*e\.\s*)\.', r' i.e.', doc_text)  # i. e . -> i.e.
        doc_text = re.sub(r'(\s+e\.\s*g\.\s*)\.', r' e.g.', doc_text)  # e. g . -> e.g.
        doc_text = re.sub(r'(\s+cf\.\s*)\.', r' cf.', doc_text)        # cf. . -> cf.
        doc_text = re.sub(r'(\s+cp\.\s*)\.', r' cp.', doc_text)        # cp. . -> cp.
        doc_text = re.sub(r'(\s+infr\.\s*)\.', r' infr.', doc_text)    # infr. . -> infr.
        
        # Step 3: Citation and reference handling
        # Preserve proper spacing in citations and page references
        doc_text = re.sub(r'(\w+)\.\s+(\d+)', r'\1. \2', doc_text)     # "p. 123" stays "p. 123"
        doc_text = re.sub(r'(\d+)\s*\.\s*,', r'\1.,', doc_text)        # "62. ," -> "62.,"
        doc_text = re.sub(r'([A-Za-z])\.\s*([A-Za-z])\.\s*', r'\1.\2. ', doc_text)  # Fix initials spacing
        
        # Step 4: Apostrophe handling with improved context awareness
        # Handle cases like "word ' s" -> "word's" and "l ' execution" -> "l'execution"
        doc_text = re.sub(r'\s+\'\s+', "'", doc_text)  # spaces on both sides
        doc_text = re.sub(r'\s+\'', "'", doc_text)     # space before apostrophe
        doc_text = re.sub(r'\'\s+', "'", doc_text)     # space after apostrophe
        
        # Step 5: General punctuation spacing
        # Remove spaces before closing punctuation, preserving academic notation
        doc_text = re.sub(r'\s+([.,;!?:)\]}])', r'\1', doc_text)
        # Remove spaces after opening punctuation
        doc_text = re.sub(r'([\(\[\{])\s+', r'\1', doc_text)
        
        # Step 6: Special character spacing
        # Fix spacing around dashes and hyphens while preserving academic usage
        doc_text = re.sub(r'\s*(—|–|-)\s*', r' \1 ', doc_text)
        # Fix multiple hyphens and dashes (often found in OCR)
        doc_text = re.sub(r'[-—–]{2,}', '—', doc_text)
        
        # Step 7: Final cleanup
        # Replace multiple spaces with a single space
        doc_text = re.sub(r'\s+', ' ', doc_text)
        
        # Step 8: Sentence integrity checking
        # Split into sentences and filter/merge fragments
        sentences = nltk.sent_tokenize(doc_text)
        processed_sentences = []
        current_fragment = ""
        
        for sent in sentences:
            # Clean the sentence
            sent = sent.strip()
            
            # Skip empty sentences
            if not sent:
                continue
                
            # Check if this is likely a sentence fragment
            is_fragment = (
                len(sent.split()) < 3 or  # Very short
                not any(char.isupper() for char in sent[0]) or  # Doesn't start with capital
                (sent[-1] not in '.!?')  # Doesn't end with sentence-ending punctuation
            )
            
            if is_fragment and current_fragment:
                # Append to current fragment
                current_fragment += " " + sent
            elif is_fragment:
                # Start new fragment
                current_fragment = sent
            else:
                # Complete sentence
                if current_fragment:
                    # Add accumulated fragment to previous sentence
                    if processed_sentences:
                        processed_sentences[-1] += " " + current_fragment
                    else:
                        processed_sentences.append(current_fragment)
                    current_fragment = ""
                processed_sentences.append(sent)
        
        # Handle any remaining fragment
        if current_fragment and processed_sentences:
            processed_sentences[-1] += " " + current_fragment
        elif current_fragment:
            processed_sentences.append(current_fragment)
            
        # Rejoin the processed sentences
        doc_text = " ".join(processed_sentences)
        
        # Trim leading/trailing spaces
        doc_text = doc_text.strip()
        # Track person entities using IOB scheme
        # Determine the correct NE tag column based on dataset alias
        if dataset_alias in ['ajmc', 'hipe2020', 'letemps']:
            # For these aliases, use 'NE-COARSE-LIT' with 'B-pers', 'I-pers'
            ne_tag_col = 'NE-COARSE-LIT'
            b_tag = 'B-pers'
            i_tag = 'I-pers'
        elif dataset_alias in ['newseye', 'sonar', 'topres19th']:
            # For these aliases, use 'NE-COARSE-LIT' with 'B-PER', 'I-PER'
            ne_tag_col = 'NE-COARSE-LIT'
            b_tag = 'B-PER'
            i_tag = 'I-PER'
        else:
            # Default or fallback if alias not explicitly handled
            ne_tag_col = 'NE-COARSE-LIT' # Assuming this is the most common
            b_tag = 'B-PER'
            i_tag = 'I-PER'

        # Process entities for the document
        # First, collect all tokens and their original NE tags
        tokens_with_tags = []
        for idx, row in doc_df.iterrows():
            token = row.get('token', '')
            ne_tag = row.get(ne_tag_col, 'O')
            if pd.isna(ne_tag) or ne_tag == '_':
                ne_tag = 'O'
            tokens_with_tags.append((token, ne_tag))

        # Build the final doc_text and simultaneously track original entity spans
        final_doc_text_builder = []
        temp_person_entities = [] # To store entities with their original text and calculated span in the built text
        current_char_offset = 0

        current_entity_tokens_buffer = []
        original_entity_tokens_buffer = [] # Store original tokens before cleaning

        def clean_entity_text(text):
            """Minimal cleaning for entity text to preserve identity"""
            # Only remove control characters and normalize whitespace
            text = re.sub(r'[\x00-\x1F\x7F]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        for token_orig, ne_tag_orig in tokens_with_tags:
            # Store the original token before cleaning for entity tracking
            original_token = token_orig
            
            # Clean the token - using minimal cleaning to preserve entity identities
            cleaned_token = re.sub(r'TOKEN\s*\d*', '', token_orig).strip()
            cleaned_token = re.sub(r'<.*?>', '', cleaned_token).strip()

            if pd.isna(cleaned_token) or cleaned_token == '':
                continue

            # Determine spacing: add space before if not first token and not punctuation
            if final_doc_text_builder and not re.match(r'[.,!?:;"\'()\[\]{}\-—–&*%$#@+=<>/\\|~`^_]', cleaned_token):
                final_doc_text_builder.append(' ')
                current_char_offset += 1

            token_start_in_final_text = current_char_offset
            final_doc_text_builder.append(cleaned_token)
            current_char_offset += len(cleaned_token)
            token_end_in_final_text = current_char_offset

            # Entity tracking logic - Modified to be more conservative
            if ne_tag_orig.upper() == b_tag.upper():
                # Begin person entity
                if current_entity_tokens_buffer:
                    # Finalize previous entity with less aggressive cleaning
                    entity_text_raw = " ".join([t for t in original_entity_tokens_buffer])
                    entity_text_reconstructed = clean_entity_text(entity_text_raw)
                    temp_person_entities.append({
                        'text': entity_text_reconstructed,
                        'original': entity_text_raw,
                        'start': current_entity_tokens_buffer[0][1],
                        'end': current_entity_tokens_buffer[-1][2]
                    })
                current_entity_tokens_buffer = [(cleaned_token, token_start_in_final_text, token_end_in_final_text)]
                original_entity_tokens_buffer = [original_token]
            elif ne_tag_orig.upper() == i_tag.upper() and current_entity_tokens_buffer:
                # Continue person entity
                current_entity_tokens_buffer.append((cleaned_token, token_start_in_final_text, token_end_in_final_text))
                original_entity_tokens_buffer.append(original_token)
            elif current_entity_tokens_buffer:
                # Finalize previous entity with minimal cleaning
                entity_text_raw = " ".join([t for t in original_entity_tokens_buffer])
                entity_text_reconstructed = clean_entity_text(entity_text_raw)
                temp_person_entities.append({
                    'text': entity_text_reconstructed,
                    'original': entity_text_raw,
                    'start': current_entity_tokens_buffer[0][1],
                    'end': current_entity_tokens_buffer[-1][2]
                })
                current_entity_tokens_buffer = []
                original_entity_tokens_buffer = []
        
        # Handle any remaining entity at the end of the document
        if current_entity_tokens_buffer:
            entity_text_raw = " ".join([t for t in original_entity_tokens_buffer])
            entity_text_reconstructed = clean_entity_text(entity_text_raw)
            temp_person_entities.append({
                'text': entity_text_reconstructed,
                'original': entity_text_raw,
                'start': current_entity_tokens_buffer[0][1],
                'end': current_entity_tokens_buffer[-1][2]
            })

        doc_text = "".join(final_doc_text_builder)

        # Apply minimal post-processing to doc_text
        doc_text = re.sub(r'[\x00-\x1F\x7F]', '', doc_text) # Remove control chars
        doc_text = re.sub(r'\s+', ' ', doc_text) # Normalize whitespace
        doc_text = doc_text.strip() # Trim leading/trailing whitespace

        # Now, re-align entities to the final doc_text
        final_person_entities = []
        for entity in temp_person_entities:
            # First try to find the entity text directly in the doc_text
            entity_text = entity['text']
            start_idx = doc_text.find(entity_text)
            
            if start_idx >= 0:
                # Found a direct match
                final_person_entities.append({
                    'text': entity_text,
                    'start': start_idx,
                    'end': start_idx + len(entity_text)
                })
            else:
                # Try normalized versions as fallback
                normalized_entity = re.sub(r'\s+', ' ', entity_text.lower().strip())
                normalized_doc = re.sub(r'\s+', ' ', doc_text.lower())
                
                start_idx = normalized_doc.find(normalized_entity)
                if start_idx >= 0:
                    # Get the actual text from the document using the found position
                    actual_text = doc_text[start_idx:start_idx + len(normalized_entity)]
                    final_person_entities.append({
                        'text': entity_text,  # Keep original entity text
                        'start': start_idx,
                        'end': start_idx + len(normalized_entity)
                    })
                else:
                    # As a last resort, use the original entity with its position
                    # This preserves the entity even if we can't perfectly align it
                    if entity['start'] < len(doc_text) and entity['end'] <= len(doc_text):
                        final_person_entities.append({
                            'text': entity_text,
                            'start': entity['start'],
                            'end': entity['end']
                        })
        
        if not doc_text.strip():
            continue
            
        # Create a standardized document ID that will be consistent between processing and validation
        # Use dataset_alias, language, split, and original doc_id to ensure uniqueness
        split = doc_df['split'].iloc[0]
        standardized_doc_id = f"{dataset_alias}_{lang}_{split}_{original_doc_id}"
            
        documents.append({
            'doc_id': standardized_doc_id,  # Use the standardized ID
            'original_doc_id': original_doc_id,  # Keep the original ID for reference
            'dataset_alias': dataset_alias,
            'language': lang,
            'split': split,
            'text': doc_text,
            'entities': final_person_entities
        })
        
        all_person_entities.extend([(entity['text'], dataset_alias, lang) 
                                   for entity in final_person_entities])
    
    return documents, all_person_entities

def save_person_entities_json(documents, output_dir):
    """
    Save person names per page in JSON files.
    Each file will be named <doc_id>.json and contain lists of person names for each page.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group documents by alias, language and split to maintain directory structure
    for doc in documents:
        dataset_alias = doc.get('dataset_alias', 'unknown')
        lang = doc.get('language', 'unknown')
        split = doc.get('split', 'unknown')
        doc_id = doc.get('doc_id')
        
        if not doc_id:
            continue
            
        # Create directories for this document
        dir_path = os.path.join(output_dir, dataset_alias, lang, split)
        os.makedirs(dir_path, exist_ok=True)
            
        text = doc.get('text', '')
        text_length = len(text)
        mid_point = text_length // 2
        
        # Organize entities by page
        page1_entities = []
        page2_entities = []
        
        for entity in doc.get('entities', []):
            # Check entity position to determine which page it belongs to
            entity_mid = (entity['start'] + entity['end']) // 2
            if entity_mid < mid_point:
                if entity['text'] not in page1_entities:  # Avoid duplicates
                    page1_entities.append(entity['text'])
            else:
                if entity['text'] not in page2_entities:  # Avoid duplicates
                    page2_entities.append(entity['text'])
        
        # Create the document structure with only person names
        doc_content = {
            'page_1': page1_entities,
            'page_2': page2_entities
        }
        
        # Extract the original doc_id from the standardized ID (last part after underscore if present)
        file_doc_id = doc_id.split('_')[-1] if '_' in doc_id else doc_id
        
        # Use the original doc_id for the filename to match the validation lookup structure
        out_path = os.path.join(dir_path, f"{file_doc_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(doc_content, f, ensure_ascii=False, indent=2)
    
    return output_dir

def validate_extracted_entities(original_dataset, processed_documents, json_dir):
    """
    Run validation on the extracted entities and generated JSON files.
    
    Args:
        original_dataset: The original HIPE dataset
        processed_documents: The processed documents with extracted entities
        json_dir: Directory containing the generated JSON files
        
    Returns:
        tuple: (entity_stats, json_stats) - Results of the validation
    """
    from DatasetTransformation.entity_validation import run_validation
    
    print("\nRunning validation on extracted entities...")
    entity_stats, json_stats = run_validation(original_dataset, processed_documents, json_dir)
    
    return entity_stats, json_stats

