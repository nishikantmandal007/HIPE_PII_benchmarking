import nltk
from continuous_processing import create_continuous_stream

# Download the sentence tokenizer model if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def segment_documents(documents, words_per_page, dataset_alias, language, split):
    """
    Segment documents into sentence-aware chunks of text with 2 pages per chunk.
    """
    continuous_text, continuous_entities = create_continuous_stream(documents)
    # Tokenize into sentences and store their start/end character indices
    sentences_with_indices = []
    current_char_index = 0
    for sentence in nltk.sent_tokenize(continuous_text):
        # Filter out very short sentences that might be artifacts
        MIN_SENTENCE_LENGTH = 10  # Minimum characters for a sentence
        if len(sentence) >= MIN_SENTENCE_LENGTH:
            start_index = continuous_text.find(sentence, current_char_index)
            if start_index != -1:
                end_index = start_index + len(sentence)
                sentences_with_indices.append((sentence, start_index, end_index))
                current_char_index = end_index

    chunks = []
    sentence_idx = 0
    chunk_idx = 0

    while sentence_idx < len(sentences_with_indices):
        # --- Create Page 1 ---
        page1_text_parts = []
        page1_word_count = 0
        page1_start_char = -1
        page1_end_char = -1

        while sentence_idx < len(sentences_with_indices) and page1_word_count < words_per_page:
            sentence, s_start, s_end = sentences_with_indices[sentence_idx]
            page1_text_parts.append(sentence)
            page1_word_count += len(sentence.split())
            
            if page1_start_char == -1: # First sentence of the page
                page1_start_char = s_start
            page1_end_char = s_end # Last sentence of the page
            
            sentence_idx += 1
        
        page1_text = " ".join(page1_text_parts).strip()

        # --- Create Page 2 ---
        page2_text_parts = []
        page2_word_count = 0
        page2_start_char = -1
        page2_end_char = -1

        while sentence_idx < len(sentences_with_indices) and page2_word_count < words_per_page:
            sentence, s_start, s_end = sentences_with_indices[sentence_idx]
            page2_text_parts.append(sentence)
            page2_word_count += len(sentence.split())
            
            if page2_start_char == -1: # First sentence of the page
                page2_start_char = s_start
            page2_end_char = s_end # Last sentence of the page
            
            sentence_idx += 1
        
        page2_text = " ".join(page2_text_parts).strip()

        # If both pages are empty, stop.
        if not page1_text and not page2_text:
            break

        # Use a simple index for the chunk part of the ID, which will be combined 
        # with other identifiers later for a fully qualified ID
        chunk_id = f"chunk{chunk_idx}"

        page1_entities = []
        if page1_start_char != -1:
            for entity in continuous_entities:
                # Check if entity is within the character range of page 1
                if (entity['start'] >= page1_start_char and entity['start'] < page1_end_char) or \
                   (entity['end'] > page1_start_char and entity['end'] <= page1_end_char) or \
                   (entity['start'] < page1_start_char and entity['end'] > page1_end_char): # Entity spans across the page
                    
                    entity_text_in_page = entity['text']
                    relative_start = page1_text.find(entity_text_in_page)
                    if relative_start != -1:
                        page1_entities.append({
                            'text': entity_text_in_page,
                            'start': relative_start,
                            'end': relative_start + len(entity_text_in_page)
                        })

        page2_entities = []
        if page2_start_char != -1:
            for entity in continuous_entities:
                # Check if entity is within the character range of page 2
                if (entity['start'] >= page2_start_char and entity['start'] < page2_end_char) or \
                   (entity['end'] > page2_start_char and entity['end'] <= page2_end_char) or \
                   (entity['start'] < page2_start_char and entity['end'] > page2_end_char):
                    
                    entity_text_in_page = entity['text']
                    relative_start = page2_text.find(entity_text_in_page)
                    if relative_start != -1:
                        page2_entities.append({
                            'text': entity_text_in_page,
                            'start': relative_start,
                            'end': relative_start + len(entity_text_in_page)
                        })

        chunks.append({
            'chunk_id': chunk_id,
            'dataset_alias': dataset_alias,
            'language': language,
            'split': split,
            'page1': {
                'text': page1_text,
                'entities': page1_entities
            },
            'page2': {
                'text': page2_text,
                'entities': page2_entities
            }
        })
        chunk_idx += 1

    return chunks
