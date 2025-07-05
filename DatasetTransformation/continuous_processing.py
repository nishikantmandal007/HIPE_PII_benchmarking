import pandas as pd

def create_continuous_stream(documents):
    """
    Create a continuous stream of text and entities from a list of documents.
    """
    continuous_text = ""
    continuous_entities = []
    current_offset = 0 # Reset offset for each call

    for doc in documents:
        doc_text = doc['text']
        doc_entities = doc['entities']

        # Add document text to the continuous stream
        continuous_text += doc_text + " "

        # Realign entities for the continuous stream
        for entity in doc_entities:
            continuous_entities.append({
                'text': entity['text'],
                'start': entity['start'] + current_offset,
                'end': entity['end'] + current_offset
            })
        
        current_offset = len(continuous_text)

    return continuous_text, continuous_entities
