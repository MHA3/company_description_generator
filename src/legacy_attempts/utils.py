import re

def chunk_text(text, max_length):
    # Split the text by sentence boundaries
    sentences = re.split('(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ''

    for sentence in sentences:
        # If adding the next sentence to the current chunk doesn't exceed the max_length
        # OR if the current_chunk is empty (which means this sentence is longer than max_length)
        while len(current_chunk) + len(
                sentence) > max_length or not current_chunk:
            # If the current_chunk isn't empty, append it to the chunks
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ''
            # If the sentence itself is longer than max_length
            if len(sentence) > max_length:
                # Split the sentence to fill the chunk to max_length and reduce the sentence length
                split_point = max_length
                current_chunk = sentence[:split_point]
                sentence = sentence[split_point:]
            else:
                break
        # Add the sentence to the current chunk
        if current_chunk:
            current_chunk += ' ' + sentence
        else:
            current_chunk = sentence

    # Append any remaining chunk to chunks
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
