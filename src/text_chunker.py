#dividing pdf page into smaller chunks.

def split_into_chunks(
        pages: list[dict],
        chunk_size: int = 200,
        overlap: int = 40,
) -> list[dict]:

    """
    Split extracted PDF pages into smaller text chunks.
    Each chunk keep its original page number and chunk index.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    for page in pages:
        words = page["text"].split()
        page_number = page["page_number"]

        start = 0
        chunk_index = 1

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text
            })

            if end == len(words):
                break

            start = end - overlap
            chunk_index += 1
    return chunks
