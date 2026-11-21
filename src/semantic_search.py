
import numpy as np


def semantic_search(
    query: str,
    chunks: list[dict],
    model,
    top_k: int = 3
) -> list[dict]:
    """
    Find the most relevant text chunks for a user's question.

    Uses embeddings and cosine similarity to rank the chunks.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not chunks:
        return []

    # Extract the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Convert the document chunks into numerical vectors
    chunk_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Convert the user's question into a numerical vector
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Calculate similarity scores
    scores = chunk_embeddings @ query_embedding

    # Get the indexes of the highest-scoring chunks
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in best_indices:
        result = chunks[int(index)].copy()
        result["similarity"] = float(scores[index])
        results.append(result)

    return results
