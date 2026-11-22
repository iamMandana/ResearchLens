#embedding storage and exact cosine similarity search by using NumPy

import hashlib
from pathlib import Path
import numpy as np

def file_hash(path: str) -> str:
    #Return a short fingerprint of a file's contents.

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]

def build_embeddings(chunks: list[dict], model) -> np.ndarray:
    #embed every chunk. return float32 array shape.

    if not chunks:
        raise ValueError("No chunks to embed.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.asarray(embeddings, dtype="float32")

def load_or_build_embeddings(
        chunks: list[dict],
        model,
        cache_key: str,
        cache_dir: str = "data/cache",
) -> np.ndarray:
    #load saved embeddings if they match the chunks otherwise rebuild them.

    cache_path = Path(cache_dir) / f"{cache_key}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        # row i must belong to chunks[i]; a size mismatch means a stale cache.
        if len(embeddings) == len(chunks):
            return embeddings

    embeddings = build_embeddings(chunks, model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings

def search_index(
        query: str,
        embeddings: np.ndarray,
        chunks: list[dict],
        model,
        top_k: int = 3
) -> list[dict]:
    #rank chukns by cosinr similarity to the query using storesd embeddings.

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not chunks:
        return []

    query_embedding = np.asarray(
        model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0],
        dtype="float32",
    )

    #vectors are normalised so a dot product equals cosine similarity.
    scores = embeddings @ query_embedding
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in best_indices:
        result = chunks[int(index)].copy()
        result["similarity"] = float(scores[index])
        results.append(result)

    return results