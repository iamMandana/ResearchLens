import numpy as np
import pytest

from src.vector_store import (
    build_embeddings,
    file_hash,
    load_or_build_embeddings,
    search_index,
)


class FakeModel:
    """Fake embedding model: fast, and needs no model download."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        vectors = {
            "heart disease treatment": [1.0, 0.0],
            "cardiac medical care": [0.9, 0.1],
            "sunny weather tomorrow": [0.0, 1.0],
            "heart disease": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts])


CHUNKS = [
    {"page_number": 1, "chunk_index": 1, "text": "sunny weather tomorrow"},
    {"page_number": 2, "chunk_index": 1, "text": "heart disease treatment"},
    {"page_number": 3, "chunk_index": 1, "text": "cardiac medical care"},
]


def test_build_embeddings_shape_and_dtype():
    embeddings = build_embeddings(CHUNKS, FakeModel())
    assert embeddings.shape == (3, 2)
    assert embeddings.dtype == np.float32


def test_build_embeddings_rejects_empty_chunks():
    with pytest.raises(ValueError):
        build_embeddings([], FakeModel())


def test_search_ranks_relevant_chunks_first():
    model = FakeModel()
    embeddings = build_embeddings(CHUNKS, model)

    results = search_index("heart disease", embeddings, CHUNKS, model, top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "heart disease treatment"
    assert results[1]["text"] == "cardiac medical care"
    assert results[0]["page_number"] == 2


def test_empty_query_raises_error():
    model = FakeModel()
    embeddings = build_embeddings(CHUNKS, model)
    with pytest.raises(ValueError):
        search_index("   ", embeddings, CHUNKS, model)


def test_top_k_must_be_positive():
    model = FakeModel()
    embeddings = build_embeddings(CHUNKS, model)
    with pytest.raises(ValueError):
        search_index("heart disease", embeddings, CHUNKS, model, top_k=0)


def test_cache_is_reused(tmp_path):
    model = FakeModel()

    load_or_build_embeddings(CHUNKS, model, "paper", cache_dir=str(tmp_path))
    assert model.calls == 1

    load_or_build_embeddings(CHUNKS, model, "paper", cache_dir=str(tmp_path))
    assert model.calls == 1  # loaded from disk, model not called again


def test_stale_cache_is_rebuilt(tmp_path):
    model = FakeModel()
    load_or_build_embeddings(CHUNKS, model, "paper", cache_dir=str(tmp_path))

    # Different number of chunks under the same key -> cache must be rebuilt.
    embeddings = load_or_build_embeddings(
        CHUNKS[:2], model, "paper", cache_dir=str(tmp_path)
    )
    assert len(embeddings) == 2
    assert model.calls == 2


def test_file_hash_changes_with_content(tmp_path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"paper one")
    second.write_bytes(b"paper two")

    assert file_hash(str(first)) == file_hash(str(first))
    assert file_hash(str(first)) != file_hash(str(second))
