
import numpy as np
import pytest

from src.semantic_search import semantic_search


class FakeModel:
    """A small fake embedding model for testing."""

    def encode(self, sentences, **kwargs):
        vectors = {
            "heart disease treatment": [1.0, 0.0],
            "cardiac medical care": [0.9, 0.1],
            "sunny weather tomorrow": [0.0, 1.0],
            "heart disease": [1.0, 0.0]
        }

        return np.array([vectors[text] for text in sentences])


def test_semantic_search_ranks_relevant_chunks_first():
    chunks = [
        {
            "page_number": 1,
            "chunk_index": 1,
            "text": "sunny weather tomorrow"
        },
        {
            "page_number": 2,
            "chunk_index": 1,
            "text": "heart disease treatment"
        },
        {
            "page_number": 3,
            "chunk_index": 1,
            "text": "cardiac medical care"
        }
    ]

    results = semantic_search(
        "heart disease",
        chunks,
        FakeModel(),
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["text"] == "heart disease treatment"
    assert results[1]["text"] == "cardiac medical care"


def test_empty_query_raises_error():
    with pytest.raises(ValueError):
        semantic_search("   ", [], FakeModel())


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        semantic_search("heart disease", [], FakeModel(), top_k=0)
