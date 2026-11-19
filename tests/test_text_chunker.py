
import pytest

from src.text_chunker import split_into_chunks


def test_split_text_into_chunks():
    pages = [{
        "page_number": 1,
        "text": "one two three four five six seven eight"
    }]

    chunks = split_into_chunks(
        pages,
        chunk_size=5,
        overlap=2
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"] == "four five six seven eight"
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 1


def test_short_text_creates_one_chunk():
    pages = [{
        "page_number": 3,
        "text": "This is a short research paper."
    }]

    chunks = split_into_chunks(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 3


def test_overlap_must_be_smaller_than_chunk_size():
    pages = [{
        "page_number": 1,
        "text": "Some example research text."
    }]

    with pytest.raises(ValueError):
        split_into_chunks(
            pages,
            chunk_size=5,
            overlap=5
        )
