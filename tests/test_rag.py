import numpy as np

from src.rag import NO_ANSWER, answer_question, build_prompt
from src.vector_store import build_embeddings


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = {
            "heart disease treatment": [1.0, 0.0],
            "sunny weather tomorrow": [0.0, 1.0],
            "heart disease": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts])


CHUNKS = [
    {"page_number": 1, "chunk_index": 1, "text": "sunny weather tomorrow"},
    {"page_number": 2, "chunk_index": 1, "text": "heart disease treatment"},
]


def test_build_prompt_contains_question_context_and_pages():
    prompt = build_prompt("What is X?", CHUNKS)

    assert "What is X?" in prompt
    assert "[Page 1]" in prompt
    assert "sunny weather tomorrow" in prompt
    assert "[Page 2]" in prompt
    assert "ONLY" in prompt
    assert NO_ANSWER in prompt


def test_answer_question_returns_answer_and_sources():
    model = FakeModel()
    embeddings = build_embeddings(CHUNKS, model)
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "It is treated. [Page 2]"

    result = answer_question(
        "heart disease", embeddings, CHUNKS, model, top_k=1, llm=fake_llm
    )

    assert result["answer"] == "It is treated. [Page 2]"
    assert result["sources"][0]["page_number"] == 2
    assert "heart disease treatment" in seen["prompt"]
    assert "sunny weather tomorrow" not in seen["prompt"]  # top_k=1


def test_no_chunks_returns_no_answer_without_calling_llm():
    def llm_must_not_run(prompt):
        raise AssertionError("LLM should not be called")

    result = answer_question(
        "heart disease", np.empty((0, 2)), [], FakeModel(), llm=llm_must_not_run
    )

    assert result == {"answer": NO_ANSWER, "sources": []}