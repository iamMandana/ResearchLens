"""Retrieval-Augmented Generation: retrieve chunks, then answer from them."""

from src.llm_client import generate
from src.vector_store import search_index

NO_ANSWER = "I could not find this in the paper."

def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine the question and retrived chunks into one prompt"""
    context = "\n\n".join(
        f"[Page {chunk['page_number']}]\n{chunk['text']}" for chunk in chunks
    )

    return (
        "You are a research assistant. Answer the question using ONLY"
        "the context below. \n"
        "cite the page numbers you used, like [page 4]. \n"
        f"If the context does not contain the answer, reply exactly: {NO_ANSWER}\n\n"
        f"Context> \n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def answer_question(
        question: str,
        embeddings,
        chunks: list[dict],
        model,
        top_k: int = 5,
        llm=generate,
) -> dict:
    """
    Retrieve relevant chunks and ask the LLM to answer from them.
    Returns {"answer": str, "sources": [{"page_number", "similarity"}, ...]}.
    `llm` is a parameter so tests can replace the real model with a fake.
    """
    results = search_index(question, embeddings, chunks, model, top_k=top_k)

    if not results:
        return {"answer": NO_ANSWER, "sources": []}

    prompt = build_prompt(question, results)
    answer = llm(prompt)

    sources = [
        {"page_number": r["page_number"], "similarity": r["similarity"]}
        for r in results
    ]
    return {"answer": answer, "sources": sources}