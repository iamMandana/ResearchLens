import json
import sys

from sentence_transformers import SentenceTransformer

from src.main import CHUNK_SIZE, MODEL_NAME, OVERLAP
from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import split_into_chunks
from src.vector_store import file_hash, load_or_build_embeddings, search_index


def is_hit(results: list[dict], keywords: list[str]) -> bool:
    """True if any retrieved chunk contains any expected keyword."""
    for result in results:
        text = result["text"].lower()
        if any(keyword.lower() in text for keyword in keywords):
            return True
    return False


def evaluate(pdf_path: str, questions_path: str, top_k: int = 3) -> float:
    pages = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    model = SentenceTransformer(MODEL_NAME)
    cache_key = f"{file_hash(pdf_path)}_{CHUNK_SIZE}_{OVERLAP}_{MODEL_NAME.split('/')[-1]}"
    embeddings = load_or_build_embeddings(chunks, model, cache_key)

    with open(questions_path, encoding="utf-8") as file:
        questions = json.load(file)

    hits = 0
    for item in questions:
        results = search_index(item["question"], embeddings, chunks, model, top_k)
        hit = is_hit(results, item["keywords"])
        hits += hit
        pages_found = [r["page_number"] for r in results]
        print(f"{'HIT ' if hit else 'MISS'} | {item['question']} | pages {pages_found}")

    rate = hits / len(questions)
    print(f"\nHit rate@{top_k}: {hits}/{len(questions)} = {rate:.0%}")
    return rate


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])