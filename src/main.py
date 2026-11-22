from sentence_transformers import SentenceTransformer

from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import split_into_chunks
from src.vector_store import file_hash, load_or_build_embeddings, search_index

CHUNK_SIZE = 200
OVERLAP = 40
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    pdf_path = input("Enter the path to your research PDF: ")

    try:
        # Step 1: Extract PDF text
        pages = extract_text_from_pdf(pdf_path)
        print(f"\nExtracted {len(pages)} pages.")

        # Step 2: Divide text into chunks
        chunks = split_into_chunks(
            pages,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )
        print(f"Created {len(chunks)} chunks.")

        # Step 3: Load the embedding model
        print("\nLoading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

        #Step 4: Embed once and cache. The key changes if the PDF,
        #chunk settings or model change, so stale caches are never reused.
        model_tag = MODEL_NAME.split("/")[-1]
        cache_key = f"{file_hash(pdf_path)}_{CHUNK_SIZE}_{OVERLAP}_{model_tag}"
        embeddings = load_or_build_embeddings(chunks, model, cache_key)
        print("Embeddings ready.")

        # Step 5: Answer as many questions as the user like
        while True:
            query = input("\nAsk a question (or type 'quit'): ")

            if query.strip().lower() == "quit":
                break

            if not query.strip():
                print("Please type a question.")
                continue

            results = search_index(query, embeddings, chunks, model, top_k=3)

            # Step 6: Display the results
            print("\nMost relevant research sections:")

            for result in results:
                print(
                    f"\nPage {result['page_number']} "
                    f"| Similarity: {result['similarity']:.3f}"
                )
                print(result["text"])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
