from sentence_transformers import SentenceTransformer

from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import split_into_chunks
from src.vector_store import file_hash, load_or_build_embeddings, search_index
from src.rag import answer_question

CHUNK_SIZE = 200
OVERLAP = 40
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5 #chunks sent to the llm 

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
            
            print("\nThinking... (a local model can take a minute)")

            try:
                result = answer_question(
                    query, embeddings, chunks, model, top_k=RAG_TOP_K
                )
            except RuntimeError as error:
                #Ollama not running, report it, white keep the app alive
                print(f"Error: {error}")
                continue

            # Step 6: Display the results
            print(f"Answer:\n{result['answer']}")
            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"  Page {source['page_number']} "
                    f"(similarity {source['similarity']:.3f})"
                )

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
