
from sentence_transformers import SentenceTransformer

from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import split_into_chunks
from src.semantic_search import semantic_search


def main():
    pdf_path = input("Enter the path to your research PDF: ")

    try:
        # Step 1: Extract PDF text
        pages = extract_text_from_pdf(pdf_path)
        print(f"\nExtracted {len(pages)} pages.")

        # Step 2: Divide text into chunks
        chunks = split_into_chunks(
            pages,
            chunk_size=200,
            overlap=40
        )
        print(f"Created {len(chunks)} chunks.")

        # Step 3: Load the embedding model
        print("\nLoading embedding model...")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 4: Ask the user a question
        query = input("\nWhat would you like to search for? ")

        # Step 5: Find relevant chunks
        results = semantic_search(
            query,
            chunks,
            model,
            top_k=3
        )

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
