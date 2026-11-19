from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import split_into_chunks

pdf_path = input("Enter the path to your research PDF:")

try:
    # first step it extracts the PDF text
    pages = extract_text_from_pdf(pdf_path)

    print(f"\nSuccessfully read {len(pages)} pages!")

    #second step it divides the extracted text into chunks
    chunks = split_into_chunks(
        pages,
        chunk_size=200,
        overlap=40
    )

    print(f"Created {len(chunks)} text chunks.")

    #Third step: display the first three chunks
    for chunk in chunks:
        print(f"\n--- Page {chunk['page_number']}"
              f"| Chunk {chunk['chunk_index']} ---")
        print(chunk["text"])

except (FileNotFoundError, ValueError) as error:
    print(f"Error: {error}")