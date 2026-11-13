from src.pdf_reader import extract_text_from_pdf

pdf_path = input("Enter the path to your research PDF:")

try:
    pages = extract_text_from_pdf(pdf_path)

    print(f"\nSuccessfully read {len(pages)} pages!")

    for page in pages:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:500])

except (FileNotFoundError, ValueError) as error:
    print(f"Error: {error}")