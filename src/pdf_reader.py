from pathlib import Path
import pymupdf

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    extract text from each page of pdf
    and returns a list containing page numbers and text.
    """

    file_path = Path(pdf_path)

    #check if the file is pdf
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("The file must be a PDF.")

    extracted_pages = []

    #open pdf
    with pymupdf.open(file_path) as document:

        #read each page
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            extracted_pages.append({
                "page_number": page_number,
                "text": text
            })

    return extracted_pages
