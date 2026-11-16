import pymupdf
import pytest

from src.pdf_reader import extract_text_from_pdf

def create_test_pdf(file_path, page_texts):
    #create a small pdf for testing.
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)

        document.save(file_path)

def test_extract_text_from_pdf(tmp_path):
    #arrange: create a temporary two-page PDF
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path, ["Hello ResearchLens", "This is page two"])

    #act: extract the PDF text
    result = extract_text_from_pdf(str(pdf_path))

    #assert: check the extracted information
    assert len(result) == 2
    assert result[0]["page_number"] == 1
    assert "Hello ResearchLens" in result[0]["text"]
    assert result[1]["page_number"] == 2
    assert "This is page two" in result[1]["text"]

def test_missing_pdf_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(str(missing_path))

def test_invalid_file_type_raises_error(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        extract_text_from_pdf(str(text_file))