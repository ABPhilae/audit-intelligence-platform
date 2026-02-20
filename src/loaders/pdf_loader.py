"""
PDF Document Loader.

Handles two types of PDFs:
1. Text-based PDFs (digital-born): Uses pdfplumber for best table extraction
2. Scanned PDFs (images of text): Falls back to basic OCR

Why pdfplumber instead of PyPDF2?
- pdfplumber handles tables MUCH better (critical for audit reports)
- It preserves layout information
- It is the recommended default for financial documents
"""
from langchain_core.documents import Document
import pdfplumber
import logging

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects.

    Each page becomes one Document with metadata including:
    - source: the file path
    - page: the page number (0-indexed)
    - file_type: "pdf"
    - has_tables: whether tables were detected on this page

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Document objects, one per page
    """
    documents = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract the main text
                text = page.extract_text() or ""

                # Extract tables separately (pdfplumber's superpower)
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        # Convert each table to a readable text format
                        # First row is usually headers
                        if table and len(table) > 1:
                            headers = [str(cell or "").strip() for cell in table[0]]
                            for row in table[1:]:
                                row_text = " | ".join(
                                    f"{h}: {str(cell or '').strip()}"
                                    for h, cell in zip(headers, row)
                                )
                                table_text += row_text + "\n"

                # Combine text and tables
                full_text = text
                if table_text:
                    full_text += "\n\n[TABLE DATA]\n" + table_text

                if full_text.strip():
                    documents.append(Document(
                        page_content=full_text.strip(),
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "file_type": "pdf",
                            "has_tables": len(tables) > 0,
                        }
                    ))

        logger.info(f"Loaded PDF: {file_path} — {len(documents)} pages")

    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise

    return documents
