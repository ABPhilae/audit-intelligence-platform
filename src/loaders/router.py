"""
Document Loading Router.

This is the "mail room" — it looks at the file extension, picks the right
loader, and returns standardised LangChain Document objects regardless of
input format.

Why a router instead of one giant function?
- Each loader is isolated (easy to test, easy to replace)
- Adding a new file type = one new loader file + one line here
- The rest of the system NEVER needs to know about file types
"""
from langchain_core.documents import Document
from src.loaders.pdf_loader import load_pdf
from src.loaders.docx_loader import load_docx
from src.loaders.xlsx_loader import load_xlsx
from src.loaders.pptx_loader import load_pptx
import logging
import os

logger = logging.getLogger(__name__)

# Map file extensions to their loader functions
LOADER_REGISTRY = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_xlsx,
    ".pptx": load_pptx,
}


def load_document(file_path: str) -> list[Document]:
    """
    Load any supported document by routing to the correct loader.

    This is the ONLY function the rest of the system calls.
    It figures out the file type and delegates to the right loader.

    Args:
        file_path: Path to the document file

    Returns:
        List of LangChain Document objects

    Raises:
        ValueError: If the file type is not supported
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get the file extension (lowercase)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Handle plain text files
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(
            page_content=content,
            metadata={"source": file_path, "file_type": ext.lstrip(".")}
        )]

    # Route to the correct loader
    loader_func = LOADER_REGISTRY.get(ext)
    if loader_func is None:
        supported = ", ".join(LOADER_REGISTRY.keys()) + ", .txt, .md"
        raise ValueError(
            f"Unsupported file type: {ext}. Supported types: {supported}"
        )

    logger.info(f"Loading {ext} file: {file_path}")
    return loader_func(file_path)


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions."""
    return list(LOADER_REGISTRY.keys()) + [".txt", ".md"]
