
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_document(file_path: str | Path):
    """Load a supported document and preserve source/page metadata."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported: PDF, DOCX, TXT."
        )

    documents = loader.load()

    for doc in documents:
        doc.metadata["document"] = path.name

        page = doc.metadata.get("page")

        if page is not None:
            doc.metadata["page"] = int(page) + 1
        else:
            doc.metadata.pop("page", None)

        doc.page_content = clean_text(doc.page_content)

    return [
        doc for doc in documents
        if doc.page_content.strip()
    ]


def clean_text(text: str) -> str:
    """Normalize extracted text without deleting meaningful punctuation."""

    text = unicodedata.normalize("NFKC", text or "")

    text = text.replace("\x00", " ").replace("\ufeff", "")

    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_documents(
    documents,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """Clean and split loaded pages into overlapping searchable chunks."""

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0.")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    return [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]


def prepare_documents(
    file_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """
    Load and prepare one document for vector storage.

    Returns:
        List of split LangChain Document chunks.
    """

    documents = load_document(file_path)

    if not documents:
        return []

    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def prepare_documents_from_files(
    file_paths,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """
    Load and prepare multiple documents for vector storage.

    Each chunk preserves its source document and page metadata.
    """

    all_documents = []

    for file_path in file_paths:
        path = Path(file_path)

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents = load_document(path)
        all_documents.extend(documents)

    if not all_documents:
        return []

    return split_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
