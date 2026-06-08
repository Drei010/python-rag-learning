from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from core.config import settings
from services.file_service import get_supported_files


def get_pdf_files() -> List[Path]:
    return sorted(
        path
        for path in get_supported_files()
        if path.is_file() and path.suffix.lower() in settings.supported_pdf_extensions
    )


def page_to_document(file_path: Path, page_number: int, text: str) -> Document:
    page_content = "\n".join(
        [
            f"Source file: {file_path.name}",
            "File type: PDF",
            f"Page: {page_number}",
            text.strip(),
        ]
    )

    return Document(
        page_content=page_content,
        metadata={
            "source": file_path.name,
            "file_type": "pdf",
            "page": page_number,
        },
        id=f"{file_path.name}:page:{page_number}",
    )


def load_pdf_documents() -> Tuple[List[Document], List[str]]:
    from pypdf import PdfReader

    loaded_documents = []
    loaded_ids = []

    for pdf_file in get_pdf_files():
        reader = PdfReader(str(pdf_file))

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            document = page_to_document(pdf_file, page_index, text)
            loaded_documents.append(document)
            loaded_ids.append(document.id)

    return loaded_documents, loaded_ids
