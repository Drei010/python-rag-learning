from pathlib import Path
from typing import List, Tuple

import pandas as pd
from langchain_core.documents import Document

from core.config import settings


def get_excel_files() -> List[Path]:
    if not settings.data_dir.exists():
        return []

    return sorted(
        path
        for path in settings.data_dir.iterdir()
        if is_supported_excel_file(path.name) and path.is_file()
    )


def is_supported_excel_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in settings.supported_excel_extensions


def row_to_document(file_path: Path, sheet_name: str, row_index: int, row) -> Document:
    fields = []
    metadata = {
        "source": file_path.name,
        "sheet": sheet_name,
        "row": int(row_index),
    }

    for column, value in row.items():
        if pd.isna(value):
            continue

        text = str(value).strip()
        if not text:
            continue

        fields.append(f"{column}: {text}")

    page_content = "\n".join(
        [
            f"Source file: {file_path.name}",
            f"Sheet: {sheet_name}",
            f"Row: {row_index}",
            *fields,
        ]
    )

    return Document(
        page_content=page_content,
        metadata=metadata,
        id=f"{file_path.name}:{sheet_name}:{row_index}",
    )


def load_excel_documents() -> Tuple[List[Document], List[str]]:
    loaded_documents = []
    loaded_ids = []

    for excel_file in get_excel_files():
        sheets = pd.read_excel(excel_file, sheet_name=None)

        for sheet_name, df in sheets.items():
            for row_index, row in df.dropna(how="all").iterrows():
                document = row_to_document(excel_file, sheet_name, row_index, row)
                if document.page_content.strip():
                    loaded_documents.append(document)
                    loaded_ids.append(document.id)

    return loaded_documents, loaded_ids
