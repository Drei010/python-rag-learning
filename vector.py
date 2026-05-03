"""
Excel RAG Ingestion Pipeline
Handles unstructured rows, mixed formats, and embedded images.

Dependencies:
    pip install langchain-ollama langchain-chroma langchain-core \
                openpyxl pandas pillow pytesseract
    # For OCR on images: also install tesseract-ocr system package
    # Ubuntu: sudo apt install tesseract-ocr
    # macOS:  brew install tesseract
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage

# Optional: OCR support for images that contain text
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
EXCEL_PATH = Path(__file__).parent / "data" / "internshiplist.xlsx"
DB_LOCATION = Path("./chroma_langchain_db")
EMBED_MODEL = "qwen3.5:9b"  
IMAGE_OUTPUT_DIR = Path("./extracted_images")


# ─────────────────────────────────────────────
# Image Extraction from openpyxl workbook
# ─────────────────────────────────────────────
def extract_images_from_sheet(
    wb_path: Path,
    sheet_name: str | None = None,
    output_dir: Path = IMAGE_OUTPUT_DIR,
) -> dict[int, list[dict[str, Any]]]:
    """
    Extract all embedded images from the worksheet.

    Returns a dict mapping approximate row index → list of image dicts:
        {"path": Path, "description": str, "row": int, "col": int}

    openpyxl anchors images to a cell via `anchor._from.row` (0-indexed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(wb_path)
    ws = wb[sheet_name] if sheet_name else wb.active

    row_images: dict[int, list[dict[str, Any]]] = {}

    for idx, drawing in enumerate(ws._images):  # type: ignore[attr-defined]
        try:
            # Resolve the anchor row (0-indexed → convert to 1-indexed to match pandas)
            anchor = drawing.anchor
            row_0idx = (
                anchor._from.row  # TwoCellAnchor / OneCellAnchor
                if hasattr(anchor, "_from")
                else anchor.row
            )
            col_0idx = (
                anchor._from.col
                if hasattr(anchor, "_from")
                else anchor.col
            )
            row_1idx = row_0idx  # openpyxl rows are 0-indexed here; pandas df rows start at 0

            # Get raw image bytes
            img_bytes: bytes = drawing._data()  # type: ignore[attr-defined]
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

            img_path = output_dir / f"img_row{row_1idx}_col{col_0idx}_{idx}.png"
            pil_img.save(img_path, format="PNG")

            # Try OCR — fall back to a placeholder description
            ocr_text = ""
            if OCR_AVAILABLE:
                try:
                    ocr_text = pytesseract.image_to_string(pil_img).strip()
                except Exception:
                    pass

            description = (
                f"[IMAGE row={row_1idx} col={col_0idx}]"
                + (f" OCR: {ocr_text}" if ocr_text else " (no readable text)")
            )

            row_images.setdefault(row_1idx, []).append(
                {
                    "path": img_path,
                    "description": description,
                    "row": row_1idx,
                    "col": col_0idx,
                    "ocr_text": ocr_text,
                }
            )
            logger.info("Extracted image → %s", img_path)

        except Exception as exc:
            logger.warning("Could not extract image #%d: %s", idx, exc)

    return row_images


# ─────────────────────────────────────────────
# Row Serializer — handles ANY shape of row
# ─────────────────────────────────────────────
def serialize_row(
    row: pd.Series,
    row_images: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    Convert one DataFrame row (arbitrary columns, mixed types) into:
      - page_content: human-readable text for embedding
      - metadata: structured dict for filtering / citation

    Strategy:
      1. Collect all non-null cells as "Field: Value" pairs
      2. Append any image descriptions (with optional OCR text)
    """
    parts: list[str] = []
    metadata: dict[str, Any] = {}

    for col_name, value in row.items():
        if pd.isna(value):
            continue
        # Normalise value to string; handle numeric, date, etc.
        if isinstance(value, float) and value.is_integer():
            str_val = str(int(value))
        else:
            str_val = str(value).strip()

        if not str_val:
            continue

        col_label = str(col_name).strip() if str(col_name) != str(col_name.__class__) else f"col_{col_name}"
        parts.append(f"{col_label}: {str_val}")
        # Store first 50 chars of each field in metadata for filtering
        metadata[col_label] = str_val[:200]

    # Append image OCR / descriptions
    image_paths: list[str] = []
    for img in row_images:
        parts.append(img["description"])
        image_paths.append(str(img["path"]))
        if img["ocr_text"]:
            # Surface OCR text as a searchable field
            parts.append(f"Image text: {img['ocr_text']}")

    if image_paths:
        metadata["image_paths"] = "|".join(image_paths)

    page_content = "\n".join(parts) if parts else "(empty row)"
    return page_content, metadata


# ─────────────────────────────────────────────
# Build Documents
# ─────────────────────────────────────────────
def build_documents(
    excel_path: Path,
    sheet_name: str | None = None,
) -> list[Document]:
    """
    Load the Excel file and produce one Document per non-empty row.
    Each Document carries the full row text + image descriptions as page_content.
    """
    logger.info("Loading Excel: %s", excel_path)

    # pandas for tabular data
    # sheet_name=None returns a dict[sheet_name → DataFrame] for multi-sheet files.
    # We always normalise to a single DataFrame here.
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)

    if isinstance(raw, dict):
        # Multi-sheet file: merge all sheets, tagging each row with its sheet name
        frames: list[pd.DataFrame] = []
        for sname, sdf in raw.items():
            sdf = sdf.copy()
            sdf["__sheet__"] = str(sname)   # preserve sheet origin in metadata
            frames.append(sdf)
        df = pd.concat(frames, ignore_index=True)
        logger.info(
            "Multi-sheet workbook: merged %d sheet(s) → %d total rows",
            len(raw), len(df),
        )
    else:
        df = raw

    # Drop rows that are completely empty
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # openpyxl for images
    row_image_map = extract_images_from_sheet(excel_path, sheet_name)

    documents: list[Document] = []

    for df_idx, row in df.iterrows():
        # openpyxl image rows: header row is row 0 in openpyxl (0-indexed),
        # data starts at row 1. df_idx is 0-indexed over data rows,
        # so openpyxl_row = df_idx + 1 to skip the header.
        openpyxl_row = int(df_idx) + 1
        images_for_row = row_image_map.get(openpyxl_row, [])

        page_content, metadata = serialize_row(row, images_for_row)

        # Enrich metadata
        metadata["source"] = str(excel_path.name)
        metadata["row_index"] = int(df_idx)  # type: ignore[arg-type]
        metadata["has_images"] = len(images_for_row) > 0

        doc = Document(
            page_content=page_content,
            metadata=metadata,
            id=str(df_idx),
        )
        documents.append(doc)
        logger.debug("Row %d → %d chars, %d image(s)", df_idx, len(page_content), len(images_for_row))

    logger.info("Built %d documents from %d rows", len(documents), len(df))
    return documents


# ─────────────────────────────────────────────
# Vector Store Setup
# ─────────────────────────────────────────────
def build_vector_store(
    documents: list[Document],
    db_location: Path,
    embed_model: str = EMBED_MODEL,
) -> Chroma:
    """
    Create (or load) the Chroma vector store.
    Documents are added in a single batch call — NOT one-at-a-time.
    """
    embeddings = OllamaEmbeddings(model=embed_model)

    vector_store = Chroma(
        collection_name="internshiplist",
        embedding_function=embeddings,
        persist_directory=str(db_location),
    )

    # ✅ Check actual document count — not folder existence.
    # A prior crashed/partial run can leave the folder empty, fooling an existence check.
    existing_count = vector_store._collection.count()
    logger.info("Chroma collection currently holds %d document(s).", existing_count)

    if existing_count == 0 and documents:
        logger.info("Embedding and ingesting %d documents...", len(documents))
        doc_ids = [doc.id for doc in documents]  # type: ignore[union-attr]
        try:
            vector_store.add_documents(documents, ids=doc_ids)
            ingested = vector_store._collection.count()
            logger.info(
                "Ingestion complete -- %d/%d documents now in Chroma at %s",
                ingested, len(documents), db_location,
            )
            if ingested == 0:
                raise RuntimeError(
                    "add_documents returned without error but collection is still empty. "
                    "Check Chroma version compatibility."
                )
        except Exception as exc:
            logger.error("Ingestion FAILED: %s", exc)
            raise  # re-raise so the caller (main.py) sees the real error
    elif existing_count > 0:
        logger.info("Chroma DB already populated (%d docs) -- skipping ingestion.", existing_count)
    else:
        logger.warning("No documents to add and collection is empty -- check your Excel file.")

    return vector_store


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def build_retriever(
    excel_path: Path = EXCEL_PATH,
    db_location: Path = DB_LOCATION,
    embed_model: str = EMBED_MODEL,
    top_k: int = 5,
    sheet_name: str | None = None,
):
    """
    Full pipeline: Excel → Documents → Chroma → Retriever.

    Usage:
        retriever = build_retriever()
        results = retriever.invoke("machine learning internship in Manila")
    """
    documents = build_documents(excel_path, sheet_name=sheet_name)
    vector_store = build_vector_store(documents, db_location, embed_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    logger.info("Retriever ready (top_k=%d)", top_k)
    return retriever


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    retriever = build_retriever()

    # Quick smoke-test
    query = "software engineering internship"
    results = retriever.invoke(query)
    print(f"\nTop results for: '{query}'\n" + "─" * 50)
    for doc in results:
        print(f"[row {doc.metadata.get('row_index')}] {doc.page_content[:200]}")
        if doc.metadata.get("has_images"):
            print(f"  📷 Images: {doc.metadata.get('image_paths')}")
        print()
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import os
# import pandas as pd


# df = pd.read_excel(os.path.join(os.path.dirname(__file__), "data", "internshiplist.xlsx"))
# embeddings = OllamaEmbeddings(model="qwen2.5-coder:7b")

# db_location = "./chrome_langchain_db"
# add_documents = not os.path.exists(db_location)

# if add_documents:
#     documents = []
#     id = []

#     for i, row in df.iterrows():
#         title_content = row.iloc[0] 
#         source_content = row.iloc[1]
#         document = Document(page_content=str(title_content), metadata={"source": str(source_content)}, id=str(i))
#         id.append(str(i))
#         documents.append(document)

# vector_store = Chroma(collection_name="internshiplist", embedding_function=embeddings, persist_directory=db_location)

# if add_documents:
#     vector_store.add_documents(documents, ids=id)

# retriever = vector_store.as_retriever(search_kwargs={"k": 5})
