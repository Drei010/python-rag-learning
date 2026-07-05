"""Agentic semantic chunking service.

When CHUNKING_MODE=hosted, documents are split into meaning-based chunks using an
LLM. When CHUNKING_MODE=local, documents pass through unchanged (current behavior).

If the hosted LLM is unreachable, a rule-based RecursiveCharacterTextSplitter is
used as a fallback.
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from services.classifier_service import SheetClassification, classify_excel_files


SEMANTIC_CHUNK_PROMPT = """You are a document chunking assistant. Split the following text into coherent topical sections.
Return ONLY a JSON array where each element has "title" and "content" keys.
Each section should be a self-contained topic. Do not omit any content from the original text.
If the text is already about a single topic, return a single-element array.

Text:
{text}"""

EXCEL_GROUP_SUMMARY_PROMPT = """Summarize what these rows are about in one sentence. Return only the summary.

Rows:
{rows}"""

FALLBACK_CHUNK_SIZE = 512
FALLBACK_CHUNK_OVERLAP = 80
FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ---------------------------------------------------------------------------
# LLM client creation
# ---------------------------------------------------------------------------


def create_chunking_llm():
    """Create the LLM client for semantic chunking based on hosted chunking settings."""
    provider = settings.hosted_chunking_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": settings.hosted_chunking_model,
            "temperature": settings.hosted_chunking_temperature,
            "max_tokens": settings.hosted_chunking_max_tokens,
            "model_kwargs": {"top_p": settings.hosted_chunking_top_p},
        }
        if settings.hosted_chunking_api_key:
            kwargs["api_key"] = settings.hosted_chunking_api_key
        if settings.hosted_chunking_base_url:
            kwargs["base_url"] = settings.hosted_chunking_base_url

        return ChatOpenAI(**kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        kwargs = {
            "model": settings.hosted_chunking_model,
            "temperature": settings.hosted_chunking_temperature,
            "max_tokens": settings.hosted_chunking_max_tokens,
            "model_kwargs": {"top_p": settings.hosted_chunking_top_p},
        }
        if settings.hosted_chunking_api_key:
            kwargs["api_key"] = settings.hosted_chunking_api_key
        if settings.hosted_chunking_base_url:
            kwargs["groq_api_base"] = settings.hosted_chunking_base_url

        return ChatGroq(**kwargs)

    raise ValueError(
        f"Unsupported hosted chunking provider: {provider}. "
        "Set HOSTED_CHUNKING_PROVIDER to 'openai' or 'groq'."
    )


def is_chunking_llm_available(llm=None) -> bool:
    """Check if the chunking LLM is reachable with a lightweight test call."""
    if llm is None:
        try:
            llm = create_chunking_llm()
        except Exception:
            return False

    try:
        response = llm.invoke("Respond with the single word: ok")
        return bool(response)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fallback rule-based splitter
# ---------------------------------------------------------------------------


_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FALLBACK_CHUNK_SIZE,
    chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    separators=FALLBACK_SEPARATORS,
)


def fallback_chunk_document(document: Document) -> List[Document]:
    """Split a document using RecursiveCharacterTextSplitter as a fallback.

    If the content is shorter than chunk_size, return it as a single chunk.
    Each chunk gets an ID of format: {original_id}:chunk:{index}
    """
    original_id = document.id or ""
    content = document.page_content

    if len(content) <= FALLBACK_CHUNK_SIZE:
        return [
            Document(
                page_content=content,
                metadata={**document.metadata, "chunk_index": 1},
                id=f"{original_id}:chunk:1",
            )
        ]

    texts = _fallback_splitter.split_text(content)

    chunks = []
    for index, text in enumerate(texts, start=1):
        chunks.append(
            Document(
                page_content=text,
                metadata={**document.metadata, "chunk_index": index},
                id=f"{original_id}:chunk:{index}",
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking for PDF/PPT via LLM
# ---------------------------------------------------------------------------


def _extract_last_sentences(text: str, count: int = 2) -> str:
    """Extract the last N sentences from text for overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= count:
        return text.strip()
    return " ".join(sentences[-count:])


def _parse_llm_json_response(response_text: str) -> Optional[List[Dict[str, str]]]:
    """Parse LLM response as JSON, handling markdown code fences."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, list):
        return None

    for item in parsed:
        if not isinstance(item, dict):
            return None
        if "title" not in item or "content" not in item:
            return None

    return parsed


def semantic_chunk_document(document: Document, llm) -> List[Document]:
    """Split a document into semantic chunks using an LLM.

    The LLM identifies topic boundaries and returns titled sections.
    Overlap of 1-2 sentences is added between consecutive chunks.
    Falls back to rule-based splitting if the LLM response is malformed.
    """
    original_id = document.id or ""
    content = document.page_content

    # Don't bother chunking very short content
    if len(content) <= FALLBACK_CHUNK_SIZE:
        return [
            Document(
                page_content=content,
                metadata={**document.metadata, "chunk_index": 1, "chunk_title": ""},
                id=f"{original_id}:chunk:1",
            )
        ]

    prompt = SEMANTIC_CHUNK_PROMPT.format(text=content)

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return fallback_chunk_document(document)

    parsed = _parse_llm_json_response(response_text)
    if parsed is None:
        return fallback_chunk_document(document)

    # Filter out empty chunks
    parsed = [chunk for chunk in parsed if chunk.get("content", "").strip()]
    if not parsed:
        return fallback_chunk_document(document)

    chunks = []
    for index, chunk_data in enumerate(parsed, start=1):
        title = chunk_data["title"].strip()
        chunk_content = chunk_data["content"].strip()

        # Add overlap: prepend last 1-2 sentences of previous chunk
        if index > 1 and chunks:
            prev_content = parsed[index - 2]["content"].strip()
            overlap = _extract_last_sentences(prev_content)
            chunk_content = f"{overlap}\n\n{chunk_content}"

        chunks.append(
            Document(
                page_content=chunk_content,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_title": title,
                },
                id=f"{original_id}:chunk:{index}",
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Classifier-aware helpers
# ---------------------------------------------------------------------------


def _get_unstructured_sheets(
    classifications: List[SheetClassification],
) -> Set[Tuple[str, str]]:
    """Return a set of (file, sheet) tuples classified as unstructured."""
    return {
        (c.file, c.sheet)
        for c in classifications
        if c.classification == "unstructured"
    }


def _passthrough_excel_docs(sheet_docs: List[Document]) -> List[Document]:
    """Return Excel documents unchanged with :chunk:1 appended to IDs."""
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": 1},
            id=f"{doc.id}:chunk:1",
        )
        for doc in sheet_docs
    ]


# ---------------------------------------------------------------------------
# Hybrid chunking for Excel sheets
# ---------------------------------------------------------------------------


def _extract_columns_from_content(content: str) -> Dict[str, str]:
    """Extract column:value pairs from a document's page_content.

    Expected format per line: "ColumnName: value"
    Skips header lines like "Source file:", "Sheet:", "Row:".
    """
    columns = {}
    skip_prefixes = ("source file:", "sheet:", "row:", "file type:")

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(skip_prefixes):
            continue
        if ": " in line:
            col_name, col_value = line.split(": ", 1)
            columns[col_name.strip()] = col_value.strip()

    return columns


def _detect_grouping_column(documents: List[Document]) -> Optional[str]:
    """Find the best column to group rows by.

    Looks for a text column with the fewest distinct values relative to
    document count (i.e., a category column like Department, Type, etc.).
    Requires at least 2 documents per group on average to be useful.
    """
    if len(documents) <= 3:
        return None

    # Collect all column values across documents
    column_values: Dict[str, List[str]] = defaultdict(list)
    for doc in documents:
        columns = _extract_columns_from_content(doc.page_content)
        for col_name, col_value in columns.items():
            column_values[col_name].append(col_value)

    best_column = None
    best_ratio = float("inf")

    for col_name, values in column_values.items():
        # Skip columns that don't appear in most documents
        if len(values) < len(documents) * 0.7:
            continue

        distinct_count = len(set(values))

        # Skip columns where every value is unique (not a category)
        if distinct_count >= len(values) * 0.8:
            continue

        # Skip columns with only 1 distinct value (no grouping possible)
        if distinct_count <= 1:
            continue

        # Ratio of distinct values to total values (lower = better grouping column)
        ratio = distinct_count / len(values)
        if ratio < best_ratio:
            best_ratio = ratio
            best_column = col_name

    return best_column


def semantic_chunk_excel_documents(
    documents: List[Document],
    llm_available: bool,
    llm=None,
    unstructured_sheets: Optional[Set[Tuple[str, str]]] = None,
) -> List[Document]:
    """Group Excel row documents by a detected category column, then title each group.

    Uses classifier results to decide per-sheet behavior:
    - Structured sheets: skip semantic chunking, keep row-by-row (passthrough).
    - Unstructured sheets: proceed with agentic chunking.

    If no classifications are provided (unstructured_sheets is None), all sheets
    are treated as unstructured (backward-compatible behavior).

    If LLM is available, it generates a summary title for each group.
    If not, the column value itself is used as the group title.
    If no good grouping column is found or sheet has ≤3 rows, documents are
    returned with :chunk:1 appended to IDs.
    """
    if not documents:
        return []

    # Group documents by (source, sheet)
    sheets: Dict[Tuple[str, str], List[Document]] = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "")
        sheet_name = doc.metadata.get("sheet", "default")
        sheets[(source, sheet_name)].append(doc)

    result_chunks = []

    for (source, sheet_name), sheet_docs in sheets.items():
        # Check classification: if structured, pass through unchanged
        if unstructured_sheets is not None and (source, sheet_name) not in unstructured_sheets:
            print(f"  Sheet '{source}/{sheet_name}': structured -> keeping row-by-row")
            result_chunks.extend(_passthrough_excel_docs(sheet_docs))
            continue

        if unstructured_sheets is not None:
            print(f"  Sheet '{source}/{sheet_name}': unstructured -> agentic chunking")

        # If too few rows, don't bother grouping
        if len(sheet_docs) <= 3:
            result_chunks.extend(_passthrough_excel_docs(sheet_docs))
            continue

        grouping_column = _detect_grouping_column(sheet_docs)

        # No good grouping column found - return as individual chunks
        if grouping_column is None:
            result_chunks.extend(_passthrough_excel_docs(sheet_docs))
            continue

        # Group documents by the column value
        groups: Dict[str, List[Document]] = defaultdict(list)
        ungrouped: List[Document] = []
        for doc in sheet_docs:
            columns = _extract_columns_from_content(doc.page_content)
            group_value = columns.get(grouping_column)
            if group_value:
                groups[group_value].append(doc)
            else:
                ungrouped.append(doc)

        # Process each group
        chunk_index_in_sheet = 0

        for group_label, group_docs in groups.items():
            chunk_index_in_sheet += 1

            # Determine row range
            rows = []
            for doc in group_docs:
                row = doc.metadata.get("row")
                if row is not None:
                    rows.append(int(row))
            row_range = f"{min(rows)}-{max(rows)}" if rows else "0-0"

            # Generate title
            if llm_available and llm is not None:
                rows_text = "\n".join(doc.page_content for doc in group_docs)
                prompt = EXCEL_GROUP_SUMMARY_PROMPT.format(rows=rows_text)
                try:
                    response = llm.invoke(prompt)
                    title = (
                        response.content.strip()
                        if hasattr(response, "content")
                        else str(response).strip()
                    )
                except Exception:
                    title = group_label
            else:
                title = group_label

            # Build group document
            group_content = f"Group: {title}\n\n" + "\n\n".join(
                doc.page_content for doc in group_docs
            )

            result_chunks.append(
                Document(
                    page_content=group_content,
                    metadata={
                        "source": source,
                        "file_type": "excel",
                        "sheet": sheet_name,
                        "row_range": row_range,
                        "group_label": group_label,
                        "chunk_index": chunk_index_in_sheet,
                    },
                    id=f"{source}:{sheet_name}:group:{row_range}:chunk:{chunk_index_in_sheet}",
                )
            )

        # Handle ungrouped documents
        result_chunks.extend(_passthrough_excel_docs(ungrouped))

    return result_chunks


# ---------------------------------------------------------------------------
# Main entry point: chunk a list of documents based on CHUNKING_MODE
# ---------------------------------------------------------------------------


def chunk_documents(
    documents: List[Document], ids: List[str]
) -> Tuple[List[Document], List[str]]:
    """Chunk documents based on CHUNKING_MODE setting.

    If CHUNKING_MODE=local: return documents and ids unchanged.
    If CHUNKING_MODE=hosted: use semantic chunking via LLM with fallback.
      - For Excel: uses classifier_service to determine structured vs unstructured.
        Structured sheets pass through as row-by-row. Unstructured sheets get
        agentic chunking.
      - For PDF/PPT: always uses semantic chunking (or fallback).
    """
    if settings.use_local_chunking:
        return documents, ids

    # Hosted mode: attempt LLM-based semantic chunking
    llm = None
    llm_available = False

    try:
        llm = create_chunking_llm()
        llm_available = is_chunking_llm_available(llm)
    except Exception:
        llm_available = False

    if llm_available:
        print("Semantic chunking: using hosted LLM")
    else:
        print("Semantic chunking: LLM unavailable, using fallback splitter")

    # Separate by file type
    excel_docs = []
    other_docs = []

    for doc in documents:
        file_type = doc.metadata.get("file_type", "")
        if file_type == "excel":
            excel_docs.append(doc)
        else:
            other_docs.append(doc)

    chunked_documents: List[Document] = []

    # Process PDF/PPT documents
    for doc in other_docs:
        if llm_available and llm is not None:
            chunks = semantic_chunk_document(doc, llm)
        else:
            chunks = fallback_chunk_document(doc)
        chunked_documents.extend(chunks)

    # Process Excel documents (classifier-aware)
    if excel_docs:
        classifications = classify_excel_files()
        unstructured_sheets = _get_unstructured_sheets(classifications)
        excel_chunks = semantic_chunk_excel_documents(
            excel_docs, llm_available, llm, unstructured_sheets
        )
        chunked_documents.extend(excel_chunks)

    chunked_ids = [doc.id for doc in chunked_documents]
    return chunked_documents, chunked_ids
