"""
diagnose.py — Run this BEFORE main.py to find exactly where the pipeline breaks.

Usage:
    python diagnose.py

Each step prints PASS or FAIL with details.
"""

from __future__ import annotations

from pathlib import Path
import sys

EXCEL_PATH = Path(__file__).parent / "data" / "internshiplist.xlsx"
DB_LOCATION = Path("./chroma_langchain_db")
EMBED_MODEL  = "qwen2.5-coder:7b"

SEP = "─" * 55

def section(title: str) -> None:
    print(f"\n{SEP}\nSTEP: {title}\n{SEP}")

# ── STEP 1: Excel file exists ──────────────────────────────
section("1 · Excel file found?")
if not EXCEL_PATH.exists():
    print(f"  FAIL — file not found at: {EXCEL_PATH}")
    sys.exit(1)
print(f"  PASS — {EXCEL_PATH}")

# ── STEP 2: pandas can read it ────────────────────────────
section("2 · pandas can read the file?")
try:
    import pandas as pd
    raw = pd.read_excel(EXCEL_PATH, sheet_name=None, header=0)
    print(f"  PASS — type returned: {type(raw).__name__}")

    if isinstance(raw, dict):
        for sname, sdf in raw.items():
            print(f"    Sheet '{sname}': {len(sdf)} rows × {len(sdf.columns)} cols")
    else:
        print(f"    Single sheet: {len(raw)} rows × {len(raw.columns)} cols")
        print(f"    Columns: {list(raw.columns)}")
        print(raw.head(3).to_string())
except Exception as exc:
    print(f"  FAIL — {exc}")
    sys.exit(1)

# ── STEP 3: build_documents returns rows ─────────────────
section("3 · build_documents() produces documents?")
try:
    from vector import build_documents
    docs = build_documents(EXCEL_PATH, sheet_name=0)
    if not docs:
        print("  FAIL — 0 documents produced. Check that the Excel has data rows.")
        sys.exit(1)
    print(f"  PASS — {len(docs)} document(s) built")
    print(f"\n  First document preview:")
    print(f"    id         : {docs[0].id}")
    print(f"    page_content: {docs[0].page_content[:200]!r}")
    print(f"    metadata   : {docs[0].metadata}")
except Exception as exc:
    print(f"  FAIL — {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── STEP 4: Ollama is reachable ───────────────────────────
section("4 · Ollama embed endpoint reachable?")
try:
    import urllib.request, json
    payload = json.dumps({"model": EMBED_MODEL, "input": ["test"]}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    dim = len(body.get("embeddings", [[]])[0])
    print(f"  PASS — model '{EMBED_MODEL}' returned embedding dim={dim}")
except Exception as exc:
    print(f"  FAIL — {exc}")
    print("  → Is Ollama running?  Run: ollama serve")
    print(f"  → Is the model pulled? Run: ollama pull {EMBED_MODEL}")
    sys.exit(1)

# ── STEP 5: Chroma collection count ──────────────────────
section("5 · Chroma collection document count?")
try:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name="internshiplist",
        embedding_function=embeddings,
        persist_directory=str(DB_LOCATION),
    )
    count = vs._collection.count()
    print(f"  Collection 'internshiplist' holds {count} document(s).")
    if count == 0:
        print("  → Collection is EMPTY — documents were never ingested.")
        print("  → Delete the DB folder and re-run main.py:")
        print(f"     rmdir /s /q {DB_LOCATION}   (Windows)")
        print(f"     rm -rf {DB_LOCATION}          (Mac/Linux)")
    else:
        print("  PASS — collection is populated.")
except Exception as exc:
    print(f"  FAIL — {exc}")
    sys.exit(1)

# ── STEP 6: Test retrieval ────────────────────────────────
section("6 · Retriever returns results?")
try:
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("internship")
    print(f"  Retrieved {len(results)} document(s) for query 'internship'")
    for r in results:
        print(f"    · row {r.metadata.get('row_index', '?')}: {r.page_content[:80]!r}")
    if not results:
        print("  → 0 results even though collection may have docs.")
        print("  → Possible embedding model mismatch between ingest and query.")
except Exception as exc:
    print(f"  FAIL — {exc}")

print(f"\n{SEP}\nDiagnosis complete.\n{SEP}\n")