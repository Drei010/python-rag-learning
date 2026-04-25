"""
main.py — RAG Chat Interface
Connects to the vector store built by excel_rag_ingest.py.

Run:
    python main.py
"""

from __future__ import annotations

import sys
from typing import Generator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM

from vector import build_retriever  # uses the refactored vector module

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "qwen2.5-coder:7b"
MAX_HISTORY_TURNS = 6   # keep last N human/AI pairs in context window
TOP_K = 5


# ─────────────────────────────────────────────
# Context formatter — turns Documents into clean text
# ─────────────────────────────────────────────
def format_context(docs: list[Document]) -> str:
    """
    Convert retrieved Documents into a clean, numbered string the LLM can
    read without being confused by Python object repr.

    Includes source citation and image flag so the LLM can reference them.
    """
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source_line = (
            f"Source: {meta.get('source', 'unknown')}, "
            f"Row: {meta.get('row_index', '?')}"
        )
        if meta.get("has_images"):
            source_line += " [contains image(s)]"

        parts.append(
            f"[{i}] {source_line}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# Prompt — with rolling conversation history
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert internship advisor with access to a structured list of internship opportunities.

Use ONLY the context below to answer. If the answer isn't in the context, say so clearly — do not hallucinate.
When referencing an opportunity, cite its [number] from the context.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),   # rolling conversation turns
    ("human", "{question}"),
])


# ─────────────────────────────────────────────
# Streaming helper
# ─────────────────────────────────────────────
def stream_response(chain, payload: dict) -> str:
    """
    Stream tokens to stdout as they arrive.
    Returns the full assembled response string.
    """
    full_response = ""
    print("Assistant: ", end="", flush=True)

    for chunk in chain.stream(payload):
        print(chunk, end="", flush=True)
        full_response += chunk

    print()  # newline after streaming ends
    return full_response


# ─────────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────────
def main() -> None:
    print("Initialising model and retriever…")

    try:
        model = OllamaLLM(model=MODEL_NAME)
        retriever = build_retriever(sheet_name=0)   # 0 = first sheet by index
    except Exception as exc:
        print(f"[ERROR] Failed to initialise: {exc}")
        sys.exit(1)

    chain = prompt | model | StrOutputParser()

    # Rolling history — list of HumanMessage / AIMessage
    history: list[HumanMessage | AIMessage] = []

    print(f"\nReady. Using model: {MODEL_NAME}")
    print("Type 'q' to quit, 'clear' to reset conversation history.\n")

    while True:
        print("─" * 50)
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "q":
            print("Goodbye.")
            break
        if question.lower() == "clear":
            history.clear()
            print("[History cleared]\n")
            continue

        # ── Retrieve ───────────────────────────────
        try:
            docs = retriever.invoke(question)
        except Exception as exc:
            print(f"[Retrieval ERROR] {exc}\n")
            continue

        context_text = format_context(docs)

        # Optional debug — comment out in production
        print(f"\n[DEBUG] {len(docs)} document(s) retrieved.")
        for doc in docs:
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"  · row {doc.metadata.get('row_index', '?')}: {preview}…")
        print()

        # ── Generate (streaming) ───────────────────
        try:
            payload = {
                "context": context_text,
                "history": history[-MAX_HISTORY_TURNS * 2:],  # trim to N turns
                "question": question,
            }
            answer = stream_response(chain, payload)
        except Exception as exc:
            print(f"\n[Generation ERROR] {exc}\n")
            continue

        # ── Update history ─────────────────────────
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever

# model = OllamaLLM(model="qwen2.5-coder:7b")

# template = """
# You are an expert assistant. Answer the user's question clearly and concisely.

# Context: 
# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# chain = prompt | model

# while True:
#     print("\n\n------------------------------")
#     question = input("Ask your question (q to quit) : ")
#     print("\n\n")
#     if question == "q":
#         break

#     context = retriever.invoke(question)
#     print(f"DEBUG: Found {len(context)} relevant documents.")
#     for d in context:
#         print(f"DEBUG Content: {d.page_content[:50]}...")

#     result = chain.invoke({"context": context, "question": question})
#     print(result)