from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from core.config import settings
from services.excel_service import load_excel_documents
from services.pdf_service import load_pdf_documents
from services.ppt_service import load_powerpoint_documents


embeddings = OllamaEmbeddings(model=settings.embedding_model)
vector_store = Chroma(
    collection_name=settings.collection_name,
    embedding_function=embeddings,
    persist_directory=str(settings.db_location),
)

documents: List[Document] = []
ids: List[str] = []
retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})


def refresh_vector_store() -> int:
    global documents, ids, retriever

    excel_documents, excel_ids = load_excel_documents()
    pdf_documents, pdf_ids = load_pdf_documents()
    powerpoint_documents, powerpoint_ids = load_powerpoint_documents()

    documents = excel_documents + pdf_documents + powerpoint_documents
    ids = excel_ids + pdf_ids + powerpoint_ids

    vector_store.reset_collection()

    if documents:
        vector_store.add_documents(documents, ids=ids)

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(documents)


refresh_vector_store()
