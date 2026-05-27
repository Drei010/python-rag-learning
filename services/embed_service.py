from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from core.config import settings
from services.excel_service import load_excel_documents


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

    documents, ids = load_excel_documents()
    vector_store.reset_collection()

    if documents:
        vector_store.add_documents(documents, ids=ids)

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(documents)


refresh_vector_store()
