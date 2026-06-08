import re
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import settings
from services.excel_service import load_excel_documents
from services.pdf_service import load_pdf_documents
from services.ppt_service import load_powerpoint_documents


def create_embeddings():
    if settings.local_llm_hosted:
        from langchain_ollama import OllamaEmbeddings
        print("Using Ollama embeddings")

        return OllamaEmbeddings(model=settings.ollama_embedding_model)

    if settings.hosted_embedding_provider == "local":
        from services.hash_embedding_service import HashEmbeddings

        return HashEmbeddings(dimensions=settings.local_embedding_dimensions)

    if settings.hosted_embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": settings.hosted_embedding_model}
        if settings.hosted_embedding_api_key:
            kwargs["api_key"] = settings.hosted_embedding_api_key
        if settings.hosted_embedding_base_url:
            kwargs["base_url"] = settings.hosted_embedding_base_url

        return OpenAIEmbeddings(**kwargs)

    raise ValueError(
        f"Unsupported hosted embedding provider: {settings.hosted_embedding_provider}"
    )


embeddings = create_embeddings()
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
    print(f"Loaded {len(excel_documents)} Excel documents")
    pdf_documents, pdf_ids = load_pdf_documents()
    powerpoint_documents, powerpoint_ids = load_powerpoint_documents()

    documents = excel_documents + pdf_documents + powerpoint_documents
    ids = excel_ids + pdf_ids + powerpoint_ids

    vector_store.reset_collection()

    if documents:
        vector_store.add_documents(documents, ids=ids)

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(documents)


def get_indexed_sources() -> List[str]:
    return sorted(
        {
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        }
    )


def normalize_search_tokens(text: str) -> List[str]:
    ignored_terms = {
        "the",
        "and",
        "file",
        "files",
        "look",
        "into",
        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 3 and token not in ignored_terms]


def get_documents_matching_source_name(query: str) -> List[Document]:
    query_tokens = normalize_search_tokens(query)
    if not query_tokens:
        return []

    matched_documents = []
    for document in documents:
        source = document.metadata.get("source", "").lower()
        if any(token in source for token in query_tokens):
            matched_documents.append(document)

    return matched_documents


def normalize_source_name(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", source.lower())


def get_sources_matching_query(query: str) -> List[str]:
    query_tokens = normalize_search_tokens(query)
    if not query_tokens:
        return []

    matched_sources = []
    for source in get_indexed_sources():
        normalized_source = normalize_source_name(source)
        if any(token in normalized_source for token in query_tokens):
            matched_sources.append(source)

    return matched_sources


def retrieve_from_all_sources(query: str) -> List[Document]:
    if not documents:
        return []

    retrieved_documents = []
    seen_ids = set()
    matched_sources = get_sources_matching_query(query)

    for document in get_documents_matching_source_name(query):
        document_key = (
            document.metadata.get("source"),
            document.metadata.get("sheet"),
            document.metadata.get("row"),
            document.metadata.get("page"),
            document.metadata.get("slide"),
            document.page_content,
        )
        seen_ids.add(document_key)
        retrieved_documents.append(document)

    target_sources = matched_sources if matched_sources else get_indexed_sources()
    for source in target_sources:
        source_documents = vector_store.similarity_search(
            query,
            k=settings.retriever_k_per_source,
            filter={"source": source},
        )

        for document in source_documents:
            document_key = (
                document.metadata.get("source"),
                document.metadata.get("sheet"),
                document.metadata.get("row"),
                document.metadata.get("page"),
                document.metadata.get("slide"),
                document.page_content,
            )
            if document_key in seen_ids:
                continue

            seen_ids.add(document_key)
            retrieved_documents.append(document)

    return retrieved_documents


refresh_vector_store()
