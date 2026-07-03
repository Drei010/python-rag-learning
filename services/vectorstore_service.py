import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from langchain_core.documents import Document

from core.config import settings


@runtime_checkable
class VectorStoreBackend(Protocol):
    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        ...

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        ...

    def delete_by_ids(self, ids: List[str]) -> None:
        ...

    def reset_collection(self) -> None:
        ...

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        ...


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_function: Any,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        from langchain_chroma import Chroma

        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_function": embedding_function,
        }

        if host:
            kwargs["host"] = host
            if port:
                kwargs["port"] = port
            if ssl:
                kwargs["ssl"] = ssl
            if headers:
                kwargs["headers"] = headers
        elif persist_directory:
            kwargs["persist_directory"] = persist_directory

        self._store = Chroma(**kwargs)

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        self._store.add_documents(documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        kwargs: Dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self._store.similarity_search(query, **kwargs)

    def delete_by_ids(self, ids: List[str]) -> None:
        if ids:
            self._store.delete(ids=ids)

    def reset_collection(self) -> None:
        self._store.reset_collection()

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        retriever_kwargs = {}
        if search_kwargs:
            retriever_kwargs["search_kwargs"] = search_kwargs
        return self._store.as_retriever(**retriever_kwargs)


class HanaVectorStore:
    def __init__(
        self,
        embedding_function: Any,
        address: str,
        port: int,
        user: str,
        password: str,
        table_name: str = "LANGCHAIN_VECTORS",
    ) -> None:
        if not address:
            raise ValueError(
                "HANA_DB_ADDRESS must be set when VECTOR_DB_BACKEND=hana."
            )

        self._embedding_function = embedding_function
        self._address = address
        self._port = port
        self._user = user
        self._password = password
        self._table_name = table_name
        self._connection = None
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = self._create_store()
        return self._store

    def _create_store(self):
        try:
            from hdbcli import dbapi
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "hdbcli is required when VECTOR_DB_BACKEND=hana. "
                "Install it with `pip install hdbcli`."
            ) from exc

        try:
            from langchain_hana import HanaDB
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "langchain-hana is required when VECTOR_DB_BACKEND=hana. "
                "Install it with `pip install langchain-hana`."
            ) from exc

        self._connection = dbapi.connect(
            address=self._address,
            port=self._port,
            user=self._user,
            password=self._password,
        )

        return HanaDB(
            connection=self._connection,
            embedding=self._embedding_function,
            table_name=self._table_name,
        )

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        self.store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        kwargs: Dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self.store.similarity_search(query, **kwargs)

    def delete_by_ids(self, ids: List[str]) -> None:
        if ids:
            self.store.delete(ids=ids)

    def reset_collection(self) -> None:
        self.store.delete(filter={})

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        retriever_kwargs = {}
        if search_kwargs:
            retriever_kwargs["search_kwargs"] = search_kwargs
        return self.store.as_retriever(**retriever_kwargs)


def _parse_chroma_headers(headers_str: str) -> Optional[Dict[str, str]]:
    if not headers_str.strip():
        return None

    try:
        parsed = json.loads(headers_str)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def build_vector_store(embedding_function: Any) -> VectorStoreBackend:
    if settings.vector_db_backend == "chroma":
        if settings.chroma_mode == "http":
            return ChromaVectorStore(
                collection_name=settings.collection_name,
                embedding_function=embedding_function,
                host=settings.chroma_host,
                port=settings.chroma_port,
                ssl=settings.chroma_ssl,
                headers=_parse_chroma_headers(settings.chroma_headers),
            )

        return ChromaVectorStore(
            collection_name=settings.collection_name,
            embedding_function=embedding_function,
            persist_directory=str(settings.db_location),
        )

    if settings.vector_db_backend == "hana":
        return HanaVectorStore(
            embedding_function=embedding_function,
            address=settings.hana_db_address,
            port=settings.hana_db_port,
            user=settings.hana_db_user,
            password=settings.hana_db_password,
            table_name=settings.hana_db_table_name,
        )

    raise ValueError(
        "Unsupported VECTOR_DB_BACKEND: "
        f"{settings.vector_db_backend}. Use 'chroma' or 'hana'."
    )
