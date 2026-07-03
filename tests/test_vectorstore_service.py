import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from services.vectorstore_service import (
    ChromaVectorStore,
    HanaVectorStore,
    VectorStoreBackend,
    _parse_chroma_headers,
    build_vector_store,
)


class FakeVectorStore:
    """A minimal class that satisfies the VectorStoreBackend protocol."""

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        pass

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        return []

    def delete_by_ids(self, ids: List[str]) -> None:
        pass

    def reset_collection(self) -> None:
        pass

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return None


class ProtocolTests(unittest.TestCase):
    def test_fake_store_satisfies_protocol(self) -> None:
        store = FakeVectorStore()
        self.assertIsInstance(store, VectorStoreBackend)

    def test_non_conforming_object_fails_protocol(self) -> None:
        self.assertNotIsInstance("not a store", VectorStoreBackend)
        self.assertNotIsInstance(42, VectorStoreBackend)

    def test_chroma_vector_store_satisfies_protocol(self) -> None:
        self.assertTrue(issubclass(ChromaVectorStore, VectorStoreBackend))

    def test_hana_vector_store_satisfies_protocol(self) -> None:
        self.assertTrue(issubclass(HanaVectorStore, VectorStoreBackend))


class ParseChromaHeadersTests(unittest.TestCase):
    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_parse_chroma_headers(""))

    def test_whitespace_returns_none(self) -> None:
        self.assertIsNone(_parse_chroma_headers("   "))

    def test_valid_json_object_returns_dict(self) -> None:
        result = _parse_chroma_headers('{"Authorization": "Bearer token123"}')
        self.assertEqual(result, {"Authorization": "Bearer token123"})

    def test_invalid_json_returns_none(self) -> None:
        self.assertIsNone(_parse_chroma_headers("not json"))

    def test_json_array_returns_none(self) -> None:
        self.assertIsNone(_parse_chroma_headers('["a", "b"]'))


class ChromaVectorStoreTests(unittest.TestCase):
    @patch("services.vectorstore_service.Chroma", create=True)
    def test_local_mode_uses_persist_directory(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        with patch("langchain_chroma.Chroma", mock_chroma_class):
            store = ChromaVectorStore(
                collection_name="test_collection",
                embedding_function=MagicMock(),
                persist_directory="/tmp/chroma_test",
            )

        mock_chroma_class.assert_called_once()
        call_kwargs = mock_chroma_class.call_args[1]
        self.assertEqual(call_kwargs["collection_name"], "test_collection")
        self.assertEqual(call_kwargs["persist_directory"], "/tmp/chroma_test")
        self.assertNotIn("host", call_kwargs)

    @patch("services.vectorstore_service.Chroma", create=True)
    def test_http_mode_uses_host_and_port(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        with patch("langchain_chroma.Chroma", mock_chroma_class):
            store = ChromaVectorStore(
                collection_name="test_collection",
                embedding_function=MagicMock(),
                host="chroma.example.com",
                port=8000,
                ssl=True,
                headers={"Authorization": "Bearer token"},
            )

        mock_chroma_class.assert_called_once()
        call_kwargs = mock_chroma_class.call_args[1]
        self.assertEqual(call_kwargs["host"], "chroma.example.com")
        self.assertEqual(call_kwargs["port"], 8000)
        self.assertTrue(call_kwargs["ssl"])
        self.assertEqual(call_kwargs["headers"], {"Authorization": "Bearer token"})
        self.assertNotIn("persist_directory", call_kwargs)

    @patch("langchain_chroma.Chroma")
    def test_add_documents_delegates(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        docs = [Document(page_content="hello")]
        ids = ["id1"]
        store.add_documents(docs, ids)

        mock_instance.add_documents.assert_called_once_with(docs, ids=ids)

    @patch("langchain_chroma.Chroma")
    def test_similarity_search_delegates(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_instance.similarity_search.return_value = [
            Document(page_content="result")
        ]
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        results = store.similarity_search("query", k=3, filter={"source": "a.pdf"})

        mock_instance.similarity_search.assert_called_once_with(
            "query", k=3, filter={"source": "a.pdf"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "result")

    @patch("langchain_chroma.Chroma")
    def test_reset_collection_delegates(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        store.reset_collection()
        mock_instance.reset_collection.assert_called_once()

    @patch("langchain_chroma.Chroma")
    def test_as_retriever_delegates(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_instance.as_retriever.return_value = mock_retriever
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        retriever = store.as_retriever(search_kwargs={"k": 5})

        mock_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        self.assertIs(retriever, mock_retriever)

    @patch("langchain_chroma.Chroma")
    def test_delete_by_ids_delegates(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        store.delete_by_ids(["id1", "id2"])
        mock_instance.delete.assert_called_once_with(ids=["id1", "id2"])

    @patch("langchain_chroma.Chroma")
    def test_delete_by_ids_empty_list_does_nothing(self, mock_chroma_class) -> None:
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        store = ChromaVectorStore(
            collection_name="test",
            embedding_function=MagicMock(),
            persist_directory="/tmp/test",
        )

        store.delete_by_ids([])
        mock_instance.delete.assert_not_called()


class HanaVectorStoreTests(unittest.TestCase):
    def test_missing_address_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            HanaVectorStore(
                embedding_function=MagicMock(),
                address="",
                port=443,
                user="user",
                password="pass",
            )

        self.assertIn("HANA_DB_ADDRESS", str(ctx.exception))

    def test_lazy_connection_not_created_on_init(self) -> None:
        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        self.assertIsNone(store._store)
        self.assertIsNone(store._connection)

    def test_store_property_creates_connection(self) -> None:
        mock_dbapi = MagicMock()
        mock_connection = MagicMock()
        mock_dbapi.connect.return_value = mock_connection

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        embedding = MagicMock()
        store = HanaVectorStore(
            embedding_function=embedding,
            address="hana.example.com",
            port=443,
            user="testuser",
            password="testpass",
            table_name="MY_VECTORS",
        )

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            result = store.store

        mock_dbapi.connect.assert_called_once_with(
            address="hana.example.com",
            port=443,
            user="testuser",
            password="testpass",
        )
        mock_hana_module.HanaDB.assert_called_once_with(
            connection=mock_connection,
            embedding=embedding,
            table_name="MY_VECTORS",
        )
        self.assertIs(result, mock_hana_instance)

    def test_reset_collection_calls_delete_empty_filter(self) -> None:
        mock_dbapi = MagicMock()
        mock_dbapi.connect.return_value = MagicMock()

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            store.reset_collection()

        mock_hana_instance.delete.assert_called_once_with(filter={})

    def test_add_documents_delegates(self) -> None:
        mock_dbapi = MagicMock()
        mock_dbapi.connect.return_value = MagicMock()

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        docs = [Document(page_content="test")]

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            store.add_documents(docs, ids=["id1"])

        mock_hana_instance.add_documents.assert_called_once_with(docs)

    def test_similarity_search_delegates(self) -> None:
        mock_dbapi = MagicMock()
        mock_dbapi.connect.return_value = MagicMock()

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_instance.similarity_search.return_value = [
            Document(page_content="found")
        ]
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            results = store.similarity_search("query", k=2, filter={"source": "x.pdf"})

        mock_hana_instance.similarity_search.assert_called_once_with(
            "query", k=2, filter={"source": "x.pdf"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "found")

    def test_delete_by_ids_delegates(self) -> None:
        mock_dbapi = MagicMock()
        mock_dbapi.connect.return_value = MagicMock()

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            store.delete_by_ids(["id1", "id2"])

        mock_hana_instance.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_delete_by_ids_empty_list_does_nothing(self) -> None:
        mock_dbapi = MagicMock()
        mock_dbapi.connect.return_value = MagicMock()

        mock_hana_module = MagicMock()
        mock_hana_instance = MagicMock()
        mock_hana_module.HanaDB.return_value = mock_hana_instance

        store = HanaVectorStore(
            embedding_function=MagicMock(),
            address="hana.example.com",
            port=443,
            user="user",
            password="pass",
        )

        with patch.dict(
            "sys.modules",
            {"hdbcli": MagicMock(dbapi=mock_dbapi), "hdbcli.dbapi": mock_dbapi, "langchain_hana": mock_hana_module},
        ):
            store.delete_by_ids([])

        mock_hana_instance.delete.assert_not_called()


class BuildVectorStoreTests(unittest.TestCase):
    @patch("services.vectorstore_service.settings")
    @patch("langchain_chroma.Chroma")
    def test_default_builds_chroma_local(self, mock_chroma, mock_settings) -> None:
        mock_settings.vector_db_backend = "chroma"
        mock_settings.chroma_mode = "local"
        mock_settings.collection_name = "test"
        mock_settings.db_location = "/tmp/db"
        mock_chroma.return_value = MagicMock()

        result = build_vector_store(MagicMock())

        self.assertIsInstance(result, ChromaVectorStore)

    @patch("services.vectorstore_service.settings")
    @patch("langchain_chroma.Chroma")
    def test_chroma_http_mode(self, mock_chroma, mock_settings) -> None:
        mock_settings.vector_db_backend = "chroma"
        mock_settings.chroma_mode = "http"
        mock_settings.collection_name = "test"
        mock_settings.chroma_host = "remote.host"
        mock_settings.chroma_port = 9000
        mock_settings.chroma_ssl = True
        mock_settings.chroma_headers = '{"X-Api-Key": "abc"}'
        mock_chroma.return_value = MagicMock()

        result = build_vector_store(MagicMock())

        self.assertIsInstance(result, ChromaVectorStore)
        call_kwargs = mock_chroma.call_args[1]
        self.assertEqual(call_kwargs["host"], "remote.host")
        self.assertEqual(call_kwargs["port"], 9000)
        self.assertTrue(call_kwargs["ssl"])
        self.assertEqual(call_kwargs["headers"], {"X-Api-Key": "abc"})

    @patch("services.vectorstore_service.settings")
    def test_hana_backend_builds_hana_store(self, mock_settings) -> None:
        mock_settings.vector_db_backend = "hana"
        mock_settings.hana_db_address = "hana.example.com"
        mock_settings.hana_db_port = 443
        mock_settings.hana_db_user = "user"
        mock_settings.hana_db_password = "pass"
        mock_settings.hana_db_table_name = "MY_TABLE"

        result = build_vector_store(MagicMock())

        self.assertIsInstance(result, HanaVectorStore)

    @patch("services.vectorstore_service.settings")
    def test_unsupported_backend_raises_value_error(self, mock_settings) -> None:
        mock_settings.vector_db_backend = "redis"

        with self.assertRaises(ValueError) as ctx:
            build_vector_store(MagicMock())

        self.assertIn("redis", str(ctx.exception))
        self.assertIn("chroma", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
