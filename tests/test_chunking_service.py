"""Tests for services/chunking_service.py - Agentic semantic chunking."""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from services.chunking_service import (
    FALLBACK_CHUNK_SIZE,
    chunk_documents,
    create_chunking_llm,
    fallback_chunk_document,
    is_chunking_llm_available,
    semantic_chunk_document,
    semantic_chunk_excel_documents,
    _detect_grouping_column,
    _extract_columns_from_content,
    _extract_last_sentences,
    _get_unstructured_sheets,
    _parse_llm_json_response,
    _passthrough_excel_docs,
)


class TestCreateChunkingLlm(unittest.TestCase):
    """Test LLM client creation based on provider settings."""

    @patch("services.chunking_service.settings")
    def test_openai_provider(self, mock_settings):
        mock_settings.hosted_chunking_provider = "openai"
        mock_settings.hosted_chunking_model = "gpt-4o-mini"
        mock_settings.hosted_chunking_api_key = "test-key"
        mock_settings.hosted_chunking_base_url = ""
        mock_settings.hosted_chunking_temperature = 0.7
        mock_settings.hosted_chunking_top_p = 1.0
        mock_settings.hosted_chunking_max_tokens = 2048

        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            llm = create_chunking_llm()
            mock_chat.assert_called_once()
            self.assertIsNotNone(llm)

    @patch("services.chunking_service.settings")
    def test_unsupported_provider_raises(self, mock_settings):
        mock_settings.hosted_chunking_provider = "unsupported"

        with self.assertRaises(ValueError) as ctx:
            create_chunking_llm()
        self.assertIn("unsupported", str(ctx.exception).lower())


class TestIsChunkingLlmAvailable(unittest.TestCase):
    """Test LLM availability check."""

    def test_available_when_llm_responds(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        self.assertTrue(is_chunking_llm_available(mock_llm))

    def test_unavailable_when_llm_raises(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection refused")
        self.assertFalse(is_chunking_llm_available(mock_llm))

    def test_unavailable_when_llm_is_none_and_creation_fails(self):
        with patch("services.chunking_service.create_chunking_llm", side_effect=ValueError("bad")):
            self.assertFalse(is_chunking_llm_available(None))


class TestFallbackChunkDocument(unittest.TestCase):
    """Test the rule-based fallback splitter."""

    def _make_doc(self, content, doc_id="test.pdf:page:1"):
        return Document(
            page_content=content,
            metadata={"source": "test.pdf", "file_type": "pdf", "page": 1},
            id=doc_id,
        )

    def test_short_content_returns_single_chunk(self):
        doc = self._make_doc("Short content here.")
        chunks = fallback_chunk_document(doc)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Short content here.")
        self.assertEqual(chunks[0].id, "test.pdf:page:1:chunk:1")
        self.assertEqual(chunks[0].metadata["chunk_index"], 1)
        self.assertEqual(chunks[0].metadata["source"], "test.pdf")

    def test_long_content_splits_into_multiple_chunks(self):
        # Create content longer than FALLBACK_CHUNK_SIZE
        content = "This is a sentence about topic one. " * 30
        doc = self._make_doc(content)
        chunks = fallback_chunk_document(doc)

        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk.id, f"test.pdf:page:1:chunk:{i}")
            self.assertEqual(chunk.metadata["chunk_index"], i)
            self.assertEqual(chunk.metadata["source"], "test.pdf")
            self.assertEqual(chunk.metadata["page"], 1)

    def test_chunk_size_respected(self):
        content = "Word " * 200  # ~1000 chars
        doc = self._make_doc(content)
        chunks = fallback_chunk_document(doc)

        for chunk in chunks:
            # Allow some tolerance for the splitter
            self.assertLessEqual(len(chunk.page_content), FALLBACK_CHUNK_SIZE + 50)

    def test_metadata_preserved(self):
        doc = Document(
            page_content="A " * 300,
            metadata={"source": "report.pdf", "file_type": "pdf", "page": 5, "custom": "val"},
            id="report.pdf:page:5",
        )
        chunks = fallback_chunk_document(doc)

        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "report.pdf")
            self.assertEqual(chunk.metadata["file_type"], "pdf")
            self.assertEqual(chunk.metadata["page"], 5)
            self.assertEqual(chunk.metadata["custom"], "val")


class TestParseJsonResponse(unittest.TestCase):
    """Test JSON response parsing from LLM."""

    def test_valid_json_array(self):
        response = json.dumps([
            {"title": "Topic A", "content": "Content about A."},
            {"title": "Topic B", "content": "Content about B."},
        ])
        result = _parse_llm_json_response(response)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Topic A")

    def test_json_with_code_fences(self):
        response = '```json\n[{"title": "T", "content": "C"}]\n```'
        result = _parse_llm_json_response(response)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_malformed_json_returns_none(self):
        self.assertIsNone(_parse_llm_json_response("not json at all"))

    def test_missing_keys_returns_none(self):
        response = json.dumps([{"title": "T"}])  # missing "content"
        self.assertIsNone(_parse_llm_json_response(response))

    def test_non_array_returns_none(self):
        response = json.dumps({"title": "T", "content": "C"})
        self.assertIsNone(_parse_llm_json_response(response))


class TestExtractLastSentences(unittest.TestCase):
    """Test sentence extraction for overlap."""

    def test_extracts_last_two_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _extract_last_sentences(text, 2)
        self.assertEqual(result, "Second sentence. Third sentence.")

    def test_short_text_returns_all(self):
        text = "Only one sentence."
        result = _extract_last_sentences(text, 2)
        self.assertEqual(result, "Only one sentence.")


class TestSemanticChunkDocument(unittest.TestCase):
    """Test LLM-based semantic chunking."""

    def _make_doc(self, content, doc_id="report.pdf:page:1"):
        return Document(
            page_content=content,
            metadata={"source": "report.pdf", "file_type": "pdf", "page": 1},
            id=doc_id,
        )

    def test_short_content_returns_single_chunk(self):
        doc = self._make_doc("Short text.")
        mock_llm = MagicMock()
        chunks = semantic_chunk_document(doc, mock_llm)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "report.pdf:page:1:chunk:1")
        mock_llm.invoke.assert_not_called()

    def test_valid_llm_response_produces_semantic_chunks(self):
        content = "A " * 300  # Long enough to trigger LLM call
        doc = self._make_doc(content)

        llm_response = MagicMock()
        llm_response.content = json.dumps([
            {"title": "Introduction", "content": "This is the intro section."},
            {"title": "Details", "content": "Here are the details."},
            {"title": "Conclusion", "content": "Final thoughts here."},
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = llm_response

        chunks = semantic_chunk_document(doc, mock_llm)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].id, "report.pdf:page:1:chunk:1")
        self.assertEqual(chunks[1].id, "report.pdf:page:1:chunk:2")
        self.assertEqual(chunks[2].id, "report.pdf:page:1:chunk:3")
        self.assertEqual(chunks[0].metadata["chunk_title"], "Introduction")
        self.assertEqual(chunks[1].metadata["chunk_title"], "Details")

    def test_overlap_present_in_second_chunk(self):
        content = "A " * 300
        doc = self._make_doc(content)

        llm_response = MagicMock()
        llm_response.content = json.dumps([
            {"title": "Part 1", "content": "First part content. End of part one."},
            {"title": "Part 2", "content": "Second part content."},
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = llm_response

        chunks = semantic_chunk_document(doc, mock_llm)

        # Second chunk should contain overlap from first chunk
        self.assertIn("End of part one.", chunks[1].page_content)
        self.assertIn("Second part content.", chunks[1].page_content)

    def test_malformed_llm_response_falls_back(self):
        content = "A " * 300
        doc = self._make_doc(content)

        llm_response = MagicMock()
        llm_response.content = "This is not JSON"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = llm_response

        chunks = semantic_chunk_document(doc, mock_llm)

        # Should fall back to rule-based splitting
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn(":chunk:", chunk.id)

    def test_llm_exception_falls_back(self):
        content = "A " * 300
        doc = self._make_doc(content)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")

        chunks = semantic_chunk_document(doc, mock_llm)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn(":chunk:", chunk.id)


class TestExtractColumnsFromContent(unittest.TestCase):
    """Test column extraction from document page_content."""

    def test_extracts_columns(self):
        content = "Source file: test.xlsx\nSheet: Sheet1\nRow: 0\nDepartment: Engineering\nName: Alice"
        result = _extract_columns_from_content(content)
        self.assertEqual(result["Department"], "Engineering")
        self.assertEqual(result["Name"], "Alice")
        self.assertNotIn("Source file", result)
        self.assertNotIn("Sheet", result)
        self.assertNotIn("Row", result)

    def test_empty_content(self):
        result = _extract_columns_from_content("")
        self.assertEqual(result, {})


class TestDetectGroupingColumn(unittest.TestCase):
    """Test grouping column detection for Excel sheets."""

    def _make_excel_doc(self, dept, name, row_index):
        content = f"Source file: test.xlsx\nSheet: Sheet1\nRow: {row_index}\nDepartment: {dept}\nName: {name}"
        return Document(
            page_content=content,
            metadata={"source": "test.xlsx", "sheet": "Sheet1", "row": row_index},
            id=f"test.xlsx:Sheet1:{row_index}",
        )

    def test_detects_category_column(self):
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        column = _detect_grouping_column(docs)
        self.assertEqual(column, "Department")

    def test_returns_none_for_few_docs(self):
        docs = [self._make_excel_doc("Eng", "Alice", 0)]
        self.assertIsNone(_detect_grouping_column(docs))

    def test_returns_none_when_all_values_unique(self):
        docs = [
            self._make_excel_doc(f"Dept{i}", f"Person{i}", i)
            for i in range(5)
        ]
        # Both columns have unique values for each row
        self.assertIsNone(_detect_grouping_column(docs))


class TestSemanticChunkExcelDocuments(unittest.TestCase):
    """Test hybrid Excel chunking with grouping + LLM titles."""

    def _make_excel_doc(self, dept, name, row_index, sheet="Sheet1"):
        content = f"Source file: data.xlsx\nSheet: {sheet}\nRow: {row_index}\nDepartment: {dept}\nName: {name}"
        return Document(
            page_content=content,
            metadata={"source": "data.xlsx", "file_type": "excel", "sheet": sheet, "row": row_index},
            id=f"data.xlsx:{sheet}:{row_index}",
        )

    def test_few_rows_returns_unchanged_with_chunk_ids(self):
        docs = [self._make_excel_doc("Eng", "Alice", 0)]
        chunks = semantic_chunk_excel_documents(docs, llm_available=False)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "data.xlsx:Sheet1:0:chunk:1")

    def test_grouping_without_llm_uses_column_value_as_title(self):
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        chunks = semantic_chunk_excel_documents(docs, llm_available=False)

        # Should produce 2 group chunks
        self.assertEqual(len(chunks), 2)
        titles = [c.page_content.split("\n")[0] for c in chunks]
        self.assertTrue(any("Engineering" in t for t in titles))
        self.assertTrue(any("Marketing" in t for t in titles))

    def test_grouping_with_llm_generates_titles(self):
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Team members summary")

        chunks = semantic_chunk_excel_documents(docs, llm_available=True, llm=mock_llm)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(mock_llm.invoke.called)

    def test_metadata_includes_row_range(self):
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 5),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        chunks = semantic_chunk_excel_documents(docs, llm_available=False)

        for chunk in chunks:
            self.assertIn("row_range", chunk.metadata)
            self.assertIn("group_label", chunk.metadata)
            self.assertIn("sheet", chunk.metadata)

    def test_empty_input_returns_empty(self):
        self.assertEqual(semantic_chunk_excel_documents([], llm_available=False), [])


class TestChunkDocuments(unittest.TestCase):
    """Test the main entry point that gates on CHUNKING_MODE."""

    def _make_pdf_doc(self, content="Short.", doc_id="f.pdf:page:1"):
        return Document(
            page_content=content,
            metadata={"source": "f.pdf", "file_type": "pdf", "page": 1},
            id=doc_id,
        )

    @patch("services.chunking_service.settings")
    def test_local_mode_returns_unchanged(self, mock_settings):
        mock_settings.use_local_chunking = True

        docs = [self._make_pdf_doc()]
        ids = ["f.pdf:page:1"]

        result_docs, result_ids = chunk_documents(docs, ids)

        self.assertEqual(result_docs, docs)
        self.assertEqual(result_ids, ids)

    @patch("services.chunking_service.settings")
    @patch("services.chunking_service.create_chunking_llm")
    @patch("services.chunking_service.is_chunking_llm_available")
    def test_hosted_mode_with_unavailable_llm_uses_fallback(
        self, mock_available, mock_create, mock_settings
    ):
        mock_settings.use_local_chunking = False
        mock_create.return_value = MagicMock()
        mock_available.return_value = False

        content = "Word " * 200  # Long enough to split
        docs = [self._make_pdf_doc(content)]
        ids = ["f.pdf:page:1"]

        result_docs, result_ids = chunk_documents(docs, ids)

        # Should produce multiple chunks via fallback
        self.assertGreater(len(result_docs), 1)
        for doc in result_docs:
            self.assertIn(":chunk:", doc.id)

    @patch("services.chunking_service.settings")
    @patch("services.chunking_service.create_chunking_llm")
    @patch("services.chunking_service.is_chunking_llm_available")
    def test_hosted_mode_with_available_llm_uses_semantic(
        self, mock_available, mock_create, mock_settings
    ):
        mock_settings.use_local_chunking = False
        mock_available.return_value = True

        mock_llm = MagicMock()
        llm_response = MagicMock()
        llm_response.content = json.dumps([
            {"title": "Section 1", "content": "Content one."},
            {"title": "Section 2", "content": "Content two."},
        ])
        mock_llm.invoke.return_value = llm_response
        mock_create.return_value = mock_llm

        content = "A " * 300  # Long enough to trigger LLM
        docs = [self._make_pdf_doc(content)]
        ids = ["f.pdf:page:1"]

        result_docs, result_ids = chunk_documents(docs, ids)

        self.assertEqual(len(result_docs), 2)
        self.assertEqual(result_docs[0].metadata["chunk_title"], "Section 1")

    @patch("services.chunking_service.settings")
    @patch("services.chunking_service.create_chunking_llm")
    @patch("services.chunking_service.is_chunking_llm_available")
    def test_short_doc_in_hosted_mode_not_split(
        self, mock_available, mock_create, mock_settings
    ):
        mock_settings.use_local_chunking = False
        mock_available.return_value = True
        mock_create.return_value = MagicMock()

        docs = [self._make_pdf_doc("Tiny.")]
        ids = ["f.pdf:page:1"]

        result_docs, result_ids = chunk_documents(docs, ids)

        self.assertEqual(len(result_docs), 1)
        self.assertEqual(result_docs[0].id, "f.pdf:page:1:chunk:1")

    @patch("services.chunking_service.settings")
    @patch("services.chunking_service.create_chunking_llm")
    @patch("services.chunking_service.is_chunking_llm_available")
    def test_chunk_ids_are_unique(self, mock_available, mock_create, mock_settings):
        mock_settings.use_local_chunking = False
        mock_available.return_value = False
        mock_create.return_value = MagicMock()

        content = "Word " * 200
        docs = [
            self._make_pdf_doc(content, "a.pdf:page:1"),
            self._make_pdf_doc(content, "a.pdf:page:2"),
        ]
        docs[1].metadata["page"] = 2
        ids = ["a.pdf:page:1", "a.pdf:page:2"]

        result_docs, result_ids = chunk_documents(docs, ids)

        # All IDs should be unique
        self.assertEqual(len(result_ids), len(set(result_ids)))


class TestClassifierAwareChunking(unittest.TestCase):
    """Test that classifier results control structured vs unstructured behavior."""

    def _make_excel_doc(self, dept, name, row_index, sheet="Sheet1", source="data.xlsx"):
        content = f"Source file: {source}\nSheet: {sheet}\nRow: {row_index}\nDepartment: {dept}\nName: {name}"
        return Document(
            page_content=content,
            metadata={"source": source, "file_type": "excel", "sheet": sheet, "row": row_index},
            id=f"{source}:{sheet}:{row_index}",
        )

    def test_structured_sheet_passes_through(self):
        """Structured sheets should keep row-by-row, no grouping."""
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Marketing", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        # Sheet is NOT in unstructured_sheets -> structured -> passthrough
        unstructured_sheets = set()

        chunks = semantic_chunk_excel_documents(
            docs, llm_available=True, llm=MagicMock(), unstructured_sheets=unstructured_sheets
        )

        # Should have 5 docs, one per row, unchanged content
        self.assertEqual(len(chunks), 5)
        for chunk in chunks:
            self.assertTrue(chunk.id.endswith(":chunk:1"))
            self.assertEqual(chunk.metadata["chunk_index"], 1)

    def test_unstructured_sheet_gets_agentic_chunking(self):
        """Unstructured sheets should proceed with grouping."""
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]
        # Sheet IS in unstructured_sheets -> agentic chunking
        unstructured_sheets = {("data.xlsx", "Sheet1")}

        chunks = semantic_chunk_excel_documents(
            docs, llm_available=False, unstructured_sheets=unstructured_sheets
        )

        # Should produce grouped chunks (2 groups: Engineering, Marketing)
        self.assertEqual(len(chunks), 2)
        titles = [c.page_content.split("\n")[0] for c in chunks]
        self.assertTrue(any("Engineering" in t for t in titles))
        self.assertTrue(any("Marketing" in t for t in titles))

    def test_mixed_sheets_different_behavior(self):
        """Mixed file: structured sheet passes through, unstructured gets chunked."""
        structured_docs = [
            self._make_excel_doc("Eng", f"Person{i}", i, sheet="Structured")
            for i in range(5)
        ]
        unstructured_docs = [
            self._make_excel_doc("Engineering", "Alice", 0, sheet="Unstructured"),
            self._make_excel_doc("Engineering", "Bob", 1, sheet="Unstructured"),
            self._make_excel_doc("Engineering", "Charlie", 2, sheet="Unstructured"),
            self._make_excel_doc("Marketing", "Dave", 3, sheet="Unstructured"),
            self._make_excel_doc("Marketing", "Eve", 4, sheet="Unstructured"),
        ]
        all_docs = structured_docs + unstructured_docs
        unstructured_sheets = {("data.xlsx", "Unstructured")}

        chunks = semantic_chunk_excel_documents(
            all_docs, llm_available=False, unstructured_sheets=unstructured_sheets
        )

        # Structured: 5 passthrough docs. Unstructured: 2 grouped docs.
        structured_chunks = [c for c in chunks if c.metadata.get("sheet") == "Structured"]
        unstructured_chunks = [c for c in chunks if c.metadata.get("sheet") == "Unstructured"]

        self.assertEqual(len(structured_chunks), 5)
        self.assertEqual(len(unstructured_chunks), 2)

    def test_none_unstructured_sheets_treats_all_as_unstructured(self):
        """When unstructured_sheets=None, backward compat: all sheets get chunked."""
        docs = [
            self._make_excel_doc("Engineering", "Alice", 0),
            self._make_excel_doc("Engineering", "Bob", 1),
            self._make_excel_doc("Engineering", "Charlie", 2),
            self._make_excel_doc("Marketing", "Dave", 3),
            self._make_excel_doc("Marketing", "Eve", 4),
        ]

        chunks = semantic_chunk_excel_documents(
            docs, llm_available=False, unstructured_sheets=None
        )

        # Should produce grouped chunks (not passthrough)
        self.assertEqual(len(chunks), 2)


class TestGetUnstructuredSheets(unittest.TestCase):
    """Test the _get_unstructured_sheets helper."""

    def test_filters_unstructured_only(self):
        from services.classifier_service import SheetClassification

        classifications = [
            SheetClassification(file="a.xlsx", sheet="Sheet1", classification="structured", reason="ok"),
            SheetClassification(file="a.xlsx", sheet="Sheet2", classification="unstructured", reason="sparse"),
            SheetClassification(file="b.xlsx", sheet="Data", classification="structured", reason="ok"),
        ]
        result = _get_unstructured_sheets(classifications)
        self.assertEqual(result, {("a.xlsx", "Sheet2")})

    def test_empty_classifications(self):
        result = _get_unstructured_sheets([])
        self.assertEqual(result, set())


class TestPassthroughExcelDocs(unittest.TestCase):
    """Test the _passthrough_excel_docs helper."""

    def test_appends_chunk_1_to_ids(self):
        docs = [
            Document(
                page_content="content",
                metadata={"source": "f.xlsx", "sheet": "S1", "row": 0},
                id="f.xlsx:S1:0",
            )
        ]
        result = _passthrough_excel_docs(docs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "f.xlsx:S1:0:chunk:1")
        self.assertEqual(result[0].metadata["chunk_index"], 1)
        self.assertEqual(result[0].page_content, "content")


if __name__ == "__main__":
    unittest.main()
