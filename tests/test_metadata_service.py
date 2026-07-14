import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from services.metadata_service import FileMetadataStore
from services.storage_service import FileDetail, LocalFileStorage


class FileMetadataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_path = Path(self.temp_dir.name) / ".file_metadata.json"
        self.store = FileMetadataStore(self.metadata_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_upload_creates_metadata_file(self) -> None:
        self.store.record_upload("report.pdf", uploaded_by="alice")

        self.assertTrue(self.metadata_path.exists())
        data = json.loads(self.metadata_path.read_text())
        self.assertIn("report.pdf", data)
        self.assertEqual(data["report.pdf"]["uploaded_by"], "alice")
        self.assertIn("uploaded_at", data["report.pdf"])

    def test_record_upload_without_uploaded_by(self) -> None:
        self.store.record_upload("report.pdf")

        data = json.loads(self.metadata_path.read_text())
        self.assertEqual(data["report.pdf"]["uploaded_by"], "")

    def test_get_metadata_returns_stored_data(self) -> None:
        self.store.record_upload("report.pdf", uploaded_by="bob")

        meta = self.store.get_metadata("report.pdf")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["uploaded_by"], "bob")
        self.assertIn("uploaded_at", meta)

    def test_get_metadata_returns_none_for_unknown_file(self) -> None:
        meta = self.store.get_metadata("unknown.pdf")
        self.assertIsNone(meta)

    def test_remove_metadata_deletes_entry(self) -> None:
        self.store.record_upload("report.pdf", uploaded_by="alice")
        self.store.record_upload("other.pdf", uploaded_by="bob")

        self.store.remove_metadata("report.pdf")

        data = json.loads(self.metadata_path.read_text())
        self.assertNotIn("report.pdf", data)
        self.assertIn("other.pdf", data)

    def test_remove_metadata_does_nothing_for_unknown_file(self) -> None:
        self.store.record_upload("report.pdf", uploaded_by="alice")
        self.store.remove_metadata("unknown.pdf")

        data = json.loads(self.metadata_path.read_text())
        self.assertIn("report.pdf", data)

    def test_all_metadata_returns_all_entries(self) -> None:
        self.store.record_upload("a.pdf", uploaded_by="alice")
        self.store.record_upload("b.pdf", uploaded_by="bob")

        all_meta = self.store.all_metadata()
        self.assertEqual(len(all_meta), 2)
        self.assertIn("a.pdf", all_meta)
        self.assertIn("b.pdf", all_meta)

    def test_all_metadata_returns_empty_dict_when_no_file(self) -> None:
        all_meta = self.store.all_metadata()
        self.assertEqual(all_meta, {})

    def test_uploaded_at_is_valid_iso_format(self) -> None:
        self.store.record_upload("report.pdf", uploaded_by="alice")

        meta = self.store.get_metadata("report.pdf")
        parsed = datetime.fromisoformat(meta["uploaded_at"])
        self.assertIsNotNone(parsed.tzinfo)


class LocalFileStorageListFileDetailsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.metadata_path = self.directory / ".file_metadata.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_file(self, filename: str) -> Path:
        path = self.directory / filename
        path.write_text("content")
        return path

    @patch("services.storage_service.settings")
    def test_list_file_details_with_metadata(self, mock_settings) -> None:
        mock_settings.supported_file_extensions = frozenset({".pdf"})
        self._create_file("report.pdf")

        store = FileMetadataStore(self.metadata_path)
        store.record_upload("report.pdf", uploaded_by="alice")

        storage = LocalFileStorage(self.directory)

        with patch("services.metadata_service.file_metadata_store", store):
            details = storage.list_file_details()

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0].filename, "report.pdf")
        self.assertEqual(details[0].uploaded_by, "alice")
        self.assertIsNotNone(details[0].uploaded_at)

    @patch("services.storage_service.settings")
    def test_list_file_details_without_metadata_uses_mtime(
        self, mock_settings
    ) -> None:
        mock_settings.supported_file_extensions = frozenset({".pdf"})
        self._create_file("report.pdf")

        store = FileMetadataStore(self.metadata_path)
        storage = LocalFileStorage(self.directory)

        with patch("services.metadata_service.file_metadata_store", store):
            details = storage.list_file_details()

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0].filename, "report.pdf")
        self.assertIsNone(details[0].uploaded_by)
        self.assertIsNotNone(details[0].uploaded_at)
        # Should be a valid ISO date string
        parsed = datetime.fromisoformat(details[0].uploaded_at)
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
