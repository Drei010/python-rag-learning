import asyncio
import io
import tempfile
import unittest
from pathlib import Path

from services.storage_service import S3FileStorage


class AsyncUpload:
    def __init__(self, payload: bytes, content_type: str = "") -> None:
        self._stream = io.BytesIO(payload)
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads = []

    def upload_file(self, filename, bucket, key, **kwargs) -> None:
        self.uploads.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "kwargs": kwargs,
                "payload": Path(filename).read_bytes(),
            }
        )


class FailingS3Client:
    def upload_file(self, filename, bucket, key, **kwargs) -> None:
        raise RuntimeError("S3 upload failed")


class S3FileStorageTests(unittest.TestCase):
    def test_save_upload_file_caches_file_and_uploads_to_prefixed_key(self) -> None:
        payload = b"%PDF-1.4\n"
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = S3FileStorage(
                bucket="example-bucket",
                prefix="rag-files",
                cache_dir=Path(temp_dir),
                client=client,
            )

            location = asyncio.run(
                storage.save_upload_file(
                    AsyncUpload(payload, content_type="application/pdf"),
                    "report.pdf",
                )
            )

            self.assertEqual(location, "s3://example-bucket/rag-files/report.pdf")
            self.assertEqual((Path(temp_dir) / "report.pdf").read_bytes(), payload)
            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(client.uploads[0]["bucket"], "example-bucket")
            self.assertEqual(client.uploads[0]["key"], "rag-files/report.pdf")
            self.assertEqual(client.uploads[0]["payload"], payload)
            self.assertEqual(
                client.uploads[0]["kwargs"],
                {"ExtraArgs": {"ContentType": "application/pdf"}},
            )

    def test_save_upload_file_removes_cache_file_when_s3_upload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = S3FileStorage(
                bucket="example-bucket",
                prefix="rag-files",
                cache_dir=Path(temp_dir),
                client=FailingS3Client(),
            )

            with self.assertRaises(RuntimeError):
                asyncio.run(
                    storage.save_upload_file(
                        AsyncUpload(b"%PDF-1.4\n", content_type="application/pdf"),
                        "report.pdf",
                    )
                )

            self.assertFalse((Path(temp_dir) / "report.pdf").exists())


if __name__ == "__main__":
    unittest.main()
