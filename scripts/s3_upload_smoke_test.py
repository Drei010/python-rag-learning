import asyncio
import io
import sys
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import settings
from services.storage_service import S3FileStorage


SMOKE_TEST_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Count 0 >> endobj\n"
    b"trailer << /Root 1 0 R >>\n"
    b"%%EOF\n"
)


class AsyncBytesUpload:
    def __init__(self, payload: bytes, content_type: str) -> None:
        self._stream = io.BytesIO(payload)
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


async def main() -> int:
    if settings.storage_backend != "s3":
        print("Set FILE_STORAGE_BACKEND=s3 before running this smoke test.")
        return 2

    storage = S3FileStorage(
        bucket=settings.aws_s3_bucket,
        prefix=settings.aws_s3_prefix,
        cache_dir=settings.storage_cache_dir,
        region_name=settings.aws_s3_region_name,
        endpoint_url=settings.aws_s3_endpoint_url,
    )
    filename = f"codex-s3-smoke-{uuid4().hex}.pdf"
    key = storage._key(filename)

    try:
        location = await storage.save_upload_file(
            AsyncBytesUpload(SMOKE_TEST_PDF, "application/pdf"),
            filename,
        )
        storage.client.head_object(Bucket=settings.aws_s3_bucket, Key=key)
        print(f"Uploaded and verified {location}")
        return 0
    finally:
        try:
            storage.delete(filename, missing_ok=True)
            print("Cleaned up smoke-test object.")
        except Exception as exc:
            print(
                f"Cleanup warning for {storage.location(filename)}: {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
