from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from core.config import settings


CHUNK_SIZE = 1024 * 1024


class FileStorage(Protocol):
    def exists(self, filename: str) -> bool:
        ...

    def list_filenames(self) -> List[str]:
        ...

    def local_files(self) -> List[Path]:
        ...

    def sync_to_local(self) -> List[Path]:
        ...

    def location(self, filename: str) -> str:
        ...

    async def save_upload_file(self, upload_file, filename: str) -> str:
        ...

    def delete(self, filename: str, missing_ok: bool = False) -> None:
        ...


class LocalFileStorage:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, filename: str) -> Path:
        return self.directory / filename

    def exists(self, filename: str) -> bool:
        return self._path(filename).is_file()

    def list_filenames(self) -> List[str]:
        return [path.name for path in self.local_files()]

    def local_files(self) -> List[Path]:
        if not self.directory.exists():
            return []

        return sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and _is_supported_file(path.name)
        )

    def sync_to_local(self) -> List[Path]:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.local_files()

    def location(self, filename: str) -> str:
        return str(self._path(filename))

    async def save_upload_file(self, upload_file, filename: str) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._path(filename)

        with destination.open("wb") as output_file:
            while chunk := await upload_file.read(CHUNK_SIZE):
                output_file.write(chunk)

        return self.location(filename)

    def delete(self, filename: str, missing_ok: bool = False) -> None:
        self._path(filename).unlink(missing_ok=missing_ok)


@dataclass(frozen=True)
class S3Object:
    filename: str
    key: str


class S3FileStorage:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: Path,
        region_name: str = "",
        endpoint_url: str = "",
    ) -> None:
        if not bucket:
            raise ValueError("AWS_S3_BUCKET must be set when FILE_STORAGE_BACKEND=s3.")

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = cache_dir
        self.region_name = region_name or None
        self.endpoint_url = endpoint_url or None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "boto3 is required when FILE_STORAGE_BACKEND=s3. "
                    "Install it with `pip install boto3`."
                ) from exc

            kwargs = {}
            if self.region_name:
                kwargs["region_name"] = self.region_name
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **kwargs)

        return self._client

    def _key(self, filename: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{filename}"

        return filename

    def _cache_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def _list_objects(self) -> List[S3Object]:
        paginator = self.client.get_paginator("list_objects_v2")
        page_options = {"Bucket": self.bucket}
        if self.prefix:
            page_options["Prefix"] = f"{self.prefix}/"

        objects = []
        for page in paginator.paginate(**page_options):
            for item in page.get("Contents", []):
                key = item["Key"]
                filename = Path(key).name
                if not filename or not _is_supported_file(filename):
                    continue

                if self.prefix:
                    relative_key = key.removeprefix(f"{self.prefix}/")
                else:
                    relative_key = key

                if "/" in relative_key:
                    continue

                objects.append(S3Object(filename=filename, key=key))

        return sorted(objects, key=lambda item: item.filename)

    def exists(self, filename: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(filename))
            return True
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            if error.get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False

            raise

    def list_filenames(self) -> List[str]:
        return [item.filename for item in self._list_objects()]

    def local_files(self) -> List[Path]:
        if not self.cache_dir.exists():
            return []

        return sorted(
            path
            for path in self.cache_dir.iterdir()
            if path.is_file() and _is_supported_file(path.name)
        )

    def sync_to_local(self) -> List[Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        objects = self._list_objects()
        remote_filenames = {item.filename for item in objects}
        for cached_file in self.local_files():
            if cached_file.name not in remote_filenames:
                cached_file.unlink(missing_ok=True)

        for item in objects:
            destination = self._cache_path(item.filename)
            temp_destination = destination.with_suffix(f"{destination.suffix}.tmp")
            self.client.download_file(self.bucket, item.key, str(temp_destination))
            temp_destination.replace(destination)

        return self.local_files()

    def location(self, filename: str) -> str:
        return f"s3://{self.bucket}/{self._key(filename)}"

    async def save_upload_file(self, upload_file, filename: str) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self._cache_path(filename)

        with destination.open("wb") as output_file:
            while chunk := await upload_file.read(CHUNK_SIZE):
                output_file.write(chunk)

        self.client.upload_file(str(destination), self.bucket, self._key(filename))
        return self.location(filename)

    def delete(self, filename: str, missing_ok: bool = False) -> None:
        if not missing_ok and not self.exists(filename):
            raise FileNotFoundError(filename)

        self.client.delete_object(Bucket=self.bucket, Key=self._key(filename))
        self._cache_path(filename).unlink(missing_ok=True)


def _is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in settings.supported_file_extensions


def build_file_storage() -> FileStorage:
    if settings.storage_backend == "local":
        return LocalFileStorage(settings.data_dir)

    if settings.storage_backend == "s3":
        return S3FileStorage(
            bucket=settings.aws_s3_bucket,
            prefix=settings.aws_s3_prefix,
            cache_dir=settings.storage_cache_dir,
            region_name=settings.aws_s3_region_name,
            endpoint_url=settings.aws_s3_endpoint_url,
        )

    raise ValueError(
        "Unsupported FILE_STORAGE_BACKEND: "
        f"{settings.storage_backend}. Use 'local' or 's3'."
    )


file_storage = build_file_storage()
