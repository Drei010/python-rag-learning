from dataclasses import dataclass
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from core.config import settings


CHUNK_SIZE = 1024 * 1024


@dataclass
class FileDetail:
    filename: str
    uploaded_by: Optional[str] = None
    uploaded_at: Optional[str] = None


class FileStorage(Protocol):
    def exists(self, filename: str) -> bool:
        ...

    def list_filenames(self) -> List[str]:
        ...

    def list_file_details(self) -> List[FileDetail]:
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

    def list_file_details(self) -> List[FileDetail]:
        from services.metadata_service import file_metadata_store

        all_metadata = file_metadata_store.all_metadata()
        details = []
        for path in self.local_files():
            filename = path.name
            meta = all_metadata.get(filename)
            if meta:
                uploaded_by = meta.get("uploaded_by") or None
                uploaded_at = meta.get("uploaded_at") or None
            else:
                uploaded_by = None
                mtime = path.stat().st_mtime
                uploaded_at = datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).isoformat()
            details.append(
                FileDetail(
                    filename=filename,
                    uploaded_by=uploaded_by,
                    uploaded_at=uploaded_at,
                )
            )
        return details

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
    last_modified: Optional[str] = None


class S3FileStorage:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: Path,
        region_name: str = "",
        endpoint_url: str = "",
        client=None,
    ) -> None:
        if not bucket:
            raise ValueError("AWS_S3_BUCKET must be set when FILE_STORAGE_BACKEND=s3.")

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = cache_dir
        self.region_name = region_name or None
        self.endpoint_url = endpoint_url or None
        self._client = client

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

    def _upload_extra_args(self, upload_file, filename: str) -> Dict[str, str]:
        content_type = (
            getattr(upload_file, "content_type", None) or guess_type(filename)[0]
        )
        if not content_type:
            return {}

        return {"ContentType": content_type}

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

    def _list_objects_with_dates(self) -> List[S3Object]:
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

                last_modified = item.get("LastModified")
                last_modified_str = (
                    last_modified.isoformat()
                    if last_modified
                    else None
                )

                objects.append(
                    S3Object(
                        filename=filename,
                        key=key,
                        last_modified=last_modified_str,
                    )
                )

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

    def list_file_details(self) -> List[FileDetail]:
        from services.metadata_service import file_metadata_store

        all_metadata = file_metadata_store.all_metadata()
        details = []
        for item in self._list_objects_with_dates():
            filename = item.filename
            meta = all_metadata.get(filename)
            if meta:
                uploaded_by = meta.get("uploaded_by") or None
                uploaded_at = meta.get("uploaded_at") or None
            else:
                uploaded_by = None
                uploaded_at = item.last_modified
            details.append(
                FileDetail(
                    filename=filename,
                    uploaded_by=uploaded_by,
                    uploaded_at=uploaded_at,
                )
            )
        return details

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

        upload_kwargs = {}
        extra_args = self._upload_extra_args(upload_file, filename)
        if extra_args:
            upload_kwargs["ExtraArgs"] = extra_args

        try:
            self.client.upload_file(
                str(destination),
                self.bucket,
                self._key(filename),
                **upload_kwargs,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

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
