from pathlib import Path
from typing import List

from services.storage_service import file_storage


def get_supported_files() -> List[Path]:
    return file_storage.local_files()


def list_stored_files() -> List[str]:
    return file_storage.list_filenames()


def sync_storage_to_local() -> List[Path]:
    return file_storage.sync_to_local()


def file_exists(filename: str) -> bool:
    return file_storage.exists(filename)


def storage_location(filename: str) -> str:
    return file_storage.location(filename)


def delete_file_from_storage(filename: str, missing_ok: bool = False) -> None:
    file_storage.delete(filename, missing_ok=missing_ok)


def is_supported_file(filename: str) -> bool:
    from core.config import settings

    return Path(filename).suffix.lower() in settings.supported_file_extensions


def sanitize_filename(filename: str) -> str:
    return Path(filename).name


async def save_upload_file(upload_file, filename: str) -> str:
    return await file_storage.save_upload_file(upload_file, filename)
