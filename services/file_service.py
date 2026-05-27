from pathlib import Path
from typing import List

from core.config import settings


def get_supported_files() -> List[Path]:
    if not settings.data_dir.exists():
        return []

    return sorted(
        path
        for path in settings.data_dir.iterdir()
        if path.is_file() and is_supported_file(path.name)
    )


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in settings.supported_file_extensions


def sanitize_filename(filename: str) -> str:
    return Path(filename).name


async def save_upload_file(upload_file, destination: Path) -> None:
    settings.data_dir.mkdir(exist_ok=True)

    with destination.open("wb") as output_file:
        while chunk := await upload_file.read(1024 * 1024):
            output_file.write(chunk)
