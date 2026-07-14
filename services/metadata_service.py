import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from core.config import settings

METADATA_FILENAME = ".file_metadata.json"


class FileMetadataStore:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.metadata_path.exists():
            return {}

        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, Dict[str, str]]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def record_upload(
        self, filename: str, uploaded_by: Optional[str] = None
    ) -> None:
        data = self._load()
        data[filename] = {
            "uploaded_by": uploaded_by or "",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(data)

    def get_metadata(self, filename: str) -> Optional[Dict[str, str]]:
        data = self._load()
        return data.get(filename)

    def remove_metadata(self, filename: str) -> None:
        data = self._load()
        if filename in data:
            del data[filename]
            self._save(data)

    def all_metadata(self) -> Dict[str, Dict[str, str]]:
        return self._load()


def _build_metadata_store() -> FileMetadataStore:
    if settings.storage_backend == "local":
        metadata_path = settings.data_dir / METADATA_FILENAME
    elif settings.storage_backend == "s3":
        metadata_path = settings.storage_cache_dir / METADATA_FILENAME
    else:
        metadata_path = Path(METADATA_FILENAME)

    return FileMetadataStore(metadata_path)


file_metadata_store = _build_metadata_store()
