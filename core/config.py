import os
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
        return
    except ModuleNotFoundError:
        pass

    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


_load_env_file()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    db_location: Path = BASE_DIR / "chrome_langchain_db"
    collection_name: str = "excel_files"
    supported_excel_extensions: FrozenSet[str] = frozenset({".xlsx", ".xls", ".xlsm"})

    local_llm_hosted: bool = _env_bool("LOCAL_LLM_HOSTED", True)
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "gemma4")
    hosted_llm_provider: str = os.getenv("HOSTED_LLM_PROVIDER", "openai")
    hosted_llm_model: str = os.getenv("HOSTED_LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    retriever_k: int = _env_int("RETRIEVER_K", 5)
    max_chat_history_messages: int = _env_int("MAX_CHAT_HISTORY_MESSAGES", 12)
    max_chat_history_content_chars: int = _env_int(
        "MAX_CHAT_HISTORY_CONTENT_CHARS",
        2000,
    )


settings = Settings()
