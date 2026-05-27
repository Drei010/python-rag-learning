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


def _first_env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    return default


_load_env_file()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    db_location: Path = BASE_DIR / "chrome_langchain_db"
    collection_name: str = "uploaded_files"
    supported_excel_extensions: FrozenSet[str] = frozenset({".xlsx", ".xls", ".xlsm"})
    supported_pdf_extensions: FrozenSet[str] = frozenset({".pdf"})
    supported_powerpoint_extensions: FrozenSet[str] = frozenset({".pptx", ".pptm"})

    local_llm_hosted: bool = _env_bool("LOCAL_LLM_HOSTED", True)
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "gemma4")
    hosted_llm_provider: str = os.getenv("HOSTED_LLM_PROVIDER", "openai")
    hosted_llm_model: str = os.getenv("HOSTED_LLM_MODEL", "gpt-4o-mini")
    hosted_llm_api_key: str = _first_env_value(
        "HOSTED_LLM_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    )
    hosted_llm_base_url: str = os.getenv("HOSTED_LLM_BASE_URL", "")
    hosted_llm_temperature: float = float(os.getenv("HOSTED_LLM_TEMPERATURE", "1"))
    hosted_llm_top_p: float = float(os.getenv("HOSTED_LLM_TOP_P", "1"))
    hosted_llm_max_tokens: int = _env_int("HOSTED_LLM_MAX_TOKENS", 2048)
    groq_reasoning_effort: str = os.getenv("GROQ_REASONING_EFFORT", "medium")

    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )
    hosted_embedding_provider: str = os.getenv("HOSTED_EMBEDDING_PROVIDER", "local")
    hosted_embedding_model: str = os.getenv(
        "HOSTED_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    hosted_embedding_api_key: str = _first_env_value(
        "HOSTED_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    )
    hosted_embedding_base_url: str = os.getenv("HOSTED_EMBEDDING_BASE_URL", "")
    local_embedding_dimensions: int = _env_int("LOCAL_EMBEDDING_DIMENSIONS", 384)

    retriever_k: int = _env_int("RETRIEVER_K", 5)
    retriever_k_per_source: int = _env_int("RETRIEVER_K_PER_SOURCE", 3)
    max_chat_history_messages: int = _env_int("MAX_CHAT_HISTORY_MESSAGES", 12)
    max_chat_history_content_chars: int = _env_int(
        "MAX_CHAT_HISTORY_CONTENT_CHARS",
        2000,
    )

    @property
    def supported_file_extensions(self) -> FrozenSet[str]:
        return (
            self.supported_excel_extensions
            | self.supported_pdf_extensions
            | self.supported_powerpoint_extensions
        )


settings = Settings()
