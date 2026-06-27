import os
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        # Keep local development predictable: values in .env should win.
        load_dotenv(BASE_DIR / ".env", override=True)
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
        os.environ[key.strip()] = value.strip().strip("\"'")


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


def _first_env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue

        try:
            return int(value)
        except ValueError:
            return default

    return default


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    return BASE_DIR / path


def _first_env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    return default


def _env_local_or_hosted_mode(name: str, default: str) -> str:
    value = os.getenv(name, default)
    normalized = value.strip().lower()
    if normalized in {"local", "ollama"}:
        return "local"
    if normalized in {"custom", "hash"}:
        return "custom"
    if normalized in {"hosted", "remote"}:
        return "hosted"

    raise ValueError(f"{name} must be either 'local', 'custom', or 'hosted'")


def _env_llm_mode() -> str:
    value = os.getenv("LLM_MODE")
    if value:
        return _env_local_or_hosted_mode("LLM_MODE", value)

    # Backward compatibility for existing .env files. Historically this meant
    # "use the local Ollama model" when true, despite the confusing name.
    return "local" if _env_bool("LOCAL_LLM_HOSTED", True) else "hosted"


def _env_chunking_mode(llm_mode: str) -> str:
    return _env_local_or_hosted_mode("CHUNKING_MODE", llm_mode)


def _env_embedding_mode() -> str:
    value = os.getenv("EMBEDDING_MODE")
    if value:
        return _env_local_or_hosted_mode("EMBEDDING_MODE", value)

    provider = os.getenv("HOSTED_EMBEDDING_PROVIDER", "").strip().lower()
    if provider and provider not in {"local", "hash"}:
        return "hosted"

    return "local"


def _env_hosted_embedding_provider() -> str:
    provider = os.getenv("HOSTED_EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider in {"", "hosted"}:
        return "openai"

    return provider


_load_env_file()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = _env_path("DATA_DIR", BASE_DIR / "data")
    storage_backend: str = os.getenv("FILE_STORAGE_BACKEND", "local").strip().lower()
    storage_cache_dir: Path = _env_path(
        "STORAGE_CACHE_DIR",
        BASE_DIR / ".storage_cache",
    )
    aws_s3_bucket: str = os.getenv("AWS_S3_BUCKET", os.getenv("S3_BUCKET_NAME", ""))
    aws_s3_prefix: str = os.getenv("AWS_S3_PREFIX", os.getenv("S3_PREFIX", "")).strip("/")
    aws_s3_region_name: str = _first_env_value(
        "AWS_REGION_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    )
    aws_s3_endpoint_url: str = os.getenv("AWS_S3_ENDPOINT_URL", "")
    db_location: Path = BASE_DIR / "chrome_langchain_db"
    collection_name: str = "uploaded_files"
    supported_excel_extensions: FrozenSet[str] = frozenset({".xlsx", ".xls", ".xlsm"})
    supported_pdf_extensions: FrozenSet[str] = frozenset({".pdf"})
    supported_powerpoint_extensions: FrozenSet[str] = frozenset({".pptx", ".pptm"})

    llm_mode: str = _env_llm_mode()
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "gemma4")
    hosted_llm_provider: str = (
        os.getenv("HOSTED_LLM_PROVIDER", "openai").strip().lower()
    )
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

    chunking_mode: str = _env_chunking_mode(llm_mode)
    ollama_chunking_model: str = os.getenv(
        "OLLAMA_CHUNKING_MODEL",
        os.getenv("OLLAMA_LLM_MODEL", "gemma4"),
    )
    hosted_chunking_provider: str = _first_env_value(
        "HOSTED_CHUNKING_PROVIDER",
        "HOSTED_LLM_CHUNKING_PROVIDER",
        default="openai",
    ).strip().lower()
    hosted_chunking_model: str = _first_env_value(
        "HOSTED_CHUNKING_MODEL",
        "HOSTED_LLM_CHUNKING_MODEL",
        default="gpt-4o-mini",
    )
    hosted_chunking_api_key: str = _first_env_value(
        "HOSTED_CHUNKING_API_KEY",
        "HOSTED_LLM_CHUNKING_API_KEY",
        "OPENAI_API_KEY",
    )
    hosted_chunking_base_url: str = _first_env_value(
        "HOSTED_CHUNKING_BASE_URL",
        "HOSTED_LLM_CHUNKING_BASE_URL",
    )
    hosted_chunking_temperature: float = float(
        _first_env_value(
            "HOSTED_CHUNKING_TEMPERATURE",
            "HOSTED_LLM_CHUNKING_TEMPERATURE",
            default="1",
        )
    )
    hosted_chunking_top_p: float = float(
        _first_env_value(
            "HOSTED_CHUNKING_TOP_P",
            "HOSTED_LLM_CHUNKING_TOP_P",
            default="1",
        )
    )
    hosted_chunking_max_tokens: int = _first_env_int(
        "HOSTED_CHUNKING_MAX_TOKENS",
        "HOSTED_LLM_CHUNKING_MAX_TOKENS",
        default=2048,
    )

    embedding_mode: str = _env_embedding_mode()
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )
    hosted_embedding_provider: str = _env_hosted_embedding_provider()
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
    def use_local_llm(self) -> bool:
        return self.llm_mode == "local"

    @property
    def local_llm_hosted(self) -> bool:
        return self.use_local_llm

    @property
    def use_local_chunking(self) -> bool:
        return self.chunking_mode == "local"

    @property
    def use_local_embeddings(self) -> bool:
        return self.embedding_mode == "local"

    @property
    def use_custom_hash_embeddings(self) -> bool:
        return self.embedding_mode == "custom"

    @property
    def supported_file_extensions(self) -> FrozenSet[str]:
        return (
            self.supported_excel_extensions
            | self.supported_pdf_extensions
            | self.supported_powerpoint_extensions
        )


settings = Settings()
