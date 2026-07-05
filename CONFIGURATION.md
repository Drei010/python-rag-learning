# Configuration

All settings are loaded from a `.env` file in the project root. Copy `.env.example` to get started:

```bash
cp .env.example .env
```

---

## File Storage

Controls where uploaded documents are stored.

| Variable | Default | Description |
| --- | --- | --- |
| `FILE_STORAGE_BACKEND` | `local` | `local` for the `data/` folder, `s3` for AWS S3 |
| `DATA_DIR` | `data` | Local directory for uploaded files |
| `AWS_S3_BUCKET` | — | S3 bucket name (required when backend is `s3`) |
| `AWS_S3_PREFIX` | — | Optional key prefix inside the bucket |
| `AWS_REGION` | — | AWS region for S3 operations |
| `AWS_S3_ENDPOINT_URL` | — | Custom S3-compatible endpoint (e.g. MinIO) |
| `AWS_ACCESS_KEY_ID` | — | AWS access key (or use instance profile) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_SESSION_TOKEN` | — | Optional session token for temporary credentials |
| `STORAGE_CACHE_DIR` | `.storage_cache` | Local cache directory when backend is `s3` |

> When using S3, the app syncs objects to a local cache for parsing. The cache is populated on startup and after uploads.

---

## LLM

Controls the language model used for answering questions.

Set `LLM_MODE=local` to use Ollama, or `LLM_MODE=hosted` for an OpenAI/Groq-compatible API.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODE` | `local` | `local` (Ollama) or `hosted` (OpenAI/Groq) |
| `OLLAMA_LLM_MODEL` | `gemma4` | Ollama model name for local inference |
| `HOSTED_LLM_PROVIDER` | `openai` | `openai` or `groq` |
| `HOSTED_LLM_MODEL` | `gpt-4o-mini` | Model name for the hosted provider |
| `HOSTED_LLM_API_KEY` | — | API key for the hosted provider |
| `HOSTED_LLM_BASE_URL` | — | Custom base URL for OpenAI-compatible APIs |
| `HOSTED_LLM_TEMPERATURE` | `1` | Sampling temperature (0–2) |
| `HOSTED_LLM_TOP_P` | `1` | Top-p (nucleus) sampling |
| `HOSTED_LLM_MAX_TOKENS` | `2048` | Maximum tokens in the response |
| `GROQ_REASONING_EFFORT` | `medium` | Reasoning effort level for Groq models |

### Fallback API keys

These are checked when provider-specific keys are not set:

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI LLM, embeddings, and chunking |
| `GROQ_API_KEY` | Groq LLM |

> **Legacy:** `LOCAL_LLM_HOSTED=true` is still recognized for backward compatibility but ignored when `LLM_MODE` is set.

---

## Embeddings

Controls how document text is converted to vectors. This is independent of `LLM_MODE`.

Set `EMBEDDING_MODE=local` for Ollama, `EMBEDDING_MODE=custom` for built-in hash embeddings, or `EMBEDDING_MODE=hosted` for a hosted provider.

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODE` | `local` | `local`, `custom`/`hash`, or `hosted` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LOCAL_EMBEDDING_DIMENSIONS` | `384` | Vector size for hash embeddings |
| `HOSTED_EMBEDDING_PROVIDER` | `openai` | `openai` or `nvidia` |
| `HOSTED_EMBEDDING_MODEL` | `text-embedding-3-small` | Model name for hosted embeddings |
| `HOSTED_EMBEDDING_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `HOSTED_EMBEDDING_BASE_URL` | — | Custom endpoint for the embedding API |

### Embedding modes explained

| Mode | Use case |
| --- | --- |
| `local` | Best quality for local dev — requires Ollama running |
| `custom`/`hash` | Deterministic, no external calls — useful for testing |
| `hosted` | Production-grade embeddings via OpenAI or NVIDIA |

---

## Chunking

Controls how parsed documents are split into smaller pieces before embedding.

If `CHUNKING_MODE` is not set, it inherits from `LLM_MODE`. When set to `hosted`, the app uses an LLM to perform semantic chunking (splitting by meaning). When `local`, documents pass through a rule-based splitter.

| Variable | Default | Description |
| --- | --- | --- |
| `CHUNKING_MODE` | value of `LLM_MODE` | `local` or `hosted` |
| `OLLAMA_CHUNKING_MODEL` | value of `OLLAMA_LLM_MODEL` | Ollama model for local chunking |
| `HOSTED_CHUNKING_PROVIDER` | `openai` | `openai` or `groq` |
| `HOSTED_CHUNKING_MODEL` | `gpt-4o-mini` | Hosted model for semantic chunking |
| `HOSTED_CHUNKING_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `HOSTED_CHUNKING_BASE_URL` | — | Custom base URL |
| `HOSTED_CHUNKING_TEMPERATURE` | `1` | Sampling temperature |
| `HOSTED_CHUNKING_TOP_P` | `1` | Top-p sampling |
| `HOSTED_CHUNKING_MAX_TOKENS` | `2048` | Max tokens for chunking responses |

> **Fallback:** If the hosted chunking LLM is unreachable, a rule-based `RecursiveCharacterTextSplitter` is used automatically.

> **Legacy names:** `HOSTED_LLM_CHUNKING_PROVIDER`, `HOSTED_LLM_CHUNKING_MODEL`, etc. are still recognized as fallbacks.

---

## Retrieval & Chat History

Tuning parameters for the retrieval pipeline and chat session memory.

| Variable | Default | Description |
| --- | --- | --- |
| `RETRIEVER_K` | `5` | Total documents retrieved per query |
| `RETRIEVER_K_PER_SOURCE` | `3` | Documents retrieved per source file |
| `MAX_CHAT_HISTORY_MESSAGES` | `12` | Messages retained per chat session |
| `MAX_CHAT_HISTORY_CONTENT_CHARS` | `2000` | Max characters kept per history message |

---

## Vector Store

Controls where embeddings are persisted and searched.

Set `VECTOR_DB_BACKEND=chroma` (default) for ChromaDB, or `VECTOR_DB_BACKEND=hana` for SAP HANA Cloud Vector Engine.

### ChromaDB

| Variable | Default | Description |
| --- | --- | --- |
| `VECTOR_DB_BACKEND` | `chroma` | `chroma` or `hana` |
| `CHROMA_MODE` | `local` | `local` (persist to disk) or `http` (client-server) |
| `CHROMA_HOST` | `localhost` | Chroma server hostname (`http` mode only) |
| `CHROMA_PORT` | `8000` | Chroma server port (`http` mode only) |
| `CHROMA_SSL` | `false` | Use HTTPS for the Chroma connection |
| `CHROMA_HEADERS` | — | JSON string of auth headers, e.g. `{"Authorization": "Bearer token"}` |

### SAP HANA Cloud

| Variable | Default | Description |
| --- | --- | --- |
| `HANA_DB_ADDRESS` | — | HANA host (required when backend is `hana`) |
| `HANA_DB_PORT` | `443` | HANA port |
| `HANA_DB_USER` | — | Database username |
| `HANA_DB_PASSWORD` | — | Database password |
| `HANA_DB_TABLE_NAME` | `LANGCHAIN_VECTORS` | Table for vector storage |

> SAP HANA requires additional dependencies: `pip install hdbcli langchain-hana`

---

## Quick Reference

Minimal `.env` for local development with Ollama:

```env
LLM_MODE=local
OLLAMA_LLM_MODEL=gemma4
EMBEDDING_MODE=local
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Minimal `.env` for hosted (OpenAI):

```env
LLM_MODE=hosted
HOSTED_LLM_PROVIDER=openai
HOSTED_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
EMBEDDING_MODE=hosted
HOSTED_EMBEDDING_PROVIDER=openai
```

Minimal `.env` for hosted (Groq + NVIDIA embeddings):

```env
LLM_MODE=hosted
HOSTED_LLM_PROVIDER=groq
HOSTED_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
EMBEDDING_MODE=hosted
HOSTED_EMBEDDING_PROVIDER=nvidia
HOSTED_EMBEDDING_MODEL=NV-Embed-QA
HOSTED_EMBEDDING_API_KEY=nvapi-...
```
