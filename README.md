# Python RAG Learning

A FastAPI application that lets you upload documents, index them into a vector store, and ask questions about their contents using retrieval-augmented generation (RAG).

Supported file types:

- Excel (`.xlsx`, `.xls`, `.xlsm`)
- PDF (`.pdf`)
- PowerPoint (`.pptx`, `.pptm`)

## How it works

1. Files are stored in a local `data/` folder or in AWS S3, depending on configuration.
2. On startup (and after uploads or deletes), the app parses each file into searchable records.
3. Records are embedded and stored in a Chroma vector database.
4. When you ask a question, relevant context is retrieved and sent to an LLM to produce an answer.

Chat sessions are tracked so follow-up questions can use prior conversation history.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (default setup uses local models)
- Optional: AWS credentials if using S3 storage
- Optional: OpenAI or Groq API keys if using hosted LLMs instead of Ollama

If you use the default local LLM setup, pull the required Ollama models first:

```bash
ollama pull gemma4
ollama pull nomic-embed-text
```

## Installation

1. Clone the repository and enter the project directory.

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root. See [Configuration](#configuration) below for available variables.

5. Place source files in the `data/` folder, or configure S3 storage and upload files through the API.

## Configuration

Environment variables are loaded from `.env`. Common settings:

### File storage

| Variable | Default | Description |
| --- | --- | --- |
| `FILE_STORAGE_BACKEND` | `local` | `local` for the `data/` folder, or `s3` for AWS |
| `DATA_DIR` | `data` | Local folder for uploaded files |
| `AWS_S3_BUCKET` | — | S3 bucket name (required when backend is `s3`) |
| `AWS_S3_PREFIX` | — | Optional key prefix inside the bucket |
| `AWS_REGION` | — | AWS region for S3 |
| `AWS_S3_ENDPOINT_URL` | — | Optional custom S3 endpoint |
| `STORAGE_CACHE_DIR` | `.storage_cache` | Local cache used when backend is `s3` |

### LLM

| Variable | Default | Description |
| --- | --- | --- |
| `LOCAL_LLM_HOSTED` | `true` | Use Ollama when `true`, hosted provider when `false` |
| `OLLAMA_LLM_MODEL` | `gemma4` | Ollama chat model |
| `HOSTED_LLM_PROVIDER` | `openai` | `openai` or `groq` when not using Ollama |
| `HOSTED_LLM_MODEL` | `gpt-4o-mini` | Hosted chat model name |
| `HOSTED_LLM_API_KEY` | — | API key for the hosted LLM provider |

### Embeddings

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `HOSTED_EMBEDDING_PROVIDER` | `local` | `local`, `openai`, or other supported provider |
| `HOSTED_EMBEDDING_MODEL` | `text-embedding-3-small` | Hosted embedding model name |
| `HOSTED_EMBEDDING_API_KEY` | — | API key for hosted embeddings |

### Retrieval

| Variable | Default | Description |
| --- | --- | --- |
| `RETRIEVER_K` | `5` | Number of documents retrieved per query |
| `RETRIEVER_K_PER_SOURCE` | `3` | Documents retrieved per source file |
| `MAX_CHAT_HISTORY_MESSAGES` | `12` | Messages kept per chat session |

## Starting the application

### API server (recommended)

```bash
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

Interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

On first startup, the app indexes all files in storage. Large datasets may take a minute or two.

### CLI mode

You can also run a simple terminal chat loop:

```bash
python3 main.py
```

Type `q` to quit.

## API endpoints

All routes are prefixed with `/ai`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/ai/chat` | Ask a question about indexed documents |
| `POST` | `/ai/upload` | Upload a supported file and reindex |
| `GET` | `/ai/files` | List files in the active storage destination |
| `DELETE` | `/ai/files/{filename}` | Delete a file and reindex |
| `POST` | `/ai/reindex` | Rebuild the vector store from stored files |

### Examples

**List stored files**

```bash
curl http://127.0.0.1:8000/ai/files
```

Example response:

```json
{
  "destination": "data",
  "location": "/path/to/project/data",
  "files": ["example.xlsx", "report.pdf"]
}
```

When `FILE_STORAGE_BACKEND=s3`, `destination` is `"aws"` and `location` is the S3 URI.

**Ask a question**

```bash
curl -X POST http://127.0.0.1:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What internships are available?"}'
```

**Upload a file**

```bash
curl -X POST http://127.0.0.1:8000/ai/upload \
  -F "file=@/path/to/document.xlsx"
```

**Delete a file**

```bash
curl -X DELETE http://127.0.0.1:8000/ai/files/document.xlsx
```

**Reindex all files**

```bash
curl -X POST http://127.0.0.1:8000/ai/reindex
```

## Project structure

```
python-rag-learning/
├── api/routes/          # FastAPI route handlers
│   ├── chat.py          # Question-answering endpoint
│   ├── delete.py        # File deletion endpoint
│   ├── files.py         # File listing endpoint
│   └── upload.py        # Upload and reindex endpoints
├── core/
│   └── config.py        # Environment-based settings
├── models/
│   └── schemas.py       # Request/response models
├── services/
│   ├── embed_service.py # Vector store and retrieval
│   ├── file_service.py  # File storage helpers
│   ├── rag_service.py   # LLM chain and chat sessions
│   ├── storage_service.py # Local and S3 storage backends
│   ├── excel_service.py # Excel parsing
│   ├── pdf_service.py   # PDF parsing
│   └── ppt_service.py   # PowerPoint parsing
├── data/                # Local file storage (default)
├── aws/                 # Example AWS IAM and bucket policies
├── main.py              # FastAPI app entry point
└── requirements.txt
```

## Storage backends

### Local (`data`)

Set `FILE_STORAGE_BACKEND=local`. Files are read from and written to the `data/` directory.

### AWS S3

Set `FILE_STORAGE_BACKEND=s3` and provide `AWS_S3_BUCKET` (and optionally `AWS_S3_PREFIX`). The app lists and syncs objects from S3, caches them locally, and uses the cache for indexing.

Example policy templates are in the `aws/` folder.
