# Atlas

> Enterprise Engineering Knowledge Platform

Atlas is an AI platform that lets engineering teams ingest, search, and reason
over organizational knowledge using Retrieval-Augmented Generation (RAG).

It unifies engineering knowledge — Git repositories and uploaded documents today,
more sources over time — into a single semantic layer where engineers ask natural
language questions and receive accurate, **citation-backed** answers.

The architecture follows the **Open/Closed Principle**: new knowledge sources
and providers plug in behind stable interfaces without changing the core.

---

## What works today

Atlas is a working RAG backend. The full pipeline runs end to end:

**Ingestion → Indexing → Search → Chat**

- **Ingest Git repositories** — clone a repo, parse supported files into a
  canonical `KnowledgeDocument`, chunk, embed, and store.
- **Upload documents** — PDF, DOCX, Markdown, and plain text, through the same
  indexing pipeline.
- **Semantic search** — ranked, metadata-rich results over the vector store.
- **Conversational RAG** — grounded, citation-annotated answers with multi-turn
  memory and optional token streaming.
- **LangGraph workflows** — ingestion, search, and chat are orchestrated as
  explicit graphs (`app/workflows/`).
- **Operational surface** — liveness/readiness health checks, in-process
  metrics, request-id + access-log middleware, structured errors, Docker.

### Deliberately out of scope (see Roadmap)

Auth/RBAC, a PostgreSQL metadata store, and Celery/Redis async ingestion are
**not** implemented yet — ingestion is synchronous and metadata lives in the
vector store. These are the next hardening steps, not part of the current build.

---

## Technology Stack

Each framework has a single, well-defined job. Nothing overlaps — the layers
plug together behind interfaces.

| Layer            | Framework                          | Role in Atlas                                                                                                    |
| ---------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Web API**      | **FastAPI**                        | HTTP routing, request/response validation, dependency injection, auto-generated OpenAPI/Swagger docs.           |
| **ASGI server**  | **Uvicorn**                        | Runs the async application; serves the API and the streaming chat endpoint.                                     |
| **Models & config** | **Pydantic** / **pydantic-settings** | Typed domain models (`KnowledgeDocument`, `SearchResult`, `Citation`) and 12-factor config loaded from `.env`. |
| **Orchestration**| **LangGraph**                      | Expresses ingestion, search, and chat as explicit state-machine graphs — each step is a node with typed state.  |
| **RAG toolkit**  | **LangChain**                      | Provides the `Document` type, the `RecursiveCharacterTextSplitter` (chunking), chat message types, and the LLM/embeddings client integrations. |
| **LLM gateway**  | **OpenRouter** (via `langchain-openai` `ChatOpenAI`) | One API to many models; Atlas adds primary→fallback failover on top.                             |
| **Embeddings**   | **sentence-transformers** (default) / **Google Generative AI** | Turns text into vectors. Default runs `BAAI/bge-large-en-v1.5` (1024-dim) fully on-device — no API key.       |
| **Vector store** | **ChromaDB** (Chroma Cloud)        | Stores chunk vectors + metadata; performs nearest-neighbour similarity search. The system of record for chunks. |
| **Git ingestion**| **GitPython**                      | Clones / pulls repositories to local disk for the Git connector.                                                |
| **File parsing** | **pypdf** / **python-docx**        | Extract plain text from uploaded PDF and DOCX documents.                                                        |
| **Logging**      | **structlog** + stdlib `logging`   | Structured, request-id-tagged logs across the whole request lifecycle.                                          |
| **Tooling**      | **uv** · **black** · **ruff** · **pytest** | Dependency/venv management, formatting, linting, and testing. |
| **Packaging**    | **Docker** / **docker-compose**    | Reproducible container image and one-command local stack.                                                       |

> **How the RAG pieces relate:** LangChain supplies the *building blocks*
> (documents, splitter, LLM/embedding clients); LangGraph *orchestrates* them
> into flows; FastAPI *exposes* those flows over HTTP; Chroma and OpenRouter are
> the *external brains* (memory and reasoning).

---

## Getting Started

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai/keys) API key (for chat)
- A [Chroma Cloud](https://www.trychroma.com/) database (api key, tenant, database)

> Embeddings run **on-device by default** (no key needed). The first run
> downloads the sentence-transformers model (`BAAI/bge-large-en-v1.5`, ~1.3 GB,
> 1024-dim) to the local HuggingFace cache.

### Setup

```bash
uv sync --all-extras --dev      # install dependencies
cp .env.example .env            # then fill in the two required API keys
```

### Run

```bash
uv run uvicorn app.main:app --reload
```

Open the interactive API docs at **http://localhost:8000/docs**.

### Run with Docker

```bash
docker compose up --build       # reads secrets from .env
```

---

## API

Base path: `/api/v1`

| Method | Path                  | Description                                       |
| ------ | --------------------- | ------------------------------------------------- |
| POST   | `/knowledge/ingest`   | Clone a Git repo and index it                     |
| POST   | `/knowledge/upload`   | Upload & index a PDF/DOCX/Markdown/text file      |
| POST   | `/search`             | Semantic search over indexed knowledge            |
| POST   | `/chat`               | Grounded, cited answer (RAG)                      |
| POST   | `/chat/stream`        | Streaming answer (`text/plain`)                   |
| GET    | `/collections`        | List vector-store collections                     |
| GET    | `/settings`           | Non-sensitive runtime configuration               |
| GET    | `/health/`            | Readiness (checks LLM + vector store)             |
| GET    | `/health/live`        | Liveness                                          |
| GET    | `/health/metrics`     | In-process counters/gauges                        |

### Examples

```bash
# Ingest a repository
curl -X POST http://localhost:8000/api/v1/knowledge/ingest \
  -H 'Content-Type: application/json' \
  -d '{"repository_url": "https://github.com/octocat/Spoon-Knife.git"}'

# Upload a document
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -F 'file=@runbook.md' -F 'collection=atlas'

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do I restart the ingestion worker?", "k": 5}'

# Chat (cited answer)
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I restart the ingestion worker?"}'
```

> **Free-tier note:** OpenRouter's `:free` models are rate-limited upstream and
> can return HTTP 429 in bursts. Atlas fails over across the configured models;
> for reliable chat, add credits or your own key
> (https://openrouter.ai/settings/integrations).

---

## Architecture

**Interactive diagram (Excalidraw):**
[open & edit](https://excalidraw.com/#json=EopnTyYpZBo4Ukud9AYm9,sQG8SiURXh-cvVeiH5fwww)
· source: [`docs/architecture.excalidraw`](docs/architecture.excalidraw)

```
Client (Swagger / curl / SDK)
        │
   FastAPI REST API                 middleware: request-id, access log, CORS
        │
   LangGraph Workflows              ingestion · search · chat
        │
┌───────────────┬───────────────┬───────────────┐
│  Connectors   │  AI pipeline  │  Generation   │
│  Git · Files  │  convert →    │  prompt →     │
│               │  split →      │  LLM →        │
│               │  embed        │  citations    │
└───────────────┴───────────────┴───────────────┘
        │                               │
   Chroma Cloud  ◀── embeddings ──▶  OpenRouter
```

Extension points (add without touching the core):

- **New file type** → implement `BaseFileParser`, register in the file factory.
- **New source** → implement `BaseConnector`.
- **New embedding backend** → implement `BaseEmbeddingProvider`, wire in the
  embedding factory (`EMBEDDING_PROVIDER`).
- **New LLM provider** → implement `BaseLLM`.

---

## How It Works — End-to-End Flow

There are two halves: **write-time** (get knowledge in) and **read-time**
(get answers out). Both share the same embedding model and vector store, which
is what makes retrieval work.

### 1. Ingestion & Indexing (write path)

Triggered by `POST /knowledge/ingest` (a Git repo) or `POST /knowledge/upload`
(a file). Orchestrated by the **ingestion LangGraph workflow**
(`app/workflows/ingestion_workflow.py`).

```
Source ─▶ Connector ─▶ Parser ─▶ KnowledgeDocument ─▶ Splitter ─▶ Embedder ─▶ Chroma
```

1. **Acquire** — `GitConnector` (GitPython) clones/pulls the repo, or the upload
   is read into memory.
2. **Parse** — `GitLoader` walks supported files; `DocumentParser` /
   `FileParserFactory` (pypdf, python-docx, text) extract text into a canonical
   **`KnowledgeDocument`** (id, source, path, content, metadata). One model for
   every source.
3. **Convert & chunk** — `DocumentConverter` maps it to a LangChain `Document`;
   `RecursiveSplitter` splits it into overlapping chunks (`CHUNK_SIZE` /
   `CHUNK_OVERLAP`), carrying provenance metadata onto every chunk.
4. **Embed** — `EmbeddingService` + the configured provider
   (`bge-large-en-v1.5` locally) turn each chunk into a 1024-dim vector.
5. **Store** — `IndexingPipeline` upserts `(id, text, vector, metadata)` into
   **Chroma Cloud**. Chunk ids are deterministic, so re-ingesting the same
   content overwrites instead of duplicating (idempotent).

### 2. Search (read path)

Triggered by `POST /search`. Orchestrated by the **search workflow**
(`app/workflows/search_workflow.py`).

1. **Embed the query** — `ChromaRetriever` embeds the query with the *same*
   model used at index time (BGE queries get a retrieval instruction prefix).
2. **Retrieve** — Chroma returns the top-`k` nearest chunks by vector distance.
3. **Shape** — results are mapped to `SearchResult` objects (title, path,
   score, snippet) and returned as ranked JSON.

### 3. Chat / RAG (read path)

Triggered by `POST /chat` (or `/chat/stream`). Orchestrated by the **chat
workflow** (`app/workflows/chat_workflow.py`): `load_history → retrieve →
generate → cite`.

```
Question ─▶ [history] ─▶ Retrieve context ─▶ Build prompt ─▶ OpenRouter LLM ─▶ Answer + Citations
```

1. **Load history** — `ConversationMemory` supplies prior turns when a
   `conversation_id` is given (multi-turn).
2. **Retrieve** — same retrieval as search: the question fetches the most
   relevant chunks from Chroma.
3. **Generate** — `ResponseGenerator` builds a grounded prompt (system rules +
   history + numbered context) and calls the **OpenRouter** LLM through
   `OpenRouterLLM`, which fails over across the configured models. If nothing
   was retrieved, it returns a safe "no context" answer instead of hallucinating.
4. **Cite** — `CitationBuilder` turns the retrieved chunks into numbered
   citations (`[1]`, `[2]`, …) that line up with the sources in the prompt, so
   every answer is traceable. `/chat/stream` streams tokens as they arrive.

Cross-cutting on every request: **middleware** attaches a request id, logs the
call, and applies CORS; **observability** records counters; errors are returned
as structured JSON.

---

## Project Structure

```
app/
├── api/v1/            # HTTP endpoints (health, knowledge, search, chat, ...)
├── core/              # config, DI, logging, exceptions, constants
├── domain/            # knowledge, search, chat services + models
├── ai/                # converters, splitters, embeddings, retrieval, generation
├── infrastructure/    # connectors (git, files), llm, vectorstore
├── workflows/         # LangGraph graphs: ingestion, search, chat
├── middleware/        # request-id, access log, exception handler
├── observability/     # metrics, tracing
└── main.py
tests/                 # unit tests (pytest)
```

---

## Testing

```bash
uv run pytest            # unit tests (offline, no API keys required)
uv run black --check .   # formatting
uv run ruff check .      # lint
```

Unit tests mock all external services, so they run without network or keys. CI
runs format + lint + tests on every push.

---

## Roadmap

- [x] Git repository ingestion & indexing
- [x] Document upload (PDF, DOCX, Markdown, text)
- [x] Chroma Cloud vector store
- [x] LangChain pipeline & LangGraph workflows
- [x] Semantic search
- [x] Conversational RAG with citations & streaming
- [x] Health checks, metrics, structured logging, Docker
- [ ] Authentication (JWT) & RBAC
- [ ] PostgreSQL metadata store (SQLAlchemy + Alembic)
- [ ] Background ingestion (Celery + Redis)
- [ ] Reranking / hybrid retrieval
- [ ] Additional connectors (GitHub, Confluence, Jira, Slack, Notion, …)

---

## License

MIT License
