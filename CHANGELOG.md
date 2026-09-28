# Changelog

All notable changes to this project are documented here. Versions 2.x
describe the upstream project, Local PDF Chat RAG by Will Wei; the fork
restarts at 0.1.0.

## [Unreleased]

### Added

- `ragsvc.embeddings`: pluggable embedding providers behind an
  `EmbeddingProvider` protocol. `SentenceTransformerEmbedder` (the default)
  imports and loads its model on first use; `HashEmbedder` is deterministic
  feature hashing of word unigrams and bigrams (no model, no download) used
  by the tests and the demo; `EmbeddingCache` keeps vectors in SQLite keyed
  by provider, model and text hash, with hit and miss counters. Selected
  with `RAG_EMBEDDING_PROVIDER`, `RAG_HASH_EMBEDDING_DIMENSION`,
  `RAG_EMBEDDING_CACHE_ENABLED` and `RAG_EMBEDDING_CACHE_PATH`.
- `GET /health` (status, version, LLM, embedding and reranker provider names,
  index size, embedding cache counters; never loads a model) and
  `GET /ready` (`200` once documents are indexed, `503` before), both outside
  `/api` so they need no token.
- Request ids and structured logs: every response carries an `X-Request-ID`
  (the client's well-formed one, or a generated UUID), log records written
  while serving a request carry its id, and one access record per request
  (method, path, status, duration) goes to the `ragsvc.access` logger.
  `RAG_LOG_FORMAT=json` (default) writes one JSON object per line;
  `RAG_LOG_LEVEL` sets the root level. Uvicorn's own access log is off.
- `core/ingest.py`: a typed ingestion pipeline (extract, chunk, embed, index)
  that returns a structured per-file report and is shared by every entry point.
- `pyproject.toml` with a committed `uv.lock` on Python 3.12; torch comes from
  the PyTorch CPU index on Linux.
- Ruff (lint + format) and strict mypy configuration in `pyproject.toml`;
  CI runs lint, formatting check, type check and tests.
- `.pre-commit-config.yaml` running the same ruff and mypy checks plus
  whitespace, YAML/TOML, large-file and private-key hooks.

### Changed

- The project is RecallMCP: `recallmcp` on GitHub and as the CLI name; the
  Python package is `ragsvc`.
- Configuration is a typed, validated `Settings` object (pydantic-settings)
  read from `RAG_`-prefixed environment variables and a `.env` file in the
  working directory, instead of module-level constants read at import time.
  Out-of-range values and unknown provider names fail at startup with the
  variable named; secrets are `SecretStr` values that never appear in
  `repr`. `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `SERPAPI_KEY` are still
  accepted under their conventional names. `create_app(settings)` builds the
  API from an explicit `Settings` instance and there is no import-time
  application object.
- The service is an importable package, `ragsvc` under `src/`, instead of
  top-level `core/`, `features/`, `utils/`, `config.py` and `api_router.py`
  modules. `uv run recallmcp` (or `python -m ragsvc`) serves the API.
- Local-first configuration: providers are `ollama` (default, auto-detected)
  and `openai` (any OpenAI-compatible endpoint via `OPENAI_*`). The
  SiliconFlow and Magick provider settings, the `hf-mirror.com` Hugging Face
  endpoint override and the global `requests` retry patch are gone. Embedding
  and reranker models plus retrieval hyperparameters are configurable through
  environment variables, and provider detection no longer runs at import time.
- `POST /api/ask` takes `provider` instead of `model_choice`, returns
  structured sources from retrieval metadata rather than regex-parsing the
  answer, and maps an empty knowledge base to `409` and provider failures to
  `502` instead of returning error text as the answer.
- `POST /api/upload` reports chunk counts from the ingestion report instead of
  parsing the demo UI's status text.
- `POST /api/ask` returns the answer as plain text and the `<think>` output of
  reasoning models as a separate `reasoning` field. Upstream rendered the
  reasoning as an HTML `<details>` block for the Gradio UI and HTML-escaped
  the rest of the answer, which corrupted text such as `x < 10` in the API
  and let model output starting with `<details` or `<summary` (attributes
  included) through unescaped.
- `example.env` is now `.env.example`; `config.py` no longer falls back to it.
- BM25 tokenizes with a lowercase regex (CJK ideographs as character
  unigrams) instead of jieba, dropping a 19 MB dictionary dependency and its
  import-time cache build.
- Comments, docstrings, log messages and LLM prompts in `core/`, `features/`
  and `utils/` are in English; the query-rewriting prompt now uses the
  `NO_FURTHER_QUERY` sentinel and web search defaults to English results.
- Conflict detection compares the set of numeric facts per source rather than
  their order of mention, and no longer special-cases upstream sample text.

### Fixed

- The unauthenticated API bound `0.0.0.0` and answered every CORS preflight
  with the caller's origin reflected and `allow-credentials: true`, so any web
  page the operator visited could read answers to private documents, replace
  the knowledge base and spend the configured API quotas. The server now binds
  `127.0.0.1` (`API_HOST`), sends CORS headers only for origins listed in
  `CORS_ALLOW_ORIGINS` and never allows credentials. `/api/upload` streams
  the body to disk and rejects files over `MAX_UPLOAD_MB` with `413` instead
  of reading an unbounded body into memory, and an optional `API_TOKEN`
  requires `Authorization: Bearer <token>` on every `/api` request.
- Query rewriting kept only the text before the first `<think>`, so reasoning
  models served by Ollama or an OpenAI-compatible server (which emit the
  reasoning first) produced an empty rewrite: the `NO_FURTHER_QUERY` sentinel
  was lost and every remaining retrieval round searched for `""`. Reasoning
  blocks are now removed wherever they appear, and an empty rewrite ends the
  loop.
- A failed upload wiped the knowledge base: ingestion cleared both indexes
  before knowing whether any file would parse, so an unsupported extension,
  a non-UTF-8 text file, a missing optional parser or an embedding error left
  the service empty while `/api/upload` answered `200` with `status: error`.
  The new indexes are now built first and swapped in only when at least one
  chunk was produced. `/api/upload` rejects unsupported extensions with `415`
  before reading the body, and the loader reports an unsupported format, a
  missing `documents` extra and a non-UTF-8 text file with distinct messages
  instead of the misleading "document is empty".
- Two uploads running at the same time corrupted the dense index:
  `VectorStore.build_index` appended to the live id list while replacing the
  FAISS index, so positions mapped to the wrong chunks, and nothing stopped
  both runs from clearing and rebuilding over each other. Ingestion runs are
  now serialized by a lock, and both the FAISS and BM25 stores build a fresh
  snapshot and install it with a single assignment, so a concurrent
  `/api/ask` sees either the old knowledge base or the new one.
- The default reranker model was a bi-encoder
  (`distiluse-base-multilingual-cased-v2`) loaded as a cross-encoder, which
  scores with an untrained head; the default is now
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `split_text(chunk_overlap=0)` used the configured default instead of zero.
- Office-format parsers (DOCX, PPTX, Excel) are an optional `documents` extra.

### Removed

- The Gradio demo UI, its screenshots, the sample Chinese PDF, the generated
  OpenWiki pages and the Chinese README. The REST API is the only interface.
- `stream_answer` (Gradio-only streaming) and the unused source-credibility
  scorer.

## [2.1.0] - 2026-08-12

### Added

- MIT license recognized by GitHub.
- English README and a concise Chinese project guide.
- Automated tests for configuration, document loading, hybrid retrieval, and missing-key behavior.
- GitHub Actions CI for source compilation and tests.
- Contribution, security, conduct, issue, and pull request guidance.
- A current application screenshot and centralized version metadata.

### Changed

- Repositioned the repository as a transparent educational and reference RAG implementation.
- Clarified supported document types, setup steps, provider choices, and known limitations.
- Added the runtime dependencies required for Excel parsing.
- Updated Gradio support to the 6.x line used by the current interface.

### Removed

- Commercial book, course, community, and store promotion from the repository.

## [2.0.0] - 2026-03-18

### Added

- Modular `core/` and `features/` structure.
- Gradio 6.x compatibility updates.
- Configurable model names and provider selection.
- FAISS and BM25 hybrid retrieval pipeline.
