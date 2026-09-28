# Changelog

All notable changes to this project are documented here. Versions 2.x
describe the upstream project, Local PDF Chat RAG by Will Wei; the fork
restarts at 0.1.0.

## [Unreleased]

### Added

- `core/ingest.py`: a typed ingestion pipeline (extract, chunk, embed, index)
  that returns a structured per-file report and is shared by every entry point.
- `pyproject.toml` with a committed `uv.lock` on Python 3.12; torch comes from
  the PyTorch CPU index on Linux.

### Changed

- `POST /api/upload` reports chunk counts from the ingestion report instead of
  parsing the demo UI's status text.
- Office-format parsers (DOCX, PPTX, Excel) are an optional `documents` extra.

### Removed

- The Gradio demo UI, its screenshots, the sample Chinese PDF, the generated
  OpenWiki pages and the Chinese README. The REST API is the only interface.

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
