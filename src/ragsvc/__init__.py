"""RecallMCP: a local-first RAG service with local embeddings and hybrid retrieval.

The package grew out of Local PDF Chat RAG by Will Wei (MIT license). The
modules under ``ragsvc.core`` and ``ragsvc.features`` are reworked versions
of that project's ``core/`` and ``features/`` packages; see README.md for
what changed and LICENSE for the terms.

Importing ``ragsvc`` is cheap: model libraries are loaded lazily by the
modules that need them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
