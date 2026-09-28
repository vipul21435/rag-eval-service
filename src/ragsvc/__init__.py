"""RecallMCP: a local-first RAG service with local embeddings and hybrid retrieval.

The package grew out of Local PDF Chat RAG by Will Wei (MIT license). The
modules under ``ragsvc.core`` and ``ragsvc.features`` are reworked versions
of that project's ``core/`` and ``features/`` packages; see README.md for
what changed and LICENSE for the terms.

Importing ``ragsvc`` is cheap: model libraries are loaded lazily by the
modules that need them.
"""

import os

__version__ = "0.1.0"

# faiss-cpu and torch each bundle their own copy of the LLVM OpenMP runtime
# on macOS. The second one to load (torch, when the sentence-transformers
# embedder first runs) aborts the process with "OMP: Error #15" unless the
# runtime is told to tolerate the duplicate. This runs before either library
# is imported anywhere in the package; an explicit value in the environment
# wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__all__ = ["__version__"]
