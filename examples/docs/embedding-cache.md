# The embedding cache

Embedding is the slow part of ingestion. A neural embedding model has to
run once per chunk, and the same chunks come back every time an unchanged
document is indexed again. RecallMCP therefore keeps computed vectors in an
SQLite file, one row per vector, and looks the cache up before calling the
embedding provider.

The cache key is the triple of provider name, model name and the SHA-256
hash of the chunk text. Keying by provider and model as well as by text
means that a vector computed by all-MiniLM-L6-v2 is never served for a
different model: switching RAG_EMBED_MODEL_NAME simply starts a new set of
rows. The text hash avoids storing the chunk itself, and the vector is
stored as raw float32 bytes together with its dimension, so a corrupt row
whose payload does not match its recorded dimension is treated as a miss
and overwritten on the next write.

The cache counts hits and misses since the process started. Every position
in a lookup counts once, including repeated texts, so the counters describe
how much embedding work the cache saved. The counters and the number of
rows on disk are reported by the GET /health endpoint under the
embedding_cache key, which makes it easy to check from outside the process
whether re-indexing is doing any work.

The cache is on by default and lives at RAG_EMBEDDING_CACHE_PATH, which
defaults to .cache/recallmcp/embeddings.sqlite3 under the working
directory. Set RAG_EMBEDDING_CACHE_ENABLED to false to compute every vector
fresh; the test suite runs with the cache off so no test writes outside its
temporary directory. The store uses only the Python standard library.
