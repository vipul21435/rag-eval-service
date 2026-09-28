# Operating the service

RecallMCP exposes two probe endpoints outside the /api prefix so that
liveness and readiness checks never need the API token. GET /health is the
liveness signal. It returns the version, the names of the LLM, embedding
and reranker providers, the size and type of the FAISS index and the
embedding cache counters. It is cheap by construction: it names the
embedding model without loading it and reads the cache counters without
embedding anything. GET /ready means ready to answer questions, so it
returns 503 with a reason until at least one document is indexed and 200
with the chunk count afterwards.

Every response carries an X-Request-ID header. When the client sends a
well formed id of printable ASCII up to 128 characters the server echoes
it; otherwise it generates a UUID. The id is bound to the request while it
is served, so every log record written during the request carries it, and
one access record per request is written to the ragsvc.access logger with
the method, path, status code, duration in milliseconds and client
address. With RAG_LOG_FORMAT set to json each record is one JSON object per
line, which log collectors ingest without parsing rules.

The defaults keep the API private to the machine it runs on. The server
binds the loopback interface, sends no CORS headers and caps uploads at
RAG_MAX_UPLOAD_MB megabytes. To reach it from another host set
RAG_API_HOST to 0.0.0.0 and set RAG_API_TOKEN, after which every /api
request must carry an Authorization header with a bearer token. The server
logs a warning when it is exposed without a token.

The container image runs as a non-root user, defaults to the hash embedder
so it needs no model download, and stores its embedding cache under /data,
which docker compose mounts as a named volume. The compose healthcheck
polls GET /health every thirty seconds.
