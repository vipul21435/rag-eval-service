# syntax=docker/dockerfile:1
# Two stages: uv resolves the locked dependencies into /opt/venv, and the
# runtime image carries only that virtualenv, run by a non-root user.
# The default image installs no optional extras and configures the hash
# embedder with no reranker, so it needs no model download and builds in
# a couple of minutes. For neural retrieval build with
#   docker build --build-arg UV_SYNC_EXTRAS="--extra neural" -t recallmcp:neural .
# and run it with RAG_EMBEDDING_PROVIDER=sentence-transformers and
# RAG_RERANK_METHOD=cross_encoder (the models are downloaded on first use).
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "uv==0.11.29"

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# Optional extras, for example "--extra documents --extra neural".
ARG UV_SYNC_EXTRAS=""
RUN uv sync --locked --no-dev --no-editable ${UV_SYNC_EXTRAS}

FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin recallmcp \
    && mkdir -p /data \
    && chown recallmcp:recallmcp /data

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    RAG_API_HOST=0.0.0.0 \
    RAG_API_PORT=17995 \
    RAG_EMBEDDING_PROVIDER=hash \
    RAG_RERANK_METHOD=none \
    RAG_EMBEDDING_CACHE_PATH=/data/embeddings.sqlite3

USER recallmcp
WORKDIR /home/recallmcp
VOLUME ["/data"]
EXPOSE 17995

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:17995/health', timeout=3)"

CMD ["recallmcp"]
