"""Command-line entry point: ``python -m ragsvc`` (or ``ragsvc``) serves the API."""

from __future__ import annotations

import logging

import uvicorn

from ragsvc.api import CANDIDATE_PORTS, create_app
from ragsvc.config import get_settings
from ragsvc.utils.network import is_port_available

logger = logging.getLogger("ragsvc")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    settings = get_settings()
    port = settings.api_port or next((p for p in CANDIDATE_PORTS if is_port_available(p)), CANDIDATE_PORTS[0])
    logger.info("Starting API on %s:%d", settings.api_host, port)
    uvicorn.run(create_app(settings), host=settings.api_host, port=port)


if __name__ == "__main__":
    main()
