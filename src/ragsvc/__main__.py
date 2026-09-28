"""Command-line entry point: ``python -m ragsvc`` (or ``ragsvc``) serves the API."""

from __future__ import annotations

import logging

import uvicorn

from ragsvc.api import CANDIDATE_PORTS, app, logger
from ragsvc.config import API_HOST, API_PORT
from ragsvc.utils.network import is_port_available


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    port = API_PORT or next((p for p in CANDIDATE_PORTS if is_port_available(p)), CANDIDATE_PORTS[0])
    logger.info("Starting API on %s:%d", API_HOST, port)
    uvicorn.run(app, host=API_HOST, port=port)


if __name__ == "__main__":
    main()
