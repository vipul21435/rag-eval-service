"""Command-line entry point: ``python -m ragsvc`` (or ``recallmcp``) serves the API."""

from __future__ import annotations

import logging

import uvicorn

from ragsvc.api import CANDIDATE_PORTS, create_app
from ragsvc.config import get_settings
from ragsvc.logging_setup import configure_logging
from ragsvc.utils.network import is_port_available

logger = logging.getLogger("ragsvc")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    port = settings.api_port or next((p for p in CANDIDATE_PORTS if is_port_available(p)), CANDIDATE_PORTS[0])
    logger.info("Starting API on %s:%d", settings.api_host, port)
    # uvicorn's own logging config and access log are off: its loggers
    # propagate to the root handler installed above, and the request-id
    # middleware writes one structured access record per request.
    uvicorn.run(create_app(settings), host=settings.api_host, port=port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
