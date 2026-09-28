"""Structured logging: JSON lines that carry the request id.

``configure_logging`` installs one handler on the root logger that writes
either JSON objects (one per line, for log collectors) or readable text.
Both carry the id of the HTTP request being served, taken from the request
context that ``ragsvc.middleware`` maintains, so every record a request
produced can be found from the id in its response header.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from ragsvc.config import LogFormat
from ragsvc.middleware import get_request_id

HANDLER_NAME = "ragsvc"
NO_REQUEST = "-"
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"

# Attributes every LogRecord has; anything else on a record came from ``extra``.
# ``color_message`` is uvicorn's terminal-coloured duplicate of ``message``.
_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("ragsvc", logging.INFO, __file__, 0, "", (), None))) | {
    "message",
    "asctime",
    "request_id",
    "color_message",
}


class RequestIdFilter(logging.Filter):
    """Attach the current request id to each record; ``-`` outside a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = get_request_id() or NO_REQUEST
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, message, request id and ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id and request_id != NO_REQUEST:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RECORD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_format: LogFormat = "json") -> logging.Handler:
    """Install the service's root handler, replacing the one from an earlier call.

    Other handlers on the root logger (a test runner's, an embedding
    program's) are left in place. Returns the installed handler.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.name == HANDLER_NAME:
            root.removeHandler(existing)
            existing.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.name = HANDLER_NAME
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    return handler
