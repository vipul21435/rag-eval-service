"""Request ids and access logs for the HTTP API.

Every request gets an id: the client's ``X-Request-ID`` when it sends a
well-formed one, otherwise a fresh UUID. The id is echoed in the response
header, bound to the request's context so that every log record written
while serving it carries the id (see ``ragsvc.logging_setup``), and
written to one access-log record per request with the method, path,
status and duration.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
# Ids safe to echo and to log: printable ASCII without separators, bounded
# in length. UUIDs, ULIDs, trace ids and plain tokens all pass.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

access_logger = logging.getLogger("ragsvc.access")

_request_id: ContextVar[str | None] = ContextVar("ragsvc_request_id", default=None)


def get_request_id() -> str | None:
    """The id of the request being served in the current context, if any."""
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


def accept_request_id(value: str | None) -> str | None:
    """``value`` stripped when it is a well-formed request id, otherwise ``None``."""
    if value is None:
        return None
    value = value.strip()
    return value if _VALID_REQUEST_ID.fullmatch(value) else None


@contextmanager
def bind_request_id(request_id: str) -> Iterator[str]:
    """Make ``request_id`` the current request id for the duration of the block."""
    token = _request_id.set(request_id)
    try:
        yield request_id
    finally:
        _request_id.reset(token)


class RequestIdMiddleware:
    """Pure ASGI middleware: assign the id, echo it, time the request and log it.

    Written against the ASGI interface rather than ``BaseHTTPMiddleware`` so
    streaming responses and background tasks are untouched. Access records
    are INFO for every response the application produced, including error
    statuses it chose to send, and ERROR when the application raised: the
    status is then 500, which is what the server's error handler sends.
    """

    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = accept_request_id(Headers(scope=scope).get(self.header_name)) or new_request_id()
        status = 500
        failed = False
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                message.setdefault("headers", [])
                MutableHeaders(scope=message).append(self.header_name, request_id)
            await send(message)

        with bind_request_id(request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            except BaseException:
                failed = True
                raise
            finally:
                duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
                client = scope.get("client")
                access_logger.log(
                    logging.ERROR if failed else logging.INFO,
                    "%s %s -> %d in %.1f ms",
                    scope["method"],
                    scope["path"],
                    status,
                    duration_ms,
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": duration_ms,
                        "request_id": request_id,
                        "client": client[0] if client else None,
                    },
                )
