"""JSON and text log output carrying the request id."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragsvc.logging_setup import HANDLER_NAME, JsonFormatter, RequestIdFilter, configure_logging
from ragsvc.middleware import REQUEST_ID_HEADER, RequestIdMiddleware, bind_request_id


@pytest.fixture
def captured() -> Iterator[tuple[logging.Logger, io.StringIO, logging.Formatter]]:
    """A private logger writing JSON through the request-id filter into a buffer."""
    logger = logging.getLogger("ragsvc.tests.logging")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.addFilter(RequestIdFilter())
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    yield logger, buffer, formatter
    logger.removeHandler(handler)


def lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_json_lines_carry_level_logger_message_extras_and_request_id(captured):
    logger, buffer, _ = captured

    logger.info("outside %s", "request", extra={"chunks": 3, "path": Path("kb.md")})
    with bind_request_id("req-9"):
        logger.warning("inside", extra={"status": 503})

    outside, inside = lines(buffer)
    assert outside["level"] == "INFO" and outside["logger"] == "ragsvc.tests.logging"
    assert outside["message"] == "outside request"
    assert outside["chunks"] == 3 and outside["path"] == "kb.md"
    assert "request_id" not in outside
    assert str(outside["time"]).endswith("+00:00")
    assert inside["level"] == "WARNING" and inside["request_id"] == "req-9" and inside["status"] == 503


def test_json_lines_include_exception_text(captured):
    logger, buffer, _ = captured

    try:
        raise ValueError("bad chunk")
    except ValueError:
        logger.exception("ingest failed")

    (record,) = lines(buffer)
    assert record["level"] == "ERROR" and record["message"] == "ingest failed"
    assert "ValueError: bad chunk" in str(record["exception"])


def test_configure_logging_replaces_only_its_own_handler():
    root = logging.getLogger()
    before = list(root.handlers)
    level_before = root.level
    try:
        first = configure_logging("DEBUG", "json")
        assert root.level == logging.DEBUG and isinstance(first.formatter, JsonFormatter)

        second = configure_logging("WARNING", "text")

        ours = [handler for handler in root.handlers if handler.name == HANDLER_NAME]
        assert ours == [second]
        assert root.level == logging.WARNING
        assert not isinstance(second.formatter, JsonFormatter)
        assert all(handler in root.handlers for handler in before)
    finally:
        for handler in list(root.handlers):
            if handler.name == HANDLER_NAME:
                root.removeHandler(handler)
        root.setLevel(level_before)


def test_text_format_shows_the_request_id_or_a_dash():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s [%(request_id)s] %(message)s"))
    logger = logging.getLogger("ragsvc.tests.text")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("no request")
        with bind_request_id("req-text"):
            logger.warning("in request")
    finally:
        logger.removeHandler(handler)

    assert stream.getvalue().splitlines() == ["WARNING [-] no request", "WARNING [req-text] in request"]


def test_records_written_while_serving_a_request_carry_its_id(captured):
    logger, buffer, _ = captured
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/work")
    def work() -> dict[str, bool]:
        logger.info("working", extra={"step": 1})
        return {"ok": True}

    TestClient(app).get("/work", headers={REQUEST_ID_HEADER: "req-served"})

    (record,) = lines(buffer)
    assert record["message"] == "working" and record["request_id"] == "req-served" and record["step"] == 1
