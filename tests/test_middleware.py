"""Request ids: echoed or generated, bound to the request context, and written to the access log."""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragsvc import api
from ragsvc.middleware import (
    REQUEST_ID_HEADER,
    RequestIdMiddleware,
    accept_request_id,
    bind_request_id,
    get_request_id,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "detect_default_provider", lambda: "ollama")
    return TestClient(api.create_app())


def access_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """The attributes (level, ``extra`` fields) of each access-log record captured."""
    return [vars(record) for record in caplog.records if record.name == "ragsvc.access"]


def test_request_id_is_generated_when_absent_and_differs_per_request(client):
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert re.fullmatch(r"[0-9a-f]{32}", second)
    assert first != second


def test_client_request_id_is_echoed(client):
    response = client.get("/health", headers={"x-request-id": "trace-123.abc:7"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-123.abc:7"


@pytest.mark.parametrize("bad", ["", "   ", "has space", "x" * 129, "semi;colon"])
def test_malformed_request_ids_are_replaced(client, bad):
    response = client.get("/health", headers={REQUEST_ID_HEADER: bad})

    echoed = response.headers[REQUEST_ID_HEADER]
    assert echoed != bad.strip()
    assert re.fullmatch(r"[0-9a-f]{32}", echoed)


def test_accept_request_id_rules():
    assert accept_request_id(None) is None
    assert accept_request_id("  abc-1  ") == "abc-1"
    assert accept_request_id("x" * 128) == "x" * 128
    assert accept_request_id("x" * 129) is None
    assert accept_request_id("line\nbreak") is None
    assert accept_request_id("snow☃man") is None


def test_error_responses_carry_the_request_id(client):
    unprocessable = client.post("/api/ask", json={"question": ""}, headers={REQUEST_ID_HEADER: "req-422"})
    not_ready = client.get("/ready", headers={REQUEST_ID_HEADER: "req-503"})
    unsupported = client.post(
        "/api/upload",
        files={"file": ("data.csv", b"a,b", "text/csv")},
        headers={REQUEST_ID_HEADER: "req-415"},
    )

    assert (unprocessable.status_code, unprocessable.headers[REQUEST_ID_HEADER]) == (422, "req-422")
    assert (not_ready.status_code, not_ready.headers[REQUEST_ID_HEADER]) == (503, "req-503")
    assert (unsupported.status_code, unsupported.headers[REQUEST_ID_HEADER]) == (415, "req-415")


def test_request_id_is_bound_while_the_request_is_served():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/id")
    def route() -> dict[str, str | None]:
        return {"id": get_request_id()}

    response = TestClient(app).get("/id", headers={REQUEST_ID_HEADER: "bound-1"})

    assert response.json() == {"id": "bound-1"}
    assert response.headers[REQUEST_ID_HEADER] == "bound-1"
    assert get_request_id() is None


def test_bind_request_id_restores_the_previous_context():
    assert get_request_id() is None
    with bind_request_id("outer"):
        with bind_request_id("inner") as bound:
            assert bound == "inner" and get_request_id() == "inner"
        assert get_request_id() == "outer"
    assert get_request_id() is None


def test_access_log_record_has_method_path_status_duration_and_request_id(client, caplog):
    with caplog.at_level(logging.INFO, logger="ragsvc.access"):
        client.get("/health", headers={REQUEST_ID_HEADER: "req-log"})
        client.get("/ready", headers={REQUEST_ID_HEADER: "req-503"})

    ok, not_ready = access_records(caplog)
    assert ok["levelno"] == logging.INFO
    assert (ok["method"], ok["path"], ok["status"], ok["request_id"]) == ("GET", "/health", 200, "req-log")
    assert isinstance(ok["duration_ms"], float) and ok["duration_ms"] >= 0.0
    assert ok["client"] == "testclient"
    assert (ok["msg"] % ok["args"]).startswith("GET /health -> 200 in ")
    # An error status the application chose to send is still an INFO access record.
    assert (not_ready["levelno"], not_ready["status"], not_ready["request_id"]) == (
        logging.INFO,
        503,
        "req-503",
    )


def test_unhandled_errors_are_logged_at_error_level_with_status_500(caplog):
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger="ragsvc.access"):
        response = client.get("/boom")

    assert response.status_code == 500
    (record,) = access_records(caplog)
    assert record["levelno"] == logging.ERROR
    assert (record["method"], record["path"], record["status"]) == ("GET", "/boom", 500)


def test_lifespan_scope_passes_through_without_an_access_record(caplog):
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    with caplog.at_level(logging.INFO, logger="ragsvc.access"), TestClient(app):
        pass

    assert access_records(caplog) == []
