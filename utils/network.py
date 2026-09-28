"""Network helpers: a shared HTTP session with retries and a port probe."""

from __future__ import annotations

import socket

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return a process-wide ``requests.Session`` that retries on 5xx responses."""
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        _session.mount("http://", HTTPAdapter(max_retries=retries))
    return _session


def is_port_available(port: int) -> bool:
    """True when nothing accepts connections on ``127.0.0.1:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0
