"""Tests for the private Blob-backed web server helpers."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from book_processing.blob_server import (
    _SESSION_COOKIE,
    _blob_name_from_path,
    _make_session,
    _parse_range,
    _require_authenticated,
)


def _request_with_cookie(cookie: str | None = None) -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", f"{_SESSION_COOKIE}={cookie}".encode("ascii")))
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "scheme": "https",
        "server": ("books.tomasonline.net", 443),
    })


def test_blob_name_defaults_to_index():
    assert _blob_name_from_path("") == "index.html"
    assert _blob_name_from_path("/") == "index.html"
    assert _blob_name_from_path("sample_book/") == "sample_book/index.html"


def test_blob_name_rejects_traversal():
    with pytest.raises(HTTPException) as error:
        _blob_name_from_path("../secret")

    assert error.value.status_code == 400


def test_parse_range_supports_common_browser_ranges():
    assert _parse_range("bytes=0-99", 1000) == (0, 99)
    assert _parse_range("bytes=500-", 1000) == (500, 999)
    assert _parse_range("bytes=-100", 1000) == (900, 999)


def test_parse_range_rejects_invalid_ranges():
    with pytest.raises(HTTPException) as error:
        _parse_range("bytes=1000-1001", 1000)

    assert error.value.status_code == 416


def test_github_session_allows_configured_user(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    session = _make_session("tkubica12", "tkubica12@gmail.com")

    assert _require_authenticated(_request_with_cookie(session)) is None


def test_github_session_rejects_other_user(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    session = _make_session("someone-else", "tkubica12@gmail.com")

    response = _require_authenticated(_request_with_cookie(session))

    assert response is not None
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://github.com/login/oauth/authorize?")


def test_github_session_rejects_tampered_cookie(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    session = _make_session("tkubica12", "tkubica12@gmail.com")
    tampered = f"{session}x"

    response = _require_authenticated(_request_with_cookie(tampered))

    assert response is not None
    assert response.status_code == 302
