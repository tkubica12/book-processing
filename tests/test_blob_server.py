"""Tests for the private Blob-backed web server helpers."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from book_processing.blob_server import (
    _SESSION_COOKIE,
    _STATE_COOKIE,
    _encode_signed_payload,
    _blob_name_from_path,
    _make_session,
    _parse_range,
    _require_authenticated,
    app,
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


def test_github_session_allows_comma_separated_logins_without_email_restriction(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "tkubica12, octocat")
    monkeypatch.delenv("ALLOWED_GITHUB_EMAILS", raising=False)
    session = _make_session("octocat", "octocat@example.com")

    assert _require_authenticated(_request_with_cookie(session)) is None


def test_github_session_enforces_comma_separated_email_allowlist(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "tkubica12, octocat")
    monkeypatch.setenv("ALLOWED_GITHUB_EMAILS", "tkubica12@gmail.com, allowed@example.com")
    session = _make_session("octocat", "octocat@example.com")

    response = _require_authenticated(_request_with_cookie(session))

    assert response is not None
    assert response.status_code == 302


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


def test_github_session_redirects_when_expired(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    session = _encode_signed_payload({
        "login": "tkubica12",
        "email": "tkubica12@gmail.com",
        "exp": 1,
    })

    response = _require_authenticated(_request_with_cookie(session))

    assert response is not None
    assert response.status_code == 302


def test_github_session_redirects_when_expiry_is_malformed(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    session = _encode_signed_payload({
        "login": "tkubica12",
        "email": "tkubica12@gmail.com",
        "exp": "not-a-timestamp",
    })

    response = _require_authenticated(_request_with_cookie(session))

    assert response is not None
    assert response.status_code == 302


def test_auth_redirect_uses_public_base_url_and_clears_stale_session(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://books.tomasonline.net")

    response = _require_authenticated(_request_with_cookie("stale-session"))

    assert response is not None
    assert response.status_code == 302
    assert "redirect_uri=https%3A%2F%2Fbooks.tomasonline.net%2Foauth%2Fgithub%2Fcallback" in response.headers["location"]
    assert any(header.startswith("book_site_session=") for header in _set_cookie_headers(response))


def test_login_defaults_return_to_public_site_root(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://books.tomasonline.net")

    client = TestClient(app)
    response = client.get("/login", follow_redirects=False)

    assert response.status_code == 302
    state_cookie = response.cookies.get(_STATE_COOKIE)
    assert state_cookie is not None
    state_payload = _decode_oauth_state(state_cookie)
    assert state_payload["return_to"] == "https://books.tomasonline.net/"


def test_callback_restarts_login_when_github_code_expired(monkeypatch):
    monkeypatch.setenv("GITHUB_OAUTH_COOKIE_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://books.tomasonline.net")

    state = "state-value"
    state_cookie = _encode_signed_payload({
        "state": state,
        "return_to": "https://books.tomasonline.net/book/",
        "exp": 9999999999,
    })

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def post(self, url, data, headers):
            return FakeResponse()

    class FakeResponse:
        status_code = 400

        def json(self):
            return {"error": "bad_verification_code"}

        def raise_for_status(self):
            raise AssertionError("expired OAuth code should be handled before raise_for_status")

    monkeypatch.setattr("book_processing.blob_server.httpx.Client", FakeClient)

    client = TestClient(app)
    client.cookies.set(_STATE_COOKIE, state_cookie)
    client.cookies.set(_SESSION_COOKIE, "stale-session")
    response = client.get(f"/oauth/github/callback?code=expired-code&state={state}", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://github.com/login/oauth/authorize?")
    new_state_payload = _decode_oauth_state(response.cookies[_STATE_COOKIE])
    assert new_state_payload["return_to"] == "https://books.tomasonline.net/book/"
    assert any(header.startswith("book_site_session=") for header in _set_cookie_headers(response))


def _set_cookie_headers(response) -> list[str]:
    if hasattr(response.headers, "get_list"):
        return response.headers.get_list("set-cookie")
    return [
        value.decode("latin-1")
        for name, value in response.raw_headers
        if name.decode("latin-1").casefold() == "set-cookie"
    ]


def _decode_oauth_state(value: str) -> dict:
    body, _ = value.rsplit(".", 1)
    import base64
    import json

    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
