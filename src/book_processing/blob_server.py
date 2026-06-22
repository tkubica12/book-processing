"""Serve private book output blobs through an authenticated Container App."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterator
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import time
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from fastapi import FastAPI, Header, HTTPException, Request, Response
import httpx
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_CONTAINER = "books"
_SESSION_COOKIE = "book_site_session"
_STATE_COOKIE = "book_site_oauth_state"
_SESSION_TTL_SECONDS = 60 * 60 * 24 * 14
_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_API_URL = "https://api.github.com"

app = FastAPI(title="Book processing private site")


class _OAuthRenewRequired(Exception):
    """Raised when the browser should restart the OAuth login flow."""


def _public_base_url(request: Request) -> str:
    return os.getenv("PUBLIC_BASE_URL", str(request.base_url).rstrip("/")).rstrip("/")


def _public_request_url(request: Request) -> str:
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{_public_base_url(request)}{request.url.path}{query}"


def _safe_return_to(request: Request, value: str | None) -> str:
    base_url = _public_base_url(request)
    if not value:
        return f"{base_url}/"

    parsed = urlsplit(value)
    if not parsed.netloc:
        return f"{base_url}/{value.lstrip('/')}"

    base = urlsplit(base_url)
    if parsed.scheme == base.scheme and parsed.netloc == base.netloc:
        return value
    return f"{base_url}/"


def _csv_env_values(name: str) -> set[str]:
    return {item.strip().casefold() for item in os.getenv(name, "").split(",") if item.strip()}


def _allowed_github_logins() -> set[str]:
    return _csv_env_values("ALLOWED_GITHUB_LOGINS") or {"tkubica12"}


def _allowed_github_emails() -> set[str]:
    return _csv_env_values("ALLOWED_GITHUB_EMAILS")


def _github_identity_is_allowed(login: str, verified_emails: set[str]) -> bool:
    if login.casefold() not in _allowed_github_logins():
        return False
    allowed_emails = _allowed_github_emails()
    return not allowed_emails or bool(verified_emails & allowed_emails)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def _cookie_secret() -> str:
    return _required_env("GITHUB_OAUTH_COOKIE_SECRET")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(value: str) -> str:
    digest = hmac.new(_cookie_secret().encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(digest)


def _encode_signed_payload(payload: dict[str, Any]) -> str:
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{body}.{_sign(body)}"


def _decode_signed_payload(value: str | None) -> dict[str, Any] | None:
    if not value or "." not in value:
        return None
    body, signature = value.rsplit(".", 1)
    if not hmac.compare_digest(signature, _sign(body)):
        return None
    try:
        payload = json.loads(_b64url_decode(body))
    except (binascii.Error, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _make_session(login: str, email: str) -> str:
    return _encode_signed_payload({
        "login": login,
        "email": email,
        "exp": int(time.time()) + _SESSION_TTL_SECONDS,
    })


def _valid_session(request: Request) -> bool:
    session = _decode_signed_payload(request.cookies.get(_SESSION_COOKIE))
    if not session:
        return False
    try:
        expires_at = int(session.get("exp", 0))
    except (TypeError, ValueError):
        return False
    if expires_at < int(time.time()):
        return False
    login = str(session.get("login", "")).casefold()
    email = str(session.get("email", "")).casefold()
    verified_emails = {email} if email else set()
    return _github_identity_is_allowed(login, verified_emails)


def _login_redirect(request: Request, return_to: str | None = None) -> RedirectResponse:
    return_to = _safe_return_to(request, return_to or _public_request_url(request))
    state = secrets.token_urlsafe(32)
    state_cookie = _encode_signed_payload({
        "state": state,
        "return_to": return_to,
        "exp": int(time.time()) + 600,
    })
    query = urlencode({
        "client_id": _required_env("GITHUB_OAUTH_CLIENT_ID"),
        "redirect_uri": f"{_public_base_url(request)}/oauth/github/callback",
        "scope": "read:user user:email",
        "state": state,
    })
    response = RedirectResponse(f"{_GITHUB_AUTHORIZE_URL}?{query}", status_code=302)
    response.set_cookie(_STATE_COOKIE, state_cookie, httponly=True, secure=True, samesite="lax", max_age=600)
    return response


def _github_user_and_verified_email(access_token: str) -> tuple[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=20.0) as client:
        user_response = client.get(f"{_GITHUB_API_URL}/user", headers=headers)
        if user_response.status_code == 401:
            raise _OAuthRenewRequired()
        user_response.raise_for_status()
        user = user_response.json()
        emails_response = client.get(f"{_GITHUB_API_URL}/user/emails", headers=headers)
        if emails_response.status_code == 401:
            raise _OAuthRenewRequired()
        emails_response.raise_for_status()
        emails = emails_response.json()
    login = str(user.get("login", "")).strip()
    verified_emails = {
        str(item.get("email", "")).strip().casefold()
        for item in emails
        if isinstance(item, dict) and item.get("verified")
    }
    allowed_emails = _allowed_github_emails()
    matched_allowed_emails = verified_emails & allowed_emails
    if allowed_emails and not matched_allowed_emails:
        raise HTTPException(status_code=403, detail="Required GitHub email is not verified on this account")
    return login, next(iter(matched_allowed_emails or verified_emails), "")


def _exchange_code_for_token(code: str, request: Request) -> str:
    body = {
        "client_id": _required_env("GITHUB_OAUTH_CLIENT_ID"),
        "client_secret": _required_env("GITHUB_OAUTH_CLIENT_SECRET"),
        "code": code,
        "redirect_uri": f"{_public_base_url(request)}/oauth/github/callback",
    }
    headers = {"Accept": "application/json"}
    with httpx.Client(timeout=20.0) as client:
        response = client.post(_GITHUB_TOKEN_URL, data=body, headers=headers)
        if response.status_code == 400:
            token_payload = response.json()
            if token_payload.get("error") == "bad_verification_code":
                raise _OAuthRenewRequired()
        response.raise_for_status()
    token_payload = response.json()
    access_token = token_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise _OAuthRenewRequired()
    return access_token


def _require_authenticated(request: Request) -> RedirectResponse | None:
    if _valid_session(request):
        return None
    response = _login_redirect(request)
    response.delete_cookie(_SESSION_COOKIE)
    return response


def _container_client() -> ContainerClient:
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = os.getenv("BLOB_CONTAINER_NAME", _DEFAULT_CONTAINER)
    service = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    return service.get_container_client(container_name)


def _blob_name_from_path(path: str) -> str:
    decoded = unquote(path).lstrip("/")
    if not decoded:
        return "index.html"
    if decoded.endswith("/"):
        return f"{decoded}index.html"
    if "\\" in decoded or any(part == ".." for part in decoded.split("/")):
        raise HTTPException(status_code=400, detail="Invalid path")
    return decoded


def _parse_range(range_header: str | None, size: int) -> tuple[int, int] | None:
    if not range_header:
        return None
    if not range_header.startswith("bytes=") or "," in range_header:
        raise HTTPException(status_code=416, detail="Unsupported range")

    start_text, separator, end_text = range_header.removeprefix("bytes=").partition("-")
    if not separator:
        raise HTTPException(status_code=416, detail="Invalid range")

    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix_length = int(end_text)
            if suffix_length == 0:
                raise HTTPException(status_code=416, detail="Invalid range")
            start = max(size - suffix_length, 0)
            end = size - 1
    except ValueError as error:
        raise HTTPException(status_code=416, detail="Invalid range") from error

    if start < 0 or end < start or start >= size:
        raise HTTPException(status_code=416, detail="Range not satisfiable")
    return start, min(end, size - 1)


def _content_type(blob_name: str, blob_content_type: str | None) -> str:
    if blob_content_type and blob_content_type != "application/octet-stream":
        return blob_content_type
    guessed, _ = mimetypes.guess_type(blob_name)
    return guessed or "application/octet-stream"


def _blob_chunks(container: ContainerClient, blob_name: str, offset: int, length: int) -> Iterator[bytes]:
    stream = container.download_blob(blob_name, offset=offset, length=length, max_concurrency=4)
    yield from stream.chunks()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return a simple health probe response."""

    return {"status": "ok"}


@app.get("/login")
def login(request: Request) -> RedirectResponse:
    """Start GitHub OAuth login."""

    return_to = request.query_params.get("return_to") or f"{_public_base_url(request)}/"
    return _login_redirect(request, return_to=return_to)


@app.get("/oauth/github/callback")
def github_oauth_callback(request: Request, code: str = "", state: str = "") -> Response:
    """Complete GitHub OAuth login and issue a signed local session cookie."""

    state_payload = _decode_signed_payload(request.cookies.get(_STATE_COOKIE))
    if (
        not state_payload
        or int(state_payload.get("exp", 0)) < int(time.time())
        or not hmac.compare_digest(str(state_payload.get("state", "")), state)
    ):
        return _login_redirect(request, return_to=f"{_public_base_url(request)}/")
    if not code:
        return _login_redirect(request, return_to=str(state_payload.get("return_to") or f"{_public_base_url(request)}/"))

    try:
        access_token = _exchange_code_for_token(code, request)
        login_name, email = _github_user_and_verified_email(access_token)
    except _OAuthRenewRequired:
        response = _login_redirect(request, return_to=str(state_payload.get("return_to") or f"{_public_base_url(request)}/"))
        response.delete_cookie(_SESSION_COOKIE)
        return response
    verified_emails = {email} if email else set()
    if not _github_identity_is_allowed(login_name, verified_emails):
        raise HTTPException(status_code=403, detail="GitHub account is not authorized")

    response = RedirectResponse(str(state_payload.get("return_to") or "/"), status_code=302)
    response.set_cookie(
        _SESSION_COOKIE,
        _make_session(login_name, email),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_SESSION_TTL_SECONDS,
    )
    response.delete_cookie(_STATE_COOKIE)
    return response


@app.get("/logout")
def logout() -> PlainTextResponse:
    """Clear the local application session."""

    response = PlainTextResponse("Signed out")
    response.delete_cookie(_SESSION_COOKIE)
    response.delete_cookie(_STATE_COOKIE)
    return response


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
def serve_blob(request: Request, path: str, range_header: str | None = Header(default=None, alias="Range")):
    """Serve a generated static file or audio blob with range request support."""

    auth_redirect = _require_authenticated(request)
    if auth_redirect is not None:
        return auth_redirect
    blob_name = _blob_name_from_path(path)
    container = _container_client()
    blob = container.get_blob_client(blob_name)
    try:
        properties = blob.get_blob_properties()
    except ResourceNotFoundError as error:
        if "." not in blob_name.rsplit("/", maxsplit=1)[-1]:
            return serve_blob(request, f"{blob_name}/", range_header)
        raise HTTPException(status_code=404, detail="Not found") from error

    size = properties.size
    byte_range = _parse_range(range_header, size)
    content_type = _content_type(blob_name, properties.content_settings.content_type)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=300",
    }

    if byte_range is None:
        status_code = 200
        start = 0
        end = size - 1
    else:
        status_code = 206
        start, end = byte_range
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    length = end - start + 1
    headers["Content-Length"] = str(length)
    if request.method == "HEAD":
        return Response(status_code=status_code, media_type=content_type, headers=headers)
    return StreamingResponse(
        _blob_chunks(container, blob_name, start, length),
        status_code=status_code,
        media_type=content_type,
        headers=headers,
    )
