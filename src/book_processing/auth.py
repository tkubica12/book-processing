"""Shared Azure CLI token acquisition for Cognitive Services."""

import json
import logging
import shutil
import subprocess
import threading
import time

from book_processing.config import AZURE_COGNITIVE_SCOPE

logger = logging.getLogger(__name__)

_cached_token: str | None = None
_token_expires_on: float = 0.0
_token_lock = threading.Lock()


def _fetch_cli_token() -> tuple[str, float]:
    """Fetch a fresh Cognitive Services token from Azure CLI."""
    az_executable = shutil.which("az.cmd") or shutil.which("az")
    if not az_executable:
        raise FileNotFoundError("Azure CLI executable 'az' was not found in PATH")

    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            az_executable,
            "account",
            "get-access-token",
            "--resource",
            AZURE_COGNITIVE_SCOPE.removesuffix("/.default"),
            "-o",
            "json",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(result.stdout)
    token = payload["accessToken"]
    expires_on = float(payload.get("expires_on") or (time.time() + 3600))
    return token, expires_on


def get_cognitive_token() -> str:
    """Return a cached Cognitive Services bearer token from Azure CLI."""
    global _cached_token, _token_expires_on
    with _token_lock:
        if _cached_token and time.time() < _token_expires_on - 300:
            return _cached_token

    token, expires_on = _fetch_cli_token()
    with _token_lock:
        _cached_token = token
        _token_expires_on = expires_on
    logger.debug("Azure CLI token refreshed (expires in %.0f min)", (_token_expires_on - time.time()) / 60)
    return token