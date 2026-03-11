"""Azure Content Understanding document extraction helpers."""

import base64
import logging
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx

from book_processing.auth import get_cognitive_token
from book_processing.config import (
    CONTENT_UNDERSTANDING_ANALYZER_ID,
    CONTENT_UNDERSTANDING_API_KEY,
    CONTENT_UNDERSTANDING_API_VERSION,
    CONTENT_UNDERSTANDING_ENDPOINT,
    CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS,
    CONTENT_UNDERSTANDING_PROCESSING_LOCATION,
)

logger = logging.getLogger(__name__)


class ContentUnderstandingNoUsableMarkdownError(RuntimeError):
    """Raised when Content Understanding succeeds but returns no usable markdown."""


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize a Foundry endpoint for REST calls."""
    return endpoint.rstrip("/")


def _is_placeholder_markdown(markdown: str) -> bool:
    """Return True when the analyzer produced an empty fenced placeholder block."""
    normalized = markdown.strip()
    return normalized.startswith("```") and normalized.endswith("```") and normalized.count("```") == 2 and not normalized.strip("` \r\n\tabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _build_auth_headers() -> dict[str, str]:
    """Build authentication headers for Content Understanding."""
    headers = {
        "Content-Type": "application/json",
        "x-ms-client-request-id": str(uuid.uuid4()),
    }
    if CONTENT_UNDERSTANDING_API_KEY:
        headers["Ocp-Apim-Subscription-Key"] = CONTENT_UNDERSTANDING_API_KEY
    else:
        headers["Authorization"] = f"Bearer {get_cognitive_token()}"
    return headers


def _analyze_url() -> str:
    """Build the analyze URL for the configured analyzer."""
    if not CONTENT_UNDERSTANDING_ENDPOINT:
        raise RuntimeError(
            "CONTENT_UNDERSTANDING_ENDPOINT is not configured. Set it in .env to your Foundry resource endpoint."
        )

    url = (
        f"{_normalize_endpoint(CONTENT_UNDERSTANDING_ENDPOINT)}/contentunderstanding/"
        f"analyzers/{quote(CONTENT_UNDERSTANDING_ANALYZER_ID, safe='._-')}:analyze"
        f"?api-version={CONTENT_UNDERSTANDING_API_VERSION}"
    )
    if CONTENT_UNDERSTANDING_PROCESSING_LOCATION:
        url += f"&processingLocation={CONTENT_UNDERSTANDING_PROCESSING_LOCATION}"
    return url


def _poll_result(client: httpx.Client, operation_location: str) -> dict:
    """Poll the async analyze operation until it completes."""
    while True:
        response = client.get(operation_location, headers=_build_auth_headers())
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "Succeeded":
            return payload
        if status in {"Failed", "Canceled"}:
            raise RuntimeError(f"Content Understanding analysis failed: {payload}")
        time.sleep(CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS)


def _extract_markdown(result_payload: dict, pdf_name: str) -> str:
    """Extract usable markdown from a completed analyze response."""
    contents = result_payload.get("result", {}).get("contents", [])
    if not contents:
        raise ContentUnderstandingNoUsableMarkdownError(
            f"Content Understanding returned no contents for {pdf_name}"
        )

    markdown = contents[0].get("markdown")
    if not markdown:
        raise ContentUnderstandingNoUsableMarkdownError(
            f"Content Understanding returned no markdown for {pdf_name}"
        )
    if _is_placeholder_markdown(markdown):
        raise ContentUnderstandingNoUsableMarkdownError(
            f"Content Understanding returned placeholder markdown for {pdf_name}. "
            "The analyzer completed, but produced no usable markdown content."
        )
    return markdown


def _analyze_input_to_markdown(input_name: str, input_bytes: bytes, mime_type: str) -> str:
    """Analyze one document or image input with Content Understanding and return markdown."""
    encoded_data = base64.b64encode(input_bytes).decode("ascii")
    body = {
        "inputs": [
            {
                "name": input_name,
                "data": encoded_data,
                "mimeType": mime_type,
            }
        ]
    }

    with httpx.Client(timeout=300) as client:
        response = client.post(_analyze_url(), json=body, headers=_build_auth_headers())
        response.raise_for_status()
        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise RuntimeError("Content Understanding response did not include Operation-Location")

        logger.info("Content Understanding accepted %s; polling result", input_name)
        result_payload = _poll_result(client, operation_location)

    markdown = _extract_markdown(result_payload, input_name)
    usage = result_payload.get("usage") or {}
    if usage:
        logger.info("Content Understanding usage for %s: %s", input_name, usage)
    return markdown


def analyze_pdf_to_markdown(pdf_path: Path) -> str:
    """Analyze one PDF with Content Understanding and return markdown."""
    return _analyze_input_to_markdown(pdf_path.name, pdf_path.read_bytes(), "application/pdf")


def analyze_image_to_markdown(
    image_name: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    """Analyze one image with Content Understanding and return markdown."""
    return _analyze_input_to_markdown(image_name, image_bytes, mime_type)
