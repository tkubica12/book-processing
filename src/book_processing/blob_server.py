"""Serve private book output blobs through an authenticated Container App."""

from __future__ import annotations

from collections.abc import Iterator
import mimetypes
import os
from urllib.parse import unquote

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_CONTAINER = "books"

app = FastAPI(title="Book processing private site")


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


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
def serve_blob(request: Request, path: str, range_header: str | None = Header(default=None, alias="Range")):
    """Serve a generated static file or audio blob with range request support."""

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
