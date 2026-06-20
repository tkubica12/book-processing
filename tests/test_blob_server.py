"""Tests for the private Blob-backed web server helpers."""

import pytest
from fastapi import HTTPException

from book_processing.blob_server import _blob_name_from_path, _parse_range


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
