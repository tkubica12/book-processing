"""Tests for Content Understanding integration helpers."""

import pytest

from book_processing.content_understanding import _extract_markdown, _is_placeholder_markdown, _normalize_endpoint
from book_processing.prompt_templates import render_prompt


def test_normalize_endpoint_removes_trailing_slash():
    assert _normalize_endpoint("https://example.services.ai.azure.com/") == "https://example.services.ai.azure.com"


def test_summary_prompt_still_mentions_shorter_output_for_short_sources():
    prompt = render_prompt(
        "simple_summary_user.j2",
        description="technical deep-dive with implementation details",
        target_words=3200,
        source_md="Short source",
    )

    assert "produce a shorter answer rather than padding it" in prompt


def test_content_understanding_requires_endpoint(monkeypatch):
    monkeypatch.setattr("book_processing.content_understanding.CONTENT_UNDERSTANDING_ENDPOINT", "")

    with pytest.raises(RuntimeError, match="CONTENT_UNDERSTANDING_ENDPOINT"):
        from book_processing.content_understanding import _analyze_url

        _analyze_url()


def test_placeholder_markdown_detection():
    assert _is_placeholder_markdown("```text\n\n```\n") is True
    assert _is_placeholder_markdown("# Heading\n\nBody") is False


def test_extract_markdown_rejects_placeholder():
    payload = {
        "result": {
            "contents": [
                {
                    "markdown": "```text\n\n```\n",
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="placeholder markdown"):
        _extract_markdown(payload, "example.pdf")


def test_extract_markdown_returns_usable_content():
    payload = {
        "result": {
            "contents": [
                {
                    "markdown": "# Title\n\nUseful body",
                }
            ]
        }
    }

    assert _extract_markdown(payload, "example.pdf") == "# Title\n\nUseful body"
