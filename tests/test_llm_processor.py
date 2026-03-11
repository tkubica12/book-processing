"""Tests for LLM retry helpers."""

from book_processing.llm_processor import _is_content_filter_error, _sanitize_filtered_prompt


def test_is_content_filter_error_matches_azure_message():
    error = Exception("Error code: 400 - {'error': {'code': 'content_filter', 'innererror': {'code': 'ResponsibleAIPolicyViolation'}}}")

    assert _is_content_filter_error(error) is True


def test_sanitize_filtered_prompt_rewrites_trigger_terms():
    prompt = "The text mentions sexual content, pornography, nudity, and rape."

    sanitized = _sanitize_filtered_prompt(prompt)

    assert "sexual" not in sanitized.lower()
    assert "pornography" not in sanitized.lower()
    assert "nudity" not in sanitized.lower()
    assert "rape" not in sanitized.lower()
    assert "intimate" in sanitized.lower()
