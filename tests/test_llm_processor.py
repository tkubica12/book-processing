"""Tests for LLM retry and filter-recovery helpers."""

from pathlib import Path

import pytest

from book_processing.llm_processor import (
    ContentFilterError,
    LlmRequestTimeoutError,
    _call_llm,
    _is_content_filter_error,
    _is_timeout_error,
    _recover_filtered_text,
    _sanitize_filtered_prompt,
    _split_text_for_filter_recovery,
    is_arxiv_source,
    summary_types_for_source,
)


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


def test_is_timeout_error_matches_timeout_message():
    assert _is_timeout_error(Exception("Request timed out.")) is True


def test_split_text_for_filter_recovery_prefers_paragraph_boundaries():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."

    left, right = _split_text_for_filter_recovery(text)

    assert left == "Paragraph one.\n\nParagraph two."
    assert right == "Paragraph three.\n\nParagraph four."


def test_summary_types_for_arxiv_source_skip_long_formats():
    source = "# Paper\n\narXiv:2601.12345\n\nShort research paper."

    summary_types = summary_types_for_source("sample_paper", source)

    assert is_arxiv_source("sample_paper", source) is True
    assert list(summary_types) == ["summary_5min", "summary_20min", "podcast_20min"]
    assert "podcast_60min" not in summary_types


def test_summary_types_for_paper_shape_skip_long_formats():
    source = "# HyDRA\n\nAbstract\n\nA short paper.\n\n1 Introduction\n\nBody.\n\nReferences\n\nOng et al."

    summary_types = summary_types_for_source("hydra_hybrid_dynamic_routing", source)

    assert is_arxiv_source("hydra_hybrid_dynamic_routing", source) is True
    assert list(summary_types) == ["summary_5min", "summary_20min", "podcast_20min"]


def test_summary_types_for_book_include_all_formats():
    summary_types = summary_types_for_source("sample_book", "# Book\n\nLong source.")

    assert is_arxiv_source("sample_book", "# Book") is False
    assert "podcast_60min" in summary_types


def test_call_llm_raises_content_filter_after_sanitized_retry():
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def create(self, *, messages, **_kwargs):
            self.calls.append(messages[1]["content"])
            raise Exception("Error code: 400 - {'error': {'code': 'content_filter'}}")

    fake_completions = FakeCompletions()
    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("FakeChat", (), {"completions": fake_completions})()},
    )()

    with pytest.raises(ContentFilterError):
        _call_llm(fake_client, "system", "The text contains sexual content.", max_tokens=1000)

    assert len(fake_completions.calls) == 2
    assert "sexual" in fake_completions.calls[0].lower()
    assert "intimate" in fake_completions.calls[1].lower()


def test_recover_filtered_text_splits_and_combines(monkeypatch, tmp_path: Path):
    left_text = " ".join(["alpha"] * 80)
    right_text = " ".join(["beta"] * 80)
    source_text = f"{left_text}\n\n{right_text}"

    def fake_call(
        _client,
        _system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16000,
        split_on_timeout: bool = False,
    ) -> str:
        assert max_tokens == 8000
        assert split_on_timeout is True
        if "alpha" in user_prompt and "beta" in user_prompt:
            raise ContentFilterError("filtered")
        return user_prompt.upper()

    monkeypatch.setattr("book_processing.llm_processor._call_llm", fake_call)

    result = _recover_filtered_text(
        client=None,
        system_prompt="system",
        render_user_prompt=lambda text: text,
        source_text=source_text,
        lang="en",
        partial_dir=tmp_path,
        cache_prefix="sample",
        max_tokens=8000,
    )

    assert result == f"{left_text.upper()}\n\n{right_text.upper()}"


def test_recover_filtered_text_inserts_placeholder_for_irreducible_fragment(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "book_processing.llm_processor._call_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ContentFilterError("filtered")),
    )

    result = _recover_filtered_text(
        client=None,
        system_prompt="system",
        render_user_prompt=lambda text: text,
        source_text="short filtered fragment",
        lang="en",
        partial_dir=tmp_path,
        cache_prefix="sample",
        max_tokens=8000,
    )

    assert "content safety filtering" in result.lower()


def test_recover_filtered_text_splits_on_timeout(monkeypatch, tmp_path: Path):
    left_text = " ".join(["alpha"] * 80)
    right_text = " ".join(["beta"] * 80)
    source_text = f"{left_text}\n\n{right_text}"

    def fake_call(
        _client,
        _system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16000,
        split_on_timeout: bool = False,
    ) -> str:
        assert split_on_timeout is True
        if "alpha" in user_prompt and "beta" in user_prompt:
            raise LlmRequestTimeoutError("timed out")
        return user_prompt.upper()

    monkeypatch.setattr("book_processing.llm_processor._call_llm", fake_call)

    result = _recover_filtered_text(
        client=None,
        system_prompt="system",
        render_user_prompt=lambda text: text,
        source_text=source_text,
        lang="en",
        partial_dir=tmp_path,
        cache_prefix="sample",
        max_tokens=8000,
    )

    assert result == f"{left_text.upper()}\n\n{right_text.upper()}"
