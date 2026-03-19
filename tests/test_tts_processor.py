"""Tests for TTS helpers."""

from pathlib import Path

from book_processing.config import TTS_JOB_STALE_AFTER_SECONDS
from book_processing.config import TTS_MAX_CONCURRENT_JOBS, SOURCE_TTS_NAME
from book_processing.tts_processor import (
    TtsJobTracker,
    _build_audio_metadata,
    _display_book_title,
    _job_has_gone_stale,
)


def test_display_book_title_from_sanitized_name():
    assert _display_book_title("inference_engineering") == "Inference Engineering"


def test_build_audio_metadata_contains_book_type_and_language():
    metadata = _build_audio_metadata("inference_engineering", "summary_2min", "en")

    assert metadata["title"] == "Inference Engineering - Summary 2 min (en)"
    assert metadata["album"] == "Inference Engineering"
    assert metadata["language"] == "en"
    assert metadata["comment"] == "inference_engineering_summary_2min_en"


def test_build_audio_metadata_for_podcast():
    metadata = _build_audio_metadata("machines_of_loving_grace", "podcast_60min", "cs")

    assert metadata["title"] == "Machines Of Loving Grace - Podcast 60 min (cs)"


def test_job_has_gone_stale_only_after_timeout():
    assert not _job_has_gone_stale({"submitted_at": 0.0}, now=TTS_JOB_STALE_AFTER_SECONDS - 1)
    assert _job_has_gone_stale({"submitted_at": 0.0}, now=TTS_JOB_STALE_AFTER_SECONDS)
    assert not _job_has_gone_stale({}, now=TTS_JOB_STALE_AFTER_SECONDS)


def test_submit_item_limits_initial_chunk_submissions(monkeypatch, tmp_path: Path):
    tracker = TtsJobTracker(tmp_path)
    text_path = tmp_path / "source_tts_cs.md"
    text_path.write_text("dummy", encoding="utf-8")
    submitted_displays: list[str] = []

    monkeypatch.setattr(
        "book_processing.tts_processor.build_chunked_ssml",
        lambda text, lang, is_podcast: [f"chunk-{idx}" for idx in range(TTS_MAX_CONCURRENT_JOBS + 2)],
    )
    monkeypatch.setattr("book_processing.tts_processor._get_token", lambda: "token")

    def fake_submit_batch_synthesis(client, headers, ssml_inputs, display_name):
        submitted_displays.append(display_name)
        return f"job-{len(submitted_displays)}"

    monkeypatch.setattr("book_processing.tts_processor._submit_batch_synthesis", fake_submit_batch_synthesis)

    tracker._submit_item(
        object(),
        {
            "book_name": "example_book",
            "name": SOURCE_TTS_NAME,
            "lang": "cs",
            "text_path": text_path,
            "is_podcast": False,
        },
    )

    display = "example_book_source_tts_cs"
    assert len(tracker._active) == TTS_MAX_CONCURRENT_JOBS
    assert len(tracker._assembly[display]["pending_chunks"]) == 2
    assert submitted_displays == [
        f"{display}_chunk{idx}" for idx in range(1, TTS_MAX_CONCURRENT_JOBS + 1)
    ]
