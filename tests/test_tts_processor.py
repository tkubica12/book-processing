"""Tests for TTS helpers."""

from pathlib import Path
from unittest.mock import Mock

import httpx

from book_processing.config import TTS_JOB_STALE_AFTER_SECONDS
from book_processing.config import TTS_MAX_CONCURRENT_JOBS, SOURCE_TTS_NAME
from book_processing.tts_processor import (
    TtsJobTracker,
    _auth_headers,
    _build_audio_metadata,
    _display_book_title,
    _job_has_gone_stale,
    _submit_batch_synthesis,
)


def test_display_book_title_from_sanitized_name():
    assert _display_book_title("inference_engineering") == "Inference Engineering"


def test_build_audio_metadata_contains_book_type_and_language():
    metadata = _build_audio_metadata("inference_engineering", "summary_5min", "en")

    assert metadata["title"] == "Inference Engineering - Summary 5 min (en)"
    assert metadata["album"] == "Inference Engineering"
    assert metadata["language"] == "en"
    assert metadata["comment"] == "inference_engineering_summary_5min_en"


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


def test_handle_completed_job_ignores_late_chunk_after_assembly(monkeypatch, tmp_path: Path):
    tracker = TtsJobTracker(tmp_path)
    audio_path = tmp_path / "book" / "book_source_tts_en.mp3"
    display = "book_source_tts_en"

    tracker._assembly[display] = {
        "total": 2,
        "parts": {},
        "audio_path": audio_path,
        "book_name": "book",
        "name": SOURCE_TTS_NAME,
        "lang": "en",
    }

    downloaded_chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    monkeypatch.setattr(
        "book_processing.tts_processor._download_audio_bytes",
        lambda client, job_data: downloaded_chunks.pop(0),
    )
    monkeypatch.setattr("book_processing.tts_processor._delete_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("book_processing.tts_processor._write_mp3_metadata", lambda *args, **kwargs: None)

    tracker._handle_completed_job(
        object(),
        {},
        {
            "display": f"{display}_chunk1",
            "job_id": "job-1",
            "audio_path": audio_path,
            "book_name": "book",
            "name": SOURCE_TTS_NAME,
            "lang": "en",
            "parent_display": display,
            "chunk_idx": 0,
        },
        {},
    )
    tracker._handle_completed_job(
        object(),
        {},
        {
            "display": f"{display}_chunk2",
            "job_id": "job-2",
            "audio_path": audio_path,
            "book_name": "book",
            "name": SOURCE_TTS_NAME,
            "lang": "en",
            "parent_display": display,
            "chunk_idx": 1,
        },
        {},
    )

    assert audio_path.read_bytes() == b"chunk-1chunk-2"


def test_submit_batch_synthesis_refreshes_token_on_401(monkeypatch):
    monkeypatch.setattr("book_processing.tts_processor.uuid.uuid4", lambda: "job-id")

    refreshed = {"count": 0}

    def fake_invalidate() -> None:
        refreshed["count"] += 1

    monkeypatch.setattr("book_processing.tts_processor.invalidate_cognitive_token", fake_invalidate)
    monkeypatch.setattr("book_processing.tts_processor._get_token", lambda: "fresh-token")
    monkeypatch.setattr("book_processing.tts_processor.time.sleep", lambda _: None)

    first_response = Mock()
    first_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401",
        request=httpx.Request("PUT", "https://example.invalid"),
        response=httpx.Response(401, request=httpx.Request("PUT", "https://example.invalid")),
    )
    second_response = Mock()
    second_response.raise_for_status.return_value = None

    calls: list[dict[str, str]] = []

    class FakeClient:
        def put(self, url, json, headers):
            calls.append(headers)
            return first_response if len(calls) == 1 else second_response

    job_id = _submit_batch_synthesis(
        FakeClient(),
        _auth_headers("stale-token"),
        ["<speak>ok</speak>"],
        "display-name",
    )

    assert job_id == "job-id"
    assert refreshed["count"] == 1
    assert calls == [
        _auth_headers("stale-token"),
        _auth_headers("fresh-token"),
    ]


def test_retry_failed_job_refreshes_token_before_resubmit(monkeypatch, tmp_path: Path):
    tracker = TtsJobTracker(tmp_path)
    refreshed = {"count": 0}
    submit_headers: list[dict[str, str]] = []

    def fake_invalidate() -> None:
        refreshed["count"] += 1

    monkeypatch.setattr("book_processing.tts_processor.invalidate_cognitive_token", fake_invalidate)
    monkeypatch.setattr("book_processing.tts_processor._get_token", lambda: "fresh-token")
    monkeypatch.setattr("book_processing.tts_processor.time.sleep", lambda _: None)
    monkeypatch.setattr("book_processing.tts_processor._delete_job", lambda *args, **kwargs: None)

    def fake_submit_batch_synthesis(client, headers, ssml_inputs, display_name):
        submit_headers.append(headers)
        return "new-job-id"

    monkeypatch.setattr("book_processing.tts_processor._submit_batch_synthesis", fake_submit_batch_synthesis)

    job = {
        "job_id": "old-job-id",
        "display": "book_source_tts_en_chunk1",
        "ssml_inputs": ["<speak>ok</speak>"],
        "retries": 0,
    }

    retried = tracker._retry_failed_job(
        object(),
        _auth_headers("stale-token"),
        job,
        RuntimeError("boom"),
    )

    assert retried is True
    assert refreshed["count"] == 1
    assert submit_headers == [_auth_headers("fresh-token")]
    assert job["job_id"] == "new-job-id"
    assert job["retries"] == 1
