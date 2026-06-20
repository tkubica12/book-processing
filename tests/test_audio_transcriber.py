"""Tests for long-form audio speech-to-text helpers."""

import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from book_processing import audio_transcriber


def _chunk(index: int, partial_dir: Path) -> audio_transcriber._AudioChunk:
    stem = f"sample_audio_stt_chunk{index + 1:04d}"
    return audio_transcriber._AudioChunk(
        index=index,
        start_seconds=float(index * 10),
        duration_seconds=10.0,
        audio_path=partial_dir / f"{stem}.mp3",
        transcript_path=partial_dir / f"{stem}.md",
    )


def test_prepare_audio_chunks_reuses_existing_manifest(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "Sample.m4b"
    audio_path.write_bytes(b"source")
    partial_dir = tmp_path / "sample" / "_partial"
    export_calls: list[int] = []

    monkeypatch.setattr(audio_transcriber, "AUDIO_STT_CHUNK_DURATION_MINUTES", 30)
    monkeypatch.setattr(audio_transcriber, "_get_audio_duration_seconds", lambda _path: 3700.0)

    def fake_export(_source: Path, chunk: audio_transcriber._AudioChunk) -> None:
        export_calls.append(chunk.index)
        chunk.audio_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.audio_path.write_bytes(f"chunk-{chunk.index}".encode("utf-8"))

    monkeypatch.setattr(audio_transcriber, "_export_audio_chunk", fake_export)

    first_chunks = audio_transcriber._prepare_audio_chunks(audio_path, partial_dir, "sample")
    second_chunks = audio_transcriber._prepare_audio_chunks(audio_path, partial_dir, "sample")

    manifest = json.loads((partial_dir / "sample_audio_stt_manifest.json").read_text(encoding="utf-8"))

    assert [chunk.index for chunk in first_chunks] == [0, 1, 2]
    assert [chunk.audio_path.name for chunk in second_chunks] == [
        "sample_audio_stt_chunk0001.mp3",
        "sample_audio_stt_chunk0002.mp3",
        "sample_audio_stt_chunk0003.mp3",
    ]
    assert export_calls == [0, 1, 2]
    assert len(manifest["chunks"]) == 3


def test_convert_audio_to_markdown_reuses_cached_chunk_transcripts(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "Sample Audio.mp3"
    audio_path.write_bytes(b"source")
    partial_dir = tmp_path / "sample_audio" / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    chunks = [_chunk(0, partial_dir), _chunk(1, partial_dir), _chunk(2, partial_dir)]
    chunks[0].transcript_path.write_text("cached first transcript", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(audio_transcriber, "AUDIO_STT_MAX_CONCURRENT_CHUNKS", 2)
    monkeypatch.setattr(audio_transcriber, "_prepare_audio_chunks", lambda *_args, **_kwargs: chunks)

    def fake_transcribe(audio_file: Path) -> str:
        calls.append(audio_file.name)
        if audio_file.name.endswith("0002.mp3"):
            time.sleep(0.05)
            return "generated second"
        return "generated third"

    monkeypatch.setattr(audio_transcriber, "_transcribe_audio_chunk", fake_transcribe)

    markdown = audio_transcriber.convert_audio_to_markdown(audio_path, output_dir=tmp_path)

    assert markdown == "cached first transcript\n\ngenerated second\n\ngenerated third"
    assert sorted(calls) == ["sample_audio_stt_chunk0002.mp3", "sample_audio_stt_chunk0003.mp3"]
    assert chunks[1].transcript_path.read_text(encoding="utf-8") == "generated second"
    assert chunks[2].transcript_path.read_text(encoding="utf-8") == "generated third"


def test_convert_audio_to_markdown_retranscribes_trivial_cached_transcript(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "Sample Audio.mp3"
    audio_path.write_bytes(b"source")
    partial_dir = tmp_path / "sample_audio" / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    chunks = [_chunk(0, partial_dir), _chunk(1, partial_dir)]
    chunks[0].transcript_path.write_text("tiny", encoding="utf-8")
    chunks[1].transcript_path.write_text("cached transcript that is long enough", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(audio_transcriber, "AUDIO_STT_MIN_TRANSCRIPT_BYTES", 10)
    monkeypatch.setattr(audio_transcriber, "_prepare_audio_chunks", lambda *_args, **_kwargs: chunks)

    def fake_transcribe(audio_file: Path) -> str:
        calls.append(audio_file.name)
        return "regenerated transcript"

    monkeypatch.setattr(audio_transcriber, "_transcribe_audio_chunk", fake_transcribe)

    markdown = audio_transcriber.convert_audio_to_markdown(audio_path, output_dir=tmp_path)

    assert markdown == "regenerated transcript\n\ncached transcript that is long enough"
    assert calls == ["sample_audio_stt_chunk0001.mp3"]
    assert chunks[0].transcript_path.read_text(encoding="utf-8") == "regenerated transcript"


def test_get_audio_duration_seconds_falls_back_to_ffmpeg(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "Sample Audio.mp3"
    audio_path.write_bytes(b"source")

    def fail_mutagen(_path: Path):
        raise audio_transcriber.InvalidAudioSourceError("bad metadata")

    monkeypatch.setattr(audio_transcriber, "mutagen_file", fail_mutagen)
    monkeypatch.setattr(audio_transcriber.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(
        audio_transcriber.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="",
            stderr="Duration: 01:02:03.50, start: 0.000000, bitrate: 64 kb/s",
        ),
    )

    duration = audio_transcriber._get_audio_duration_seconds(audio_path)

    assert duration == pytest.approx(3723.5)


def test_export_audio_chunk_raises_invalid_audio_source_on_ffmpeg_failure(monkeypatch, tmp_path: Path):
    source_path = tmp_path / "source.mp3"
    source_path.write_bytes(b"source")
    chunk = audio_transcriber._AudioChunk(
        index=0,
        start_seconds=0.0,
        duration_seconds=10.0,
        audio_path=tmp_path / "chunk.mp3",
        transcript_path=tmp_path / "chunk.md",
    )

    monkeypatch.setattr(audio_transcriber.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(
        audio_transcriber.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="ffmpeg version 7\n[mp3float @ 1] Header missing\nError opening input files: Invalid data found when processing input",
        ),
    )

    with pytest.raises(audio_transcriber.InvalidAudioSourceError, match="Invalid data found when processing input"):
        audio_transcriber._export_audio_chunk(source_path, chunk)


def test_transcribe_audio_chunk_retries_transient_failures(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"chunk")
    request = httpx.Request("POST", "https://example.test/transcribe")
    responses: list[httpx.Response | Exception] = [
        httpx.Response(429, headers={"Retry-After": "7"}, request=request),
        httpx.ReadTimeout("timed out", request=request),
        httpx.Response(200, json={"combinedPhrases": [{"text": "hello world"}]}, request=request),
    ]
    post_headers: list[dict[str, str]] = []
    post_files: list[dict[str, tuple]] = []
    sleep_calls: list[float] = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url: str, *, headers, files):
            post_headers.append(headers)
            post_files.append(files)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(audio_transcriber.httpx, "Client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(audio_transcriber, "get_cognitive_token", lambda: "token")
    monkeypatch.setattr(audio_transcriber.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    transcript = audio_transcriber._transcribe_audio_chunk(audio_path)

    assert transcript == "hello world"
    assert sleep_calls == [7.0, 2.0]
    assert all(headers["Authorization"] == "Bearer token" for headers in post_headers)
    assert post_files[-1]["definition"][1] == '{"locales": []}'


def test_transcribe_audio_chunk_refreshes_token_once_on_unauthorized(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"chunk")
    request = httpx.Request("POST", "https://example.test/transcribe")
    responses: list[httpx.Response] = [
        httpx.Response(401, request=request),
        httpx.Response(200, json={"combinedPhrases": [{"text": "hello again"}]}, request=request),
    ]
    invalidations: list[str] = []
    sleep_calls: list[float] = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url: str, *, headers, files):
            return responses.pop(0)

    monkeypatch.setattr(audio_transcriber.httpx, "Client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(audio_transcriber, "get_cognitive_token", lambda: "token")
    monkeypatch.setattr(audio_transcriber, "invalidate_cognitive_token", lambda: invalidations.append("cleared"))
    monkeypatch.setattr(audio_transcriber.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    transcript = audio_transcriber._transcribe_audio_chunk(audio_path)

    assert transcript == "hello again"
    assert invalidations == ["cleared"]
    assert sleep_calls == [1.0]
