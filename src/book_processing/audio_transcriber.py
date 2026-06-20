"""Audio speech-to-text helpers for long-form audiobook and podcast sources."""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import imageio_ffmpeg
from mutagen import File as mutagen_file

from book_processing.auth import get_cognitive_token, invalidate_cognitive_token
from book_processing.config import (
    AUDIO_STT_CHUNK_DURATION_MINUTES,
    AUDIO_STT_EXPORT_BITRATE,
    AUDIO_STT_EXPORT_SAMPLE_RATE_HZ,
    AUDIO_STT_MAX_CONCURRENT_CHUNKS,
    AUDIO_STT_MIN_TRANSCRIPT_BYTES,
    AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS,
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION,
    AZURE_SPEECH_TRANSCRIPTION_LOCALES,
    AZURE_SPEECH_TRANSCRIPTION_MODEL,
    OUTPUT_DIR,
    book_name_from_source,
    book_output_dir,
)

logger = logging.getLogger(__name__)

_CHUNK_AUDIO_EXTENSION = ".mp3"
_MANIFEST_VERSION = 1
_RETRYABLE_STATUS_CODES = {408, 422, 429, 500, 502, 503, 504}
_REQUEST_TIMEOUT = httpx.Timeout(1800.0, connect=30.0)
_TRANSCRIPTION_DEFINITION = {"locales": AZURE_SPEECH_TRANSCRIPTION_LOCALES}
if AZURE_SPEECH_TRANSCRIPTION_MODEL:
    _TRANSCRIPTION_DEFINITION["enhancedMode"] = {
        "enabled": True,
        "model": AZURE_SPEECH_TRANSCRIPTION_MODEL,
        "transcribeStyle": "verbatim",
    }
_FFMPEG_DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_MIME_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".m4b": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".wma": "audio/x-ms-wma",
}


class InvalidAudioSourceError(RuntimeError):
    """Raised when an audio source cannot be decoded well enough for transcription."""


@dataclass(frozen=True, slots=True)
class _AudioChunk:
    """One locally prepared audio chunk and its recovery artifacts."""

    index: int
    start_seconds: float
    duration_seconds: float
    audio_path: Path
    transcript_path: Path


def convert_audio_to_markdown(
    audio_path: Path,
    output_dir: Path = OUTPUT_DIR,
    *,
    book_name: str | None = None,
    artifact_stem: str | None = None,
) -> str:
    """Chunk a long audio source, transcribe it with recovery, and return markdown text."""
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_path}")

    book_name = book_name or book_name_from_source(audio_path)
    artifact_stem = artifact_stem or book_name
    partial_dir = book_output_dir(book_name, output_dir) / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    chunks = _prepare_audio_chunks(audio_path, partial_dir, artifact_stem)
    transcripts: dict[int, str] = {}
    missing_chunks: list[_AudioChunk] = []

    for chunk in chunks:
        if _should_reuse_transcript(chunk.transcript_path):
            logger.info(
                "Reusing cached transcript chunk %d/%d for %s",
                chunk.index + 1,
                len(chunks),
                audio_path.name,
            )
            transcripts[chunk.index] = chunk.transcript_path.read_text(encoding="utf-8")
        else:
            missing_chunks.append(chunk)

    if missing_chunks:
        max_workers = min(len(missing_chunks), max(1, AUDIO_STT_MAX_CONCURRENT_CHUNKS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_transcribe_and_persist_chunk, chunk, len(chunks)): chunk
                for chunk in missing_chunks
            }
            for future in as_completed(futures):
                chunk = futures[future]
                transcripts[chunk.index] = future.result()

    ordered_chunks = [transcripts[index] for index in range(len(chunks))]
    combined_markdown = _assemble_markdown(ordered_chunks)
    combined_path = partial_dir / f"{artifact_stem}_audio_stt_assembled.md"
    _write_text_atomic(combined_path, combined_markdown)
    return combined_markdown


def _prepare_audio_chunks(audio_path: Path, partial_dir: Path, artifact_stem: str) -> list[_AudioChunk]:
    """Prepare chunk files and the manifest for one source audio file."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = partial_dir / f"{artifact_stem}_audio_stt_manifest.json"
    source_signature = _source_signature(audio_path)
    settings = _chunk_settings()
    manifest = _load_manifest(manifest_path)

    if manifest is None or not _manifest_matches(manifest, source_signature, settings):
        if manifest is not None:
            logger.info("Audio source changed for %s; rebuilding STT chunk artifacts.", audio_path.name)
            _cleanup_chunk_artifacts(partial_dir, artifact_stem)
        chunks = _plan_audio_chunks(audio_path, partial_dir, artifact_stem)
        _write_manifest(manifest_path, source_signature, settings, chunks)
    else:
        chunks = [_chunk_from_manifest_entry(partial_dir, entry) for entry in manifest["chunks"]]

    for chunk in chunks:
        if chunk.audio_path.exists() and chunk.audio_path.stat().st_size > 0:
            continue
        _export_audio_chunk(audio_path, chunk)

    return chunks


def _plan_audio_chunks(audio_path: Path, partial_dir: Path, artifact_stem: str) -> list[_AudioChunk]:
    """Plan chunk boundaries for the source audio."""
    total_duration_seconds = _get_audio_duration_seconds(audio_path)
    chunk_duration_seconds = max(60, AUDIO_STT_CHUNK_DURATION_MINUTES * 60)
    total_chunks = max(1, math.ceil(total_duration_seconds / chunk_duration_seconds))
    chunks: list[_AudioChunk] = []

    for index in range(total_chunks):
        start_seconds = float(index * chunk_duration_seconds)
        duration_seconds = min(chunk_duration_seconds, total_duration_seconds - start_seconds)
        chunk_id = f"{index + 1:04d}"
        stem = f"{artifact_stem}_audio_stt_chunk{chunk_id}"
        chunks.append(
            _AudioChunk(
                index=index,
                start_seconds=round(start_seconds, 3),
                duration_seconds=round(duration_seconds, 3),
                audio_path=partial_dir / f"{stem}{_CHUNK_AUDIO_EXTENSION}",
                transcript_path=partial_dir / f"{stem}.md",
            )
        )

    return chunks


def _get_audio_duration_seconds(audio_path: Path) -> float:
    """Read the source audio duration from file metadata."""
    if audio_path.stat().st_size <= 0:
        raise InvalidAudioSourceError(f"Audio source is empty: {audio_path}")

    mutagen_error: Exception | None = None
    try:
        audio_file = mutagen_file(audio_path)
        length = getattr(getattr(audio_file, "info", None), "length", 0.0)
        if length and length > 0:
            return float(length)
    except Exception as error:
        mutagen_error = error

    ffmpeg_error: Exception | None = None
    try:
        return _get_audio_duration_seconds_with_ffmpeg(audio_path)
    except Exception as error:
        ffmpeg_error = error

    details: list[str] = []
    if mutagen_error is not None:
        details.append(f"mutagen: {mutagen_error}")
    else:
        details.append("mutagen: no usable duration metadata")
    if ffmpeg_error is not None:
        details.append(f"ffmpeg: {ffmpeg_error}")
    raise InvalidAudioSourceError(
        f"Could not determine audio duration for {audio_path} ({'; '.join(details)})"
    )


def _get_audio_duration_seconds_with_ffmpeg(audio_path: Path) -> float:
    """Read the source audio duration by parsing FFmpeg probe output."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_exe, "-i", str(audio_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = _FFMPEG_DURATION_PATTERN.search(result.stderr)
    if match is None:
        diagnostic = _summarize_ffmpeg_diagnostic(result.stderr, result.stdout, result.returncode)
        raise RuntimeError(diagnostic)

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    duration_seconds = float(hours * 3600 + minutes * 60) + seconds
    if duration_seconds <= 0:
        raise RuntimeError(f"FFmpeg reported a non-positive duration for {audio_path}")
    return duration_seconds


def _export_audio_chunk(source_path: Path, chunk: _AudioChunk) -> None:
    """Export one chunk to a speech-friendly MP3 file using bundled FFmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(
        "Exporting audio chunk %d to %s (start=%.1fs duration=%.1fs)",
        chunk.index + 1,
        chunk.audio_path.name,
        chunk.start_seconds,
        chunk.duration_seconds,
    )
    temp_path = chunk.audio_path.with_name(f"{chunk.audio_path.stem}.tmp{chunk.audio_path.suffix}")
    command = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{chunk.start_seconds:.3f}",
        "-t",
        f"{chunk.duration_seconds:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_STT_EXPORT_SAMPLE_RATE_HZ),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        AUDIO_STT_EXPORT_BITRATE,
        "-f",
        "mp3",
        str(temp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise InvalidAudioSourceError(
            f"FFmpeg failed while exporting {chunk.audio_path.name}: "
            f"{_summarize_ffmpeg_diagnostic(result.stderr, result.stdout, result.returncode)}"
        )
    temp_path.replace(chunk.audio_path)


_UNPROCESSABLE_CHUNK_PLACEHOLDER = "[audio segment could not be transcribed]"


def _transcribe_and_persist_chunk(chunk: _AudioChunk, total_chunks: int) -> str:
    """Transcribe one chunk and atomically persist its transcript."""
    logger.info("Transcribing chunk %d/%d: %s", chunk.index + 1, total_chunks, chunk.audio_path.name)
    try:
        transcript = _transcribe_audio_chunk(chunk.audio_path)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 422:
            logger.warning(
                "Chunk %d/%d (%s) could not be transcribed after max retries (HTTP 422); "
                "persisting placeholder and continuing.",
                chunk.index + 1,
                total_chunks,
                chunk.audio_path.name,
            )
            transcript = _UNPROCESSABLE_CHUNK_PLACEHOLDER
        else:
            raise
    _write_text_atomic(chunk.transcript_path, transcript)
    return transcript


_MAX_RETRIES_FOR_422 = 3


def _transcribe_audio_chunk(audio_path: Path) -> str:
    """Call Azure Speech fast transcription with unbounded retry on transient failures."""
    url = (
        f"{AZURE_SPEECH_ENDPOINT.rstrip('/')}/speechtotext/transcriptions:transcribe"
        f"?api-version={AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION}"
    )

    attempt = 0
    attempts_at_422 = 0
    refreshed_after_401 = False
    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        while True:
            attempt += 1
            try:
                with audio_path.open("rb") as audio_file:
                    response = client.post(
                        url,
                        headers={"Authorization": f"Bearer {get_cognitive_token()}"},
                        files={
                            "audio": (
                                audio_path.name,
                                audio_file,
                                _AUDIO_MIME_TYPES.get(audio_path.suffix.lower(), "application/octet-stream"),
                            ),
                            "definition": (
                                None,
                                json.dumps(_TRANSCRIPTION_DEFINITION),
                                "application/json",
                            ),
                        },
                    )
            except httpx.RequestError as error:
                wait_seconds = _retry_delay_seconds(attempt)
                logger.warning(
                    "Fast transcription request failed for %s (attempt %d): %s. Retrying in %.1fs...",
                    audio_path.name,
                    attempt,
                    error,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 401 and not refreshed_after_401:
                refreshed_after_401 = True
                invalidate_cognitive_token()
                wait_seconds = _retry_delay_seconds(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "Fast transcription returned HTTP 401 for %s (attempt %d). "
                    "Refreshing token and retrying in %.1fs...",
                    audio_path.name,
                    attempt,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if response.status_code == 422:
                    attempts_at_422 += 1
                    if attempts_at_422 > _MAX_RETRIES_FOR_422:
                        logger.error(
                            "Fast transcription returned HTTP 422 for %s %d times; giving up.",
                            audio_path.name,
                            attempts_at_422,
                        )
                        response.raise_for_status()
                wait_seconds = _retry_delay_seconds(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "Fast transcription returned HTTP %d for %s (attempt %d). Retrying in %.1fs...",
                    response.status_code,
                    audio_path.name,
                    attempt,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return _extract_transcript_text(response.json(), audio_path)


def _extract_transcript_text(payload: dict, audio_path: Path) -> str:
    """Extract the best available transcript text from the API response."""
    combined_phrases = payload.get("combinedPhrases", [])
    combined_text = [item.get("text", "").strip() for item in combined_phrases if item.get("text")]
    if combined_text:
        return "\n\n".join(combined_text)

    phrases = payload.get("phrases", [])
    phrase_text = [item.get("text", "").strip() for item in phrases if item.get("text")]
    if phrase_text:
        return "\n".join(phrase_text)

    raise RuntimeError(f"Azure Speech returned no usable transcription text for {audio_path.name}")


def _assemble_markdown(chunks: list[str]) -> str:
    """Combine chunk transcripts into deterministic markdown text."""
    return "\n\n".join(text.strip() for text in chunks if text.strip()).strip()


def _should_reuse_transcript(transcript_path: Path) -> bool:
    """Return whether a transcript checkpoint is substantial enough to reuse."""
    return transcript_path.exists() and transcript_path.stat().st_size >= AUDIO_STT_MIN_TRANSCRIPT_BYTES


def _retry_delay_seconds(attempt: int, retry_after_header: str | None = None) -> float:
    """Compute a capped exponential retry delay, honoring Retry-After when present."""
    delay = min(float(AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS), float(2 ** max(attempt - 1, 0)))
    retry_after_seconds = _retry_after_seconds(retry_after_header)
    if retry_after_seconds is not None:
        delay = max(delay, retry_after_seconds)
    return delay


def _retry_after_seconds(retry_after_header: str | None) -> float | None:
    """Parse the Retry-After header value when it is a delta-seconds integer."""
    if not retry_after_header:
        return None
    try:
        return max(0.0, float(retry_after_header))
    except ValueError:
        return None


def _summarize_ffmpeg_diagnostic(stderr: str, stdout: str, returncode: int) -> str:
    """Collapse noisy FFmpeg output into a concise diagnostic message."""
    combined_output = "\n".join(part for part in (stderr, stdout) if part).strip()
    if not combined_output:
        return f"exit code {returncode}"

    filtered_lines = [
        line.strip()
        for line in combined_output.splitlines()
        if line.strip()
        and not line.startswith("ffmpeg version")
        and not line.startswith("  built with")
        and not line.startswith("  configuration:")
        and not line.startswith("  lib")
    ]
    if not filtered_lines:
        return f"exit code {returncode}"
    return " | ".join(filtered_lines[-8:])


def _source_signature(audio_path: Path) -> dict[str, int | str]:
    """Create a cache key for one source audio file."""
    stat = audio_path.stat()
    return {
        "name": audio_path.name,
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }


def _chunk_settings() -> dict[str, int | str]:
    """Return the chunking settings that affect cached artifacts."""
    return {
        "chunk_duration_minutes": AUDIO_STT_CHUNK_DURATION_MINUTES,
        "export_bitrate": AUDIO_STT_EXPORT_BITRATE,
        "sample_rate_hz": AUDIO_STT_EXPORT_SAMPLE_RATE_HZ,
        "manifest_version": _MANIFEST_VERSION,
        "transcription_locales": ",".join(AZURE_SPEECH_TRANSCRIPTION_LOCALES),
        "transcription_model": AZURE_SPEECH_TRANSCRIPTION_MODEL,
    }


def _manifest_matches(
    manifest: dict,
    source_signature: dict[str, int | str],
    settings: dict[str, int | str],
) -> bool:
    """Return whether a manifest is reusable for the current source and settings."""
    return (
        manifest.get("version") == _MANIFEST_VERSION
        and manifest.get("source") == source_signature
        and manifest.get("settings") == settings
    )


def _load_manifest(manifest_path: Path) -> dict | None:
    """Load a manifest if it exists and is valid JSON."""
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid STT manifest at %s", manifest_path)
        manifest_path.unlink(missing_ok=True)
        return None


def _write_manifest(
    manifest_path: Path,
    source_signature: dict[str, int | str],
    settings: dict[str, int | str],
    chunks: list[_AudioChunk],
) -> None:
    """Persist the chunk manifest for crash recovery."""
    payload = {
        "version": _MANIFEST_VERSION,
        "source": source_signature,
        "settings": settings,
        "chunks": [
            {
                "index": chunk.index,
                "start_seconds": chunk.start_seconds,
                "duration_seconds": chunk.duration_seconds,
                "audio_file": chunk.audio_path.name,
                "transcript_file": chunk.transcript_path.name,
            }
            for chunk in chunks
        ],
    }
    _write_json_atomic(manifest_path, payload)


def _chunk_from_manifest_entry(partial_dir: Path, entry: dict) -> _AudioChunk:
    """Rebuild a chunk object from a manifest entry."""
    return _AudioChunk(
        index=int(entry["index"]),
        start_seconds=float(entry["start_seconds"]),
        duration_seconds=float(entry["duration_seconds"]),
        audio_path=partial_dir / entry["audio_file"],
        transcript_path=partial_dir / entry["transcript_file"],
    )


def _cleanup_chunk_artifacts(partial_dir: Path, artifact_stem: str) -> None:
    """Remove stale chunk artifacts when the source audio changes."""
    for artifact in partial_dir.glob(f"{artifact_stem}_audio_stt_chunk*"):
        artifact.unlink(missing_ok=True)
    (partial_dir / f"{artifact_stem}_audio_stt_manifest.json").unlink(missing_ok=True)
    (partial_dir / f"{artifact_stem}_audio_stt_assembled.md").unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically write a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)
