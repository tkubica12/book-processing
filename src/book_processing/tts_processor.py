"""Text-to-Speech using Azure Batch Synthesis REST API with parallel job execution.

Key optimization: large files (source_tts) are split into independent parallel
jobs per SSML chunk.  Each chunk synthesizes independently on Azure, then
results are concatenated.  This reduces wall-clock time from 30-60 min to
5-10 min for a full-book audio file.
"""

import io
import logging
import threading
import time
import uuid
import zipfile
from pathlib import Path
from queue import Empty, Queue

import httpx
from mutagen.id3 import COMM, ID3, TALB, TCON, TIT2, TLAN, TPE1

from book_processing.auth import get_cognitive_token
from book_processing.config import (
    AUDIO_OUTPUT_FORMAT,
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_API_VERSION,
    LANGUAGES,
    OUTPUT_DIR,
    SOURCE_TTS_NAME,
    SUMMARY_TYPES,
    TTS_JOB_MAX_RETRIES,
    TTS_MAX_CONCURRENT_JOBS,
    output_text_path,
    output_audio_path,
)
from book_processing.ssml_builder import build_chunked_ssml

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 10
CONTENT_TITLE_LABELS = {
    "summary_2min": "Summary 2 min",
    "summary_5min": "Summary 5 min",
    "summary_20min": "Summary 20 min",
    "podcast_60min": "Podcast 60 min",
    SOURCE_TTS_NAME: "Source TTS",
}


def _get_token() -> str:
    """Get a cached access token for Azure Speech."""
    for attempt in range(5):
        try:
            return get_cognitive_token()
        except Exception as e:
            wait = 10 * (attempt + 1)
            if attempt < 4:
                logger.warning("Token refresh failed (attempt %d): %s. Retrying in %ds...",
                               attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _display_book_title(book_name: str) -> str:
    """Convert a sanitized book identifier into a readable title."""
    return book_name.replace("_", " ").strip().title()


def _build_audio_metadata(book_name: str, name: str, lang: str) -> dict[str, str]:
    """Build user-facing MP3 metadata values for one output artifact."""
    book_title = _display_book_title(book_name)
    content_label = CONTENT_TITLE_LABELS.get(name, name.replace("_", " ").title())
    return {
        "title": f"{book_title} - {content_label} ({lang})",
        "album": book_title,
        "artist": "book-processing",
        "language": lang,
        "comment": f"{book_name}_{name}_{lang}",
        "genre": "Speech",
    }


def _write_mp3_metadata(audio_path: Path, book_name: str, name: str, lang: str) -> None:
    """Write ID3 metadata to a synthesized MP3 file."""
    metadata = _build_audio_metadata(book_name, name, lang)
    tags = ID3()
    tags.add(TIT2(encoding=3, text=metadata["title"]))
    tags.add(TALB(encoding=3, text=metadata["album"]))
    tags.add(TPE1(encoding=3, text=metadata["artist"]))
    tags.add(TLAN(encoding=3, text=[metadata["language"]]))
    tags.add(TCON(encoding=3, text=metadata["genre"]))
    tags.add(COMM(encoding=3, lang="eng", desc="source", text=metadata["comment"]))
    tags.save(audio_path, v2_version=3)


def _submit_batch_synthesis(
    client: httpx.Client,
    headers: dict[str, str],
    ssml_inputs: list[str],
    display_name: str,
) -> str:
    """Submit a batch synthesis job and return the job ID."""
    job_id = str(uuid.uuid4())
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )
    body = {
        "displayName": display_name,
        "description": f"Book processing: {display_name}",
        "inputKind": "SSML",
        "inputs": [{"content": ssml} for ssml in ssml_inputs],
        "properties": {
            "outputFormat": AUDIO_OUTPUT_FORMAT,
            "concatenateResult": True,
        },
    }

    logger.info("Submitting TTS job '%s' (id=%s, %d input(s))...", display_name, job_id, len(ssml_inputs))
    for attempt in range(5):
        try:
            response = client.put(url, json=body, headers=headers)
            response.raise_for_status()
            logger.info("Job submitted: %s", job_id)
            return job_id
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning("Submit failed (attempt %d): %s. Retrying...", attempt + 1, e)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Failed to submit TTS job '{display_name}' after 5 attempts")


def _check_job_status(client: httpx.Client, headers: dict[str, str], job_id: str) -> dict | None:
    """Check status. Returns job data if Succeeded, None if still running, raises on failure."""
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )
    try:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "Unknown")
        if status == "Succeeded":
            return data
        if status in ("Failed", "Cancelled"):
            error = data.get("properties", {}).get("error", "Unknown error")
            raise RuntimeError(f"Batch synthesis job {job_id} failed: {error}")
        return None
    except (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        OSError,
    ) as e:
        logger.warning("Poll error for %s: %s", job_id, e)
        return None


def _download_audio_bytes(client: httpx.Client, job_data: dict) -> bytes:
    """Download synthesized audio bytes from a completed job (SAS URL, no auth header)."""
    result_url = job_data.get("outputs", {}).get("result")
    if not result_url:
        raise ValueError(f"No result URL in job data: {job_data}")

    for attempt in range(3):
        try:
            response = client.get(result_url, follow_redirects=True, timeout=600)
            response.raise_for_status()
            break
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.ReadError,
        ) as e:
            logger.warning("Download failed (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(15)

    zip_data = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_data) as zf:
        audio_files = [f for f in zf.namelist() if f.endswith(('.mp3', '.wav'))]
        if not audio_files:
            audio_files = [f for f in zf.namelist() if not f.endswith('/')]
        if not audio_files:
            raise ValueError(f"No audio files in result ZIP: {zf.namelist()}")
        with zf.open(audio_files[0]) as af:
            return af.read()


def _delete_job(client: httpx.Client, headers: dict[str, str], job_id: str) -> None:
    """Delete a completed job to clean up."""
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )
    try:
        client.delete(url, headers=headers)
        logger.debug("Deleted job %s", job_id)
    except Exception:
        logger.warning("Failed to delete job %s (non-critical)", job_id)


# ---------------------------------------------------------------------------
# TtsJobTracker - queue-driven, thread-safe TTS manager
# ---------------------------------------------------------------------------

class TtsJobTracker:
    """Manages TTS batch synthesis jobs with a queue-driven, poll-based loop.

    Large files (multiple SSML chunks) are exploded into independent parallel
    jobs.  Each chunk is synthesized separately, then the MP3 results are
    concatenated in order once all chunks for that file complete.

    Usage::

        tracker = TtsJobTracker()
        Thread(target=tracker.poll_loop).start()
        tracker.enqueue("book_name", "summary_2min", "en", path, is_podcast=False)
        tracker.finalize()
        tracker.wait()
        outputs = tracker.get_outputs()
    """

    def __init__(self) -> None:
        self._pending: Queue[dict | None] = Queue()
        self._active: list[dict] = []      # individual chunk jobs
        self._completed: dict[str, Path] = {}
        self._finalized = threading.Event()
        self._done = threading.Event()
        self._error: Exception | None = None
        # Multi-chunk assembly: display -> {total, parts: {idx: bytes}}
        self._assembly: dict[str, dict] = {}
        self._assembly_lock = threading.Lock()

    def enqueue(self, book_name: str, name: str, lang: str, text_path: Path, is_podcast: bool) -> None:
        """Thread-safe: add a new TTS request to the processing queue."""
        self._pending.put({
            "book_name": book_name, "name": name, "lang": lang, "text_path": text_path, "is_podcast": is_podcast,
        })

    def finalize(self) -> None:
        """Signal that no more jobs will be enqueued."""
        self._finalized.set()

    def wait(self) -> None:
        """Block until all jobs are done. Re-raises any error from the poll loop."""
        self._done.wait()
        if self._error:
            raise self._error

    def get_outputs(self) -> dict[str, Path]:
        return dict(self._completed)

    def poll_loop(self) -> None:
        """Main processing loop: dequeue, submit, poll, download."""
        client = httpx.Client(timeout=300)
        try:
            while True:
                # Drain pending queue, submit up to concurrency limit
                while len(self._active) < TTS_MAX_CONCURRENT_JOBS:
                    try:
                        item = self._pending.get_nowait()
                    except Empty:
                        break
                    if item is None:
                        continue
                    self._submit_item(client, item)

                # Poll active jobs
                if self._active:
                    token = _get_token()
                    headers = _auth_headers(token)
                    completed = []
                    for job in self._active:
                        try:
                            data = _check_job_status(client, headers, job["job_id"])
                        except RuntimeError as exc:
                            if self._retry_failed_job(client, headers, job, exc):
                                continue
                            raise
                        if data is not None:
                            self._handle_completed_job(client, headers, job, data)
                            completed.append(job)
                    for job in completed:
                        self._active.remove(job)

                # Check exit condition
                if self._finalized.is_set() and self._pending.empty() and not self._active:
                    break

                time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error("TTS poll loop error: %s", e)
            self._error = e
        finally:
            client.close()
            self._done.set()

    def _handle_completed_job(
        self, client: httpx.Client, headers: dict[str, str],
        job: dict, job_data: dict,
    ) -> None:
        """Process a completed job - either save directly or assemble chunks."""
        display = job["display"]
        audio_bytes = _download_audio_bytes(client, job_data)
        _delete_job(client, headers, job["job_id"])

        parent = job.get("parent_display")
        if parent is None:
            # Single-job file: write directly
            job["audio_path"].write_bytes(audio_bytes)
            _write_mp3_metadata(job["audio_path"], job["book_name"], job["name"], job["lang"])
            self._completed[display] = job["audio_path"]
            logger.info("TTS done: %s (%.1f MB)", display,
                        job["audio_path"].stat().st_size / 1024 / 1024)
        else:
            # Multi-chunk file: collect this chunk
            chunk_idx = job["chunk_idx"]
            logger.info("TTS chunk done: %s chunk %d (%.1f MB)",
                        parent, chunk_idx + 1, len(audio_bytes) / 1024 / 1024)
            with self._assembly_lock:
                entry = self._assembly[parent]
                entry["parts"][chunk_idx] = audio_bytes
                done_count = len(entry["parts"])
                total = entry["total"]

            if done_count == total:
                self._assemble_chunks(parent)

    def _assemble_chunks(self, display: str) -> None:
        """Concatenate all chunk MP3s into the final output file."""
        with self._assembly_lock:
            entry = self._assembly[display]
        audio_path = entry["audio_path"]

        logger.info("Assembling %d chunks for %s...", entry["total"], display)
        with open(audio_path, "wb") as f:
            for idx in range(entry["total"]):
                f.write(entry["parts"][idx])

        _write_mp3_metadata(audio_path, entry["book_name"], entry["name"], entry["lang"])

        size_mb = audio_path.stat().st_size / 1024 / 1024
        self._completed[display] = audio_path
        logger.info("TTS assembled: %s (%.1f MB from %d chunks)",
                     display, size_mb, entry["total"])

        # Free memory
        with self._assembly_lock:
            del self._assembly[display]

    def _retry_failed_job(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        job: dict,
        error: RuntimeError,
    ) -> bool:
        """Retry a failed Azure batch job in place when retry budget remains."""
        retries = job.get("retries", 0)
        display = job["display"]
        if retries >= TTS_JOB_MAX_RETRIES:
            logger.error("TTS job %s exhausted retries: %s", display, error)
            return False

        logger.warning(
            "Retrying failed TTS job %s (%d/%d): %s",
            display,
            retries + 1,
            TTS_JOB_MAX_RETRIES,
            error,
        )
        _delete_job(client, headers, job["job_id"])
        time.sleep(5 * (retries + 1))
        job["job_id"] = _submit_batch_synthesis(
            client,
            headers,
            job["ssml_inputs"],
            display,
        )
        job["retries"] = retries + 1
        return True

    def _submit_item(self, client: httpx.Client, item: dict) -> None:
        """Prepare SSML and submit job(s) for one text file.

        For files with multiple SSML chunks, each chunk becomes an independent
        parallel job for maximum throughput.
        """
        book_name, name, lang = item["book_name"], item["name"], item["lang"]
        audio_path = output_audio_path(book_name, name, lang)
        display = f"{book_name}_{name}_{lang}"

        if audio_path.exists() and audio_path.stat().st_size > 1000:
            logger.info("TTS already exists: %s", display)
            self._completed[display] = audio_path
            return

        text = item["text_path"].read_text(encoding="utf-8")
        ssml_chunks = build_chunked_ssml(text, lang, is_podcast=item["is_podcast"])
        token = _get_token()
        headers = _auth_headers(token)

        if len(ssml_chunks) == 1:
            # Small file: single job
            job_id = _submit_batch_synthesis(client, headers, ssml_chunks, display)
            self._active.append({
                "job_id": job_id,
                "display": display,
                "audio_path": audio_path,
                "book_name": book_name,
                "name": name,
                "lang": lang,
                "ssml_inputs": ssml_chunks,
                "retries": 0,
            })
        else:
            # Large file: one independent job per chunk for parallel synthesis
            logger.info("Splitting %s into %d parallel TTS jobs", display, len(ssml_chunks))
            with self._assembly_lock:
                self._assembly[display] = {
                    "total": len(ssml_chunks),
                    "parts": {},
                    "audio_path": audio_path,
                    "book_name": book_name,
                    "name": name,
                    "lang": lang,
                }
            for idx, ssml in enumerate(ssml_chunks):
                chunk_display = f"{display}_chunk{idx + 1}"
                job_id = _submit_batch_synthesis(client, headers, [ssml], chunk_display)
                self._active.append({
                    "job_id": job_id,
                    "display": chunk_display,
                    "audio_path": audio_path,
                    "book_name": book_name,
                    "name": name,
                    "lang": lang,
                    "parent_display": display,
                    "chunk_idx": idx,
                    "ssml_inputs": [ssml],
                    "retries": 0,
                })


# ---------------------------------------------------------------------------
# Standalone run() - scans output dir and processes all text files
# ---------------------------------------------------------------------------

def run(book_names: list[str], output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Process all available text files into audio (standalone mode)."""
    tracker = TtsJobTracker()

    poll_thread = threading.Thread(target=tracker.poll_loop, daemon=True)
    poll_thread.start()

    for book_name in book_names:
        for summary_type, spec in SUMMARY_TYPES.items():
            for lang in LANGUAGES:
                text_path = output_text_path(book_name, summary_type, lang)
                if text_path.exists():
                    tracker.enqueue(book_name, summary_type, lang, text_path, spec["is_podcast"])

        for lang in LANGUAGES:
            text_path = output_text_path(book_name, SOURCE_TTS_NAME, lang)
            if text_path.exists():
                tracker.enqueue(book_name, SOURCE_TTS_NAME, lang, text_path, False)

    tracker.finalize()
    tracker.wait()
    poll_thread.join()

    return tracker.get_outputs()
