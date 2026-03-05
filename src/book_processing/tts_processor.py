"""Text-to-Speech using Azure Batch Synthesis REST API."""

import logging
import time
import uuid
from pathlib import Path

import httpx
from azure.identity import DefaultAzureCredential

from book_processing.config import (
    AUDIO_OUTPUT_FORMAT,
    AZURE_COGNITIVE_SCOPE,
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_API_VERSION,
    LANGUAGES,
    OUTPUT_DIR,
    SOURCE_TTS_NAME,
    SUMMARY_TYPES,
    output_audio_path,
    output_text_path,
)
from book_processing.ssml_builder import build_chunked_ssml

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 10
MAX_POLL_ATTEMPTS = 360  # 1 hour max


def _get_auth_headers() -> dict[str, str]:
    """Get authorization headers using Entra authentication."""
    credential = DefaultAzureCredential()
    token = credential.get_token(AZURE_COGNITIVE_SCOPE).token
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _submit_batch_synthesis(
    client: httpx.Client,
    headers: dict[str, str],
    ssml_inputs: list[str],
    display_name: str,
) -> str:
    """Submit a batch synthesis job and return the job ID.

    Args:
        client: HTTP client.
        headers: Auth headers.
        ssml_inputs: List of SSML strings to synthesize.
        display_name: Human-readable name for the job.

    Returns:
        The batch synthesis job ID.
    """
    job_id = str(uuid.uuid4())
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )

    inputs = [{"content": ssml} for ssml in ssml_inputs]

    body = {
        "displayName": display_name,
        "description": f"Book processing: {display_name}",
        "inputKind": "SSML",
        "inputs": inputs,
        "properties": {
            "outputFormat": AUDIO_OUTPUT_FORMAT,
            "concatenateResult": True,
        },
    }

    logger.info("Submitting batch synthesis job '%s' (id=%s, %d input(s))...", display_name, job_id, len(inputs))
    response = client.put(url, json=body, headers=headers)
    response.raise_for_status()
    logger.info("Job submitted successfully: %s", job_id)
    return job_id


def _poll_job(client: httpx.Client, headers: dict[str, str], job_id: str) -> dict:
    """Poll a batch synthesis job until completion.

    Returns:
        The final job status response as a dict.
    """
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )

    consecutive_errors = 0
    for attempt in range(MAX_POLL_ATTEMPTS):
        # Refresh token every 5 min (30 polls × 10s)
        if attempt > 0 and attempt % 30 == 0:
            try:
                credential = DefaultAzureCredential()
                token = credential.get_token(AZURE_COGNITIVE_SCOPE).token
                headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.warning("Token refresh failed: %s", e)

        try:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "Unknown")
            consecutive_errors = 0

            if status == "Succeeded":
                logger.info("Job %s completed successfully", job_id)
                return data
            elif status in ("Failed", "Cancelled"):
                error = data.get("properties", {}).get("error", "Unknown error")
                raise RuntimeError(f"Batch synthesis job {job_id} failed: {error}")
            else:
                logger.debug("Job %s status: %s (attempt %d/%d)", job_id, status, attempt + 1, MAX_POLL_ATTEMPTS)
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            consecutive_errors += 1
            logger.warning("Poll error (attempt %d, %d consecutive): %s", attempt + 1, consecutive_errors, e)
            if consecutive_errors >= 10:
                raise RuntimeError(f"Too many consecutive poll errors for job {job_id}") from e

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Job {job_id} did not complete within {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s")


def _download_result(client: httpx.Client, headers: dict[str, str], job_data: dict, output_path: Path) -> None:
    """Download the synthesized audio from a completed job."""
    outputs = job_data.get("outputs", {})
    result_url = outputs.get("result")
    if not result_url:
        raise ValueError(f"No result URL in job data: {job_data}")

    logger.info("Downloading audio to %s...", output_path)
    # SAS URL already contains auth in query string — don't send Bearer token
    response = client.get(result_url, follow_redirects=True)
    response.raise_for_status()

    # The result is a ZIP containing the audio file(s)
    import io
    import zipfile

    zip_data = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_data) as zf:
        # Find the audio file in the ZIP
        audio_files = [f for f in zf.namelist() if f.endswith(('.mp3', '.wav'))]
        if not audio_files:
            # If concatenateResult is true, there might be a single file
            audio_files = [f for f in zf.namelist() if not f.endswith('/')]

        if not audio_files:
            raise ValueError(f"No audio files found in result ZIP: {zf.namelist()}")

        # Use the first (or only) audio file
        with zf.open(audio_files[0]) as audio_f:
            output_path.write_bytes(audio_f.read())

    logger.info("Saved audio: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1024 / 1024)


def _delete_job(client: httpx.Client, headers: dict[str, str], job_id: str) -> None:
    """Delete a completed batch synthesis job to clean up."""
    url = (
        f"{AZURE_SPEECH_ENDPOINT}/texttospeech/batchsyntheses/{job_id}"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )
    try:
        client.delete(url, headers=headers)
        logger.debug("Deleted job %s", job_id)
    except Exception:
        logger.warning("Failed to delete job %s (non-critical)", job_id)


def synthesize_text(
    client: httpx.Client,
    headers: dict[str, str],
    text: str,
    lang: str,
    output_path: Path,
    is_podcast: bool = False,
    display_name: str = "",
) -> Path:
    """Synthesize a single text to audio via the Batch Synthesis API.

    Includes retry logic for transient network failures.

    Args:
        client: HTTP client.
        headers: Auth headers.
        text: Plain text or podcast script.
        lang: Language code ('en' or 'cs').
        output_path: Where to save the audio file.
        is_podcast: If True, use multi-voice podcast SSML.
        display_name: Human-readable name for the job.

    Returns:
        Path to the saved audio file.
    """
    ssml_chunks = build_chunked_ssml(text, lang, is_podcast=is_podcast)
    job_id = _submit_batch_synthesis(client, headers, ssml_chunks, display_name or output_path.stem)
    job_data = _poll_job(client, headers, job_id)
    _download_result(client, headers, job_data, output_path)
    _delete_job(client, headers, job_id)
    return output_path


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Run the full TTS stage — synthesize all text outputs to audio.

    Expects text files to already exist in output_dir from the LLM stage.

    Returns:
        Dictionary mapping output names to their audio file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = _get_auth_headers()
    outputs: dict[str, Path] = {}

    with httpx.Client(timeout=300) as client:
        # Synthesize summaries
        for summary_type, spec in SUMMARY_TYPES.items():
            for lang in LANGUAGES:
                text_path = output_text_path(summary_type, lang)
                if not text_path.exists():
                    logger.warning("Missing text file: %s — skipping", text_path)
                    continue

                text = text_path.read_text(encoding="utf-8")
                audio_path = output_audio_path(summary_type, lang)
                display = f"{summary_type}_{lang}"

                if audio_path.exists() and audio_path.stat().st_size > 1000:
                    logger.info("Skipping %s (audio already exists with %d bytes)", display, audio_path.stat().st_size)
                    outputs[display] = audio_path
                    continue

                # Refresh token before each job
                headers = _get_auth_headers()
                logger.info("Synthesizing %s...", display)
                synthesize_text(
                    client, headers, text, lang, audio_path,
                    is_podcast=spec["is_podcast"],
                    display_name=display,
                )
                outputs[display] = audio_path

        # Synthesize full TTS source
        for lang in LANGUAGES:
            text_path = output_text_path(SOURCE_TTS_NAME, lang)
            if not text_path.exists():
                logger.warning("Missing TTS source file: %s — skipping", text_path)
                continue

            text = text_path.read_text(encoding="utf-8")
            audio_path = output_audio_path(SOURCE_TTS_NAME, lang)
            display = f"{SOURCE_TTS_NAME}_{lang}"

            if audio_path.exists() and audio_path.stat().st_size > 1000:
                logger.info("Skipping %s (audio already exists with %d bytes)", display, audio_path.stat().st_size)
                outputs[display] = audio_path
                continue

            # Refresh token before each job
            headers = _get_auth_headers()
            logger.info("Synthesizing full source %s...", display)
            synthesize_text(
                client, headers, text, lang, audio_path,
                is_podcast=False,
                display_name=display,
            )
            outputs[display] = audio_path

    return outputs
