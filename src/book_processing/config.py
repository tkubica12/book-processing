"""Centralized configuration for the book processing pipeline."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# === Project Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    """Return a trimmed environment override or the provided default."""
    return os.getenv(name, default).strip()


def _csv_env(name: str, default: str = "") -> list[str]:
    """Return a comma-separated environment override as a list of trimmed values."""
    return [value.strip() for value in _env(name, default).split(",") if value.strip()]


# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT = _env(
    "AZURE_OPENAI_ENDPOINT",
    "https://tomaskubica-foundry-resource.cognitiveservices.azure.com",
)
AZURE_OPENAI_MODEL = _env("AZURE_OPENAI_MODEL", "gpt-5.4")
AZURE_OPENAI_API_VERSION = "2025-04-01-preview"

# === Azure Speech (Batch Synthesis API) ===
AZURE_SPEECH_ENDPOINT = _env("AZURE_SPEECH_ENDPOINT", AZURE_OPENAI_ENDPOINT)
AZURE_SPEECH_API_VERSION = "2024-04-01"
AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION = _env("AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION", "2025-10-15")
AZURE_SPEECH_TRANSCRIPTION_MODEL = _env("AZURE_SPEECH_TRANSCRIPTION_MODEL")
AZURE_SPEECH_TRANSCRIPTION_LOCALES = _csv_env("AZURE_SPEECH_TRANSCRIPTION_LOCALES")
AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# === Azure Content Understanding ===
CONTENT_UNDERSTANDING_ENDPOINT = _env("CONTENT_UNDERSTANDING_ENDPOINT")
CONTENT_UNDERSTANDING_API_VERSION = _env("CONTENT_UNDERSTANDING_API_VERSION", "2025-11-01")
CONTENT_UNDERSTANDING_ANALYZER_ID = _env("CONTENT_UNDERSTANDING_ANALYZER_ID", "prebuilt-documentSearch")
CONTENT_UNDERSTANDING_API_KEY = _env("CONTENT_UNDERSTANDING_API_KEY")
CONTENT_UNDERSTANDING_PROCESSING_LOCATION = _env("CONTENT_UNDERSTANDING_PROCESSING_LOCATION")
CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS = float(
    os.getenv("CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS", "2")
)

# === Voice Configuration ===
# English output uses the latest MAI Voice model; Czech stays on proven multilingual HD voices.
VOICE_MALE = _env("VOICE_MALE", "en-US-Ethan:MAI-Voice-2")
VOICE_FEMALE = _env("VOICE_FEMALE", "en-US-Harper:MAI-Voice-2")
CZECH_VOICE_MALE = _env("CZECH_VOICE_MALE", "en-US-Andrew:DragonHDLatestNeural")
CZECH_VOICE_FEMALE = _env("CZECH_VOICE_FEMALE", "en-US-Emma:DragonHDLatestNeural")
VOICE_BY_LANGUAGE = {
    "en": {"male": VOICE_MALE, "female": VOICE_FEMALE},
    "cs": {"male": CZECH_VOICE_MALE, "female": CZECH_VOICE_FEMALE},
}

# === SSML / Prosody ===
PROSODY_RATE = "+20%"
AUDIO_OUTPUT_FORMAT = "audio-48khz-192kbitrate-mono-mp3"

# === Language Configuration ===
LANGUAGES = {
    "en": {"xml_lang": "en-US", "label": "English"},
    "cs": {"xml_lang": "cs-CZ", "label": "Czech"},
}

# === Podcast Speaker Tags ===
PODCAST_SPEAKERS = {
    "en": {"male": "Andrew", "female": "Emma"},
    "cs": {"male": "Tomáš", "female": "Kateřina"},
}

# === Word Count Targets ===
# Calibrated from measured MP3 durations:
#   EN at +30%: ~181 WPM → base ~139 → at +20%: ~167 WPM
#   CS at +30%: ~167 WPM → base ~128 → at +20%: ~154 WPM
#   Using 160 WPM as unified target (slightly favors shorter output for safety)
WPM_AT_TARGET_SPEED = 160

SUMMARY_TYPES = {
    "summary_5min": {
        "duration_min": 5,
        "target_words": 5 * WPM_AT_TARGET_SPEED,  # ~800
        "description": "short summary capturing the core ideas of the book",
        "is_podcast": False,
    },
    "summary_20min": {
        "duration_min": 20,
        "target_words": 20 * WPM_AT_TARGET_SPEED,  # ~3200
        "description": "in-depth condensed version of the book",
        "is_podcast": False,
    },
    "podcast_20min": {
        "duration_min": 20,
        "target_words": 20 * WPM_AT_TARGET_SPEED,  # ~3200
        "description": "Two-host conversational podcast covering the core of the book",
        "is_podcast": True,
    },
    "podcast_60min": {
        "duration_min": 60,
        "target_words": 60 * WPM_AT_TARGET_SPEED,  # ~9600
        "description": "Two-host in-depth conversational podcast covering the book in detail",
        "is_podcast": True,
    },
}

# === Parallelism ===
BOOK_MAX_WORKERS = 2  # Concurrent books processed in parallel batches
LLM_MAX_WORKERS = 6  # Concurrent LLM requests
TTS_MAX_CONCURRENT_JOBS = 8  # Concurrent Batch Synthesis jobs
TTS_JOB_MAX_RETRIES = 5  # Retries per submitted batch job before failing
TTS_JOB_STALE_AFTER_SECONDS = int(
    os.getenv("TTS_JOB_STALE_AFTER_SECONDS", "3600")
)  # Retry batch jobs that stay running for too long (1 hour default)
TTS_MAX_CHARS_PER_CHUNK = int(
    os.getenv("TTS_MAX_CHARS_PER_CHUNK", "25000")
)  # Smaller chunks are more reliable for long-form TTS
AUDIO_STT_CHUNK_DURATION_MINUTES = int(
    os.getenv("AUDIO_STT_CHUNK_DURATION_MINUTES", "30")
)
AUDIO_STT_MAX_CONCURRENT_CHUNKS = int(
    os.getenv("AUDIO_STT_MAX_CONCURRENT_CHUNKS", "4")
)
AUDIO_STT_EXPORT_SAMPLE_RATE_HZ = int(
    os.getenv("AUDIO_STT_EXPORT_SAMPLE_RATE_HZ", "16000")
)
AUDIO_STT_EXPORT_BITRATE = _env("AUDIO_STT_EXPORT_BITRATE", "64k")
AUDIO_STT_MIN_TRANSCRIPT_BYTES = int(
    os.getenv("AUDIO_STT_MIN_TRANSCRIPT_BYTES", "20")
)
AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS = int(
    os.getenv("AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS", "300")
)

# === File Naming ===
def sanitize_book_name(name: str) -> str:
    """Convert a source filename stem into a filesystem-safe book identifier."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return normalized.lower() or "book"


def book_name_from_source(source_path: Path) -> str:
    """Return the sanitized book name derived from a source filename."""
    return sanitize_book_name(source_path.stem)


def book_name_from_pdf(pdf_path: Path) -> str:
    """Return the sanitized book name derived from a PDF filename."""
    return book_name_from_source(pdf_path)


def book_output_dir(book_name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the output directory for a specific book."""
    return output_dir / book_name


def wiki_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the repo-level wiki directory alongside the output directory."""
    return output_dir.parent / "wiki"


def output_text_path(
    book_name: str,
    name: str,
    lang: str | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Return the output path for a text file for a specific book."""
    book_dir = book_output_dir(book_name, output_dir)
    if lang is None:
        return book_dir / f"{book_name}_{name}.md"
    return book_dir / f"{book_name}_{name}_{lang}.md"


def wiki_text_path(book_name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the repo-level wiki Markdown path for a specific book."""
    return wiki_output_dir(output_dir) / f"{book_name}.md"


def output_audio_path(book_name: str, name: str, lang: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the output path for an audio file for a specific book."""
    return book_output_dir(book_name, output_dir) / f"{book_name}_{name}_{lang}.mp3"


def output_html_path(book_name: str, name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the output path for a single-file HTML artifact for a specific book."""
    return book_output_dir(book_name, output_dir) / f"{book_name}_{name}.html"


# Names for the full-length TTS-preprocessed source
SOURCE_RAW_NAME = "source_raw"
SOURCE_TTS_NAME = "source_tts"
VISUAL_SUMMARY_NAME = "visual_summary_en"
