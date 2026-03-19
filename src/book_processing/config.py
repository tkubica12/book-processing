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

# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT = "https://sw-v2-project-resource.cognitiveservices.azure.com"
AZURE_OPENAI_MODEL = "gpt-5.2"
AZURE_OPENAI_API_VERSION = "2025-04-01-preview"

# === Azure Speech (Batch Synthesis API) ===
AZURE_SPEECH_ENDPOINT = "https://sw-v2-project-resource.cognitiveservices.azure.com"
AZURE_SPEECH_API_VERSION = "2024-04-01"
AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# === Azure Content Understanding ===
CONTENT_UNDERSTANDING_ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT", "").strip()
CONTENT_UNDERSTANDING_API_VERSION = os.getenv("CONTENT_UNDERSTANDING_API_VERSION", "2025-11-01").strip()
CONTENT_UNDERSTANDING_ANALYZER_ID = os.getenv("CONTENT_UNDERSTANDING_ANALYZER_ID", "prebuilt-documentSearch").strip()
CONTENT_UNDERSTANDING_API_KEY = os.getenv("CONTENT_UNDERSTANDING_API_KEY", "").strip()
CONTENT_UNDERSTANDING_PROCESSING_LOCATION = os.getenv("CONTENT_UNDERSTANDING_PROCESSING_LOCATION", "").strip()
CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS = float(
    os.getenv("CONTENT_UNDERSTANDING_POLL_INTERVAL_SECONDS", "2")
)

# === Voice Configuration ===
# Dragon HD voices are multilingual — same voices for EN and CZ
VOICE_MALE = "en-US-Andrew:DragonHDLatestNeural"
VOICE_FEMALE = "en-US-Emma:DragonHDLatestNeural"

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
    "summary_2min": {
        "duration_min": 2,
        "target_words": 2 * WPM_AT_TARGET_SPEED,  # ~320
        "description": "High-level architectural summary",
        "is_podcast": False,
    },
    "summary_5min": {
        "duration_min": 5,
        "target_words": 5 * WPM_AT_TARGET_SPEED,  # ~800
        "description": "Architecture patterns, trade-offs, technology choices",
        "is_podcast": False,
    },
    "summary_20min": {
        "duration_min": 20,
        "target_words": 20 * WPM_AT_TARGET_SPEED,  # ~3200
        "description": "Technical deep-dive with implementation details",
        "is_podcast": False,
    },
    "podcast_60min": {
        "duration_min": 60,
        "target_words": 60 * WPM_AT_TARGET_SPEED,  # ~9600
        "description": "Two-host technical podcast, entertaining and deeply technical",
        "is_podcast": True,
    },
}

# === Parallelism ===
BOOK_MAX_WORKERS = 2  # Concurrent books processed in parallel batches
LLM_MAX_WORKERS = 6  # Concurrent LLM requests
TTS_MAX_CONCURRENT_JOBS = 8  # Concurrent Batch Synthesis jobs
TTS_JOB_MAX_RETRIES = 3  # Retries per submitted batch job before failing
TTS_JOB_STALE_AFTER_SECONDS = int(
    os.getenv("TTS_JOB_STALE_AFTER_SECONDS", "600")
)  # Retry batch jobs that stay running for too long
TTS_MAX_CHARS_PER_CHUNK = 25_000  # Smaller chunks are more reliable for long-form TTS

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


def output_audio_path(book_name: str, name: str, lang: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Return the output path for an audio file for a specific book."""
    return book_output_dir(book_name, output_dir) / f"{book_name}_{name}_{lang}.mp3"


# Names for the full-length TTS-preprocessed source
SOURCE_RAW_NAME = "source_raw"
SOURCE_TTS_NAME = "source_tts"
