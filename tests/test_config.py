"""Tests for the configuration module."""

import importlib
from pathlib import Path

import book_processing.config as config
from book_processing.config import (
    AUDIO_STT_CHUNK_DURATION_MINUTES,
    AUDIO_STT_EXPORT_BITRATE,
    AUDIO_STT_EXPORT_SAMPLE_RATE_HZ,
    AUDIO_STT_MAX_CONCURRENT_CHUNKS,
    AUDIO_STT_MIN_TRANSCRIPT_BYTES,
    AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS,
    AZURE_OPENAI_MODEL,
    BOOK_MAX_WORKERS,
    CONTENT_UNDERSTANDING_ANALYZER_ID,
    CONTENT_UNDERSTANDING_API_VERSION,
    SUMMARY_TYPES,
    VISUAL_SUMMARY_NAME,
    LANGUAGES,
    PODCAST_SPEAKERS,
    WPM_AT_TARGET_SPEED,
    PROSODY_RATE,
    LLM_MAX_WORKERS,
    AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION,
    TTS_JOB_MAX_RETRIES,
    TTS_MAX_CHARS_PER_CHUNK,
    TTS_MAX_CONCURRENT_JOBS,
    book_output_dir,
    book_name_from_source,
    book_name_from_pdf,
    output_text_path,
    output_audio_path,
    output_html_path,
    sanitize_book_name,
    wiki_output_dir,
    wiki_text_path,
)


def test_wpm_calibrated():
    assert WPM_AT_TARGET_SPEED == 160


def test_prosody_rate():
    assert PROSODY_RATE == "+20%"


def test_summary_types_word_counts():
    assert SUMMARY_TYPES["summary_5min"]["target_words"] == 800
    assert SUMMARY_TYPES["summary_20min"]["target_words"] == 3200
    assert SUMMARY_TYPES["podcast_20min"]["target_words"] == 3200
    assert SUMMARY_TYPES["podcast_60min"]["target_words"] == 9600
    assert "summary_2min" not in SUMMARY_TYPES


def test_podcast_is_flagged():
    for name, spec in SUMMARY_TYPES.items():
        if "podcast" in name:
            assert spec["is_podcast"] is True
        else:
            assert spec["is_podcast"] is False


def test_languages():
    assert "en" in LANGUAGES
    assert "cs" in LANGUAGES
    assert LANGUAGES["en"]["xml_lang"] == "en-US"
    assert LANGUAGES["cs"]["xml_lang"] == "cs-CZ"


def test_podcast_speakers():
    assert PODCAST_SPEAKERS["en"]["male"] == "Andrew"
    assert PODCAST_SPEAKERS["en"]["female"] == "Emma"
    assert PODCAST_SPEAKERS["cs"]["male"] == "Tomáš"
    assert PODCAST_SPEAKERS["cs"]["female"] == "Kateřina"


def test_output_text_path():
    path = output_text_path("inference_engineering", "summary_5min", "en")
    assert path.name == "inference_engineering_summary_5min_en.md"
    assert path.parent == book_output_dir("inference_engineering")
    assert isinstance(path, Path)


def test_output_audio_path():
    path = output_audio_path("inference_engineering", "podcast_60min", "cs")
    assert path.name == "inference_engineering_podcast_60min_cs.mp3"
    assert path.parent == book_output_dir("inference_engineering")


def test_output_html_path():
    path = output_html_path("inference_engineering", VISUAL_SUMMARY_NAME)
    assert path.name == "inference_engineering_visual_summary_en.html"
    assert path.parent == book_output_dir("inference_engineering")


def test_output_raw_text_path():
    path = output_text_path("inference_engineering", "source_raw")
    assert path.name == "inference_engineering_source_raw.md"
    assert path.parent == book_output_dir("inference_engineering")


def test_wiki_text_path():
    path = wiki_text_path("inference_engineering")
    assert path.name == "inference_engineering.md"
    assert path.parent == wiki_output_dir()


def test_sanitize_book_name():
    assert sanitize_book_name("Inference Engineering 2nd Edition") == "inference_engineering_2nd_edition"


def test_book_name_from_pdf():
    assert book_name_from_pdf(Path(r"C:\tmp\Inference Engineering.pdf")) == "inference_engineering"


def test_book_name_from_source_markdown():
    assert book_name_from_source(Path(r"C:\tmp\Inference Engineering.md")) == "inference_engineering"


def test_parallelism_config():
    assert BOOK_MAX_WORKERS >= 1
    assert LLM_MAX_WORKERS >= 1
    assert TTS_MAX_CONCURRENT_JOBS >= 1


def test_content_understanding_defaults():
    assert CONTENT_UNDERSTANDING_ANALYZER_ID == "prebuilt-documentSearch"
    assert CONTENT_UNDERSTANDING_API_VERSION == "2025-11-01"


def test_tts_reliability_config():
    assert TTS_JOB_MAX_RETRIES >= 1
    assert TTS_MAX_CHARS_PER_CHUNK <= 25_000


def test_audio_stt_config():
    assert AZURE_OPENAI_MODEL
    assert AZURE_SPEECH_FAST_TRANSCRIPTION_API_VERSION == "2025-10-15"
    assert AUDIO_STT_CHUNK_DURATION_MINUTES >= 1
    assert AUDIO_STT_MAX_CONCURRENT_CHUNKS >= 1
    assert AUDIO_STT_EXPORT_SAMPLE_RATE_HZ >= 8_000
    assert AUDIO_STT_EXPORT_BITRATE.endswith("k")
    assert AUDIO_STT_MIN_TRANSCRIPT_BYTES >= 1
    assert AUDIO_STT_RETRY_MAX_BACKOFF_SECONDS >= 30


def test_endpoint_env_overrides(monkeypatch):
    original_openai = config.AZURE_OPENAI_ENDPOINT
    original_model = config.AZURE_OPENAI_MODEL
    original_speech = config.AZURE_SPEECH_ENDPOINT

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://openai.example")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://speech.example")
    reloaded = importlib.reload(config)

    assert reloaded.AZURE_OPENAI_ENDPOINT == "https://openai.example"
    assert reloaded.AZURE_OPENAI_MODEL == "gpt-4.1"
    assert reloaded.AZURE_SPEECH_ENDPOINT == "https://speech.example"

    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    restored = importlib.reload(config)

    assert restored.AZURE_OPENAI_ENDPOINT == original_openai
    assert restored.AZURE_OPENAI_MODEL == original_model
    assert restored.AZURE_SPEECH_ENDPOINT == original_speech
