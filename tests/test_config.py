"""Tests for the configuration module."""

from pathlib import Path

from book_processing.config import (
    BOOK_MAX_WORKERS,
    CONTENT_UNDERSTANDING_ANALYZER_ID,
    CONTENT_UNDERSTANDING_API_VERSION,
    SUMMARY_TYPES,
    LANGUAGES,
    PODCAST_SPEAKERS,
    WPM_AT_TARGET_SPEED,
    PROSODY_RATE,
    LLM_MAX_WORKERS,
    TTS_JOB_MAX_RETRIES,
    TTS_MAX_CHARS_PER_CHUNK,
    TTS_MAX_CONCURRENT_JOBS,
    book_name_from_source,
    book_name_from_pdf,
    output_text_path,
    output_audio_path,
    sanitize_book_name,
)


def test_wpm_calibrated():
    assert WPM_AT_TARGET_SPEED == 160


def test_prosody_rate():
    assert PROSODY_RATE == "+20%"


def test_summary_types_word_counts():
    assert SUMMARY_TYPES["summary_2min"]["target_words"] == 320
    assert SUMMARY_TYPES["summary_5min"]["target_words"] == 800
    assert SUMMARY_TYPES["summary_20min"]["target_words"] == 3200
    assert SUMMARY_TYPES["podcast_60min"]["target_words"] == 9600


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
    path = output_text_path("inference_engineering", "summary_2min", "en")
    assert path.name == "inference_engineering_summary_2min_en.md"
    assert isinstance(path, Path)


def test_output_audio_path():
    path = output_audio_path("inference_engineering", "podcast_60min", "cs")
    assert path.name == "inference_engineering_podcast_60min_cs.mp3"


def test_output_raw_text_path():
    path = output_text_path("inference_engineering", "source_raw")
    assert path.name == "inference_engineering_source_raw.md"


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
