"""Tests for the configuration module."""

from pathlib import Path

from book_processing.config import (
    SUMMARY_TYPES,
    LANGUAGES,
    PODCAST_SPEAKERS,
    WPM_AT_TARGET_SPEED,
    PROSODY_RATE,
    LLM_MAX_WORKERS,
    TTS_JOB_MAX_RETRIES,
    TTS_MAX_CHARS_PER_CHUNK,
    TTS_MAX_CONCURRENT_JOBS,
    output_text_path,
    output_audio_path,
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
    path = output_text_path("summary_2min", "en")
    assert path.name == "summary_2min_en.md"
    assert isinstance(path, Path)


def test_output_audio_path():
    path = output_audio_path("podcast_60min", "cs")
    assert path.name == "podcast_60min_cs.mp3"


def test_parallelism_config():
    assert LLM_MAX_WORKERS >= 1
    assert TTS_MAX_CONCURRENT_JOBS >= 1


def test_tts_reliability_config():
    assert TTS_JOB_MAX_RETRIES >= 1
    assert TTS_MAX_CHARS_PER_CHUNK <= 25_000
