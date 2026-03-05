"""Tests for the configuration module."""

from pathlib import Path

from book_processing.config import (
    SUMMARY_TYPES,
    LANGUAGES,
    PODCAST_SPEAKERS,
    WPM_AT_130_PERCENT,
    output_text_path,
    output_audio_path,
)


def test_wpm_at_130_percent():
    assert WPM_AT_130_PERCENT == 195


def test_summary_types_word_counts():
    assert SUMMARY_TYPES["summary_2min"]["target_words"] == 390
    assert SUMMARY_TYPES["summary_5min"]["target_words"] == 975
    assert SUMMARY_TYPES["summary_20min"]["target_words"] == 3900
    assert SUMMARY_TYPES["podcast_60min"]["target_words"] == 11700


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
