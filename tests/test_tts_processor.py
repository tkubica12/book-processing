"""Tests for TTS metadata helpers."""

from book_processing.tts_processor import _build_audio_metadata, _display_book_title


def test_display_book_title_from_sanitized_name():
    assert _display_book_title("inference_engineering") == "Inference Engineering"


def test_build_audio_metadata_contains_book_type_and_language():
    metadata = _build_audio_metadata("inference_engineering", "summary_2min", "en")

    assert metadata["title"] == "Inference Engineering - Summary 2 min (en)"
    assert metadata["album"] == "Inference Engineering"
    assert metadata["language"] == "en"
    assert metadata["comment"] == "inference_engineering_summary_2min_en"


def test_build_audio_metadata_for_podcast():
    metadata = _build_audio_metadata("machines_of_loving_grace", "podcast_60min", "cs")

    assert metadata["title"] == "Machines Of Loving Grace - Podcast 60 min (cs)"