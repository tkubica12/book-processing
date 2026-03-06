"""Tests for the SSML builder module."""

from book_processing.ssml_builder import (
    build_single_voice_ssml,
    build_podcast_ssml,
    parse_podcast_script,
    chunk_text,
    build_chunked_ssml,
)


def test_build_single_voice_ssml_english():
    ssml = build_single_voice_ssml("Hello world", "en")
    assert 'xml:lang="en-US"' in ssml
    assert 'name="en-US-Andrew:DragonHDLatestNeural"' in ssml
    assert 'rate="+20%"' in ssml
    assert "Hello world" in ssml
    assert ssml.startswith("<speak")
    assert ssml.endswith("</speak>")


def test_build_single_voice_ssml_czech():
    ssml = build_single_voice_ssml("Ahoj světe", "cs")
    assert 'xml:lang="cs-CZ"' in ssml
    assert "Ahoj světe" in ssml


def test_build_single_voice_ssml_escapes_xml():
    ssml = build_single_voice_ssml("x < y & z > w", "en")
    assert "&lt;" in ssml
    assert "&amp;" in ssml
    assert "&gt;" in ssml


def test_parse_podcast_script_english():
    script = "[Andrew]: Hello everyone!\n[Emma]: Welcome to the show!\n[Andrew]: Let's dive in."
    segments = parse_podcast_script(script, "en")
    assert len(segments) == 3
    assert segments[0][0] == "en-US-Andrew:DragonHDLatestNeural"
    assert "Hello everyone!" in segments[0][1]
    assert segments[1][0] == "en-US-Emma:DragonHDLatestNeural"
    assert "Welcome to the show!" in segments[1][1]


def test_parse_podcast_script_czech():
    script = "[Tomáš]: Ahoj!\n[Kateřina]: Vítejte!"
    segments = parse_podcast_script(script, "cs")
    assert len(segments) == 2
    assert segments[0][0] == "en-US-Andrew:DragonHDLatestNeural"
    assert "Ahoj!" in segments[0][1]
    assert segments[1][0] == "en-US-Emma:DragonHDLatestNeural"


def test_build_podcast_ssml():
    script = "[Andrew]: First segment.\n[Emma]: Second segment."
    ssml = build_podcast_ssml(script, "en")
    assert ssml.count('<voice name=') == 2
    assert "First segment." in ssml
    assert "Second segment." in ssml
    assert 'rate="+20%"' in ssml


def test_chunk_text_short():
    text = "Short text"
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    para = "A" * 100
    text = "\n\n".join([para] * 20)  # ~2000+ chars
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    # Verify no chunk exceeds max (approximately, due to paragraph boundaries)
    for chunk in chunks:
        assert len(chunk) <= 600  # some tolerance for paragraph boundary


def test_build_chunked_ssml_single():
    ssml_list = build_chunked_ssml("Short text", "en", is_podcast=False)
    assert len(ssml_list) == 1
    assert "<speak" in ssml_list[0]


def test_build_chunked_ssml_podcast():
    script = "[Andrew]: Hello!\n[Emma]: Hi there!"
    ssml_list = build_chunked_ssml(script, "en", is_podcast=True)
    assert len(ssml_list) >= 1
    assert ssml_list[0].count('<voice name=') == 2
