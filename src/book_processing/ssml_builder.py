"""SSML construction helpers for Azure Text-to-Speech."""

import re
import xml.sax.saxutils as saxutils

from book_processing.config import (
    LANGUAGES,
    PODCAST_SPEAKERS,
    PROSODY_RATE,
    VOICE_FEMALE,
    VOICE_MALE,
)

# Azure Batch Synthesis has a practical SSML input limit (~64KB of text per input).
# We chunk long texts to stay well within limits.
MAX_CHARS_PER_CHUNK = 50_000


def _escape_xml(text: str) -> str:
    """Escape XML special characters in text content."""
    return saxutils.escape(text)


def build_single_voice_ssml(text: str, lang: str, voice: str = VOICE_MALE) -> str:
    """Build SSML for a single voice with prosody rate adjustment.

    Args:
        text: Plain text content to speak.
        lang: Language code ('en' or 'cs').
        voice: Voice name to use.

    Returns:
        Complete SSML string.
    """
    xml_lang = LANGUAGES[lang]["xml_lang"]
    escaped = _escape_xml(text)

    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{xml_lang}">'
        f'<voice name="{voice}">'
        f'<lang xml:lang="{xml_lang}">'
        f'<prosody rate="{PROSODY_RATE}">'
        f"{escaped}"
        f"</prosody>"
        f"</lang>"
        f"</voice>"
        f"</speak>"
    )


def parse_podcast_script(script: str, lang: str) -> list[tuple[str, str]]:
    """Parse a podcast script with [Speaker]: tags into (voice_name, text) pairs.

    Expected format:
        [Andrew]: Hello and welcome...
        [Emma]: Thanks! Today we're diving into...

    Args:
        script: Raw podcast script with speaker tags.
        lang: Language code for looking up speaker→voice mapping.

    Returns:
        List of (voice_name, text_segment) tuples.
    """
    speakers = PODCAST_SPEAKERS[lang]
    male_name = speakers["male"]
    female_name = speakers["female"]

    # Build name→voice mapping
    name_to_voice = {
        male_name.lower(): VOICE_MALE,
        female_name.lower(): VOICE_FEMALE,
    }

    # Pattern: [SpeakerName]: followed by text until the next [SpeakerName]: or end
    pattern = re.compile(
        r"\[(" + re.escape(male_name) + r"|" + re.escape(female_name) + r")\]:\s*",
        re.IGNORECASE,
    )

    segments: list[tuple[str, str]] = []
    parts = pattern.split(script)

    # parts[0] is text before first tag (usually empty), then alternating name, text
    i = 1  # skip preamble
    while i < len(parts) - 1:
        speaker_name = parts[i].strip().lower()
        text = parts[i + 1].strip()
        if text:
            voice = name_to_voice.get(speaker_name, VOICE_MALE)
            segments.append((voice, text))
        i += 2

    return segments


def build_podcast_ssml(script: str, lang: str) -> str:
    """Build multi-voice SSML from a podcast script with speaker tags.

    Args:
        script: Raw podcast script with [Speaker]: tags.
        lang: Language code ('en' or 'cs').

    Returns:
        Complete SSML string with alternating voices.
    """
    xml_lang = LANGUAGES[lang]["xml_lang"]
    segments = parse_podcast_script(script, lang)

    if not segments:
        raise ValueError("No speaker segments found in podcast script")

    voice_elements: list[str] = []
    for voice, text in segments:
        escaped = _escape_xml(text)
        voice_elements.append(
            f'<voice name="{voice}">'
            f'<lang xml:lang="{xml_lang}">'
            f'<prosody rate="{PROSODY_RATE}">'
            f"{escaped}"
            f"</prosody>"
            f"</lang>"
            f"</voice>"
        )

    body = "".join(voice_elements)
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{xml_lang}">'
        f"{body}"
        f"</speak>"
    )


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into chunks at paragraph boundaries, respecting max size.

    Args:
        text: Full text to chunk.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for \n\n separator
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_chunked_ssml(text: str, lang: str, is_podcast: bool = False) -> list[str]:
    """Build a list of SSML strings, chunking long text as needed.

    Args:
        text: Full text content or podcast script.
        lang: Language code ('en' or 'cs').
        is_podcast: If True, parse speaker tags for multi-voice SSML.

    Returns:
        List of SSML strings, one per chunk.
    """
    if is_podcast:
        # For podcast, we don't chunk — the multi-voice SSML is built as one piece.
        # If it's too long, we chunk the script by speaker segments.
        ssml = build_podcast_ssml(text, lang)
        if len(ssml) <= MAX_CHARS_PER_CHUNK * 2:  # SSML overhead is larger
            return [ssml]
        # For very long podcasts, split into halves at a speaker boundary
        segments = parse_podcast_script(text, lang)
        mid = len(segments) // 2
        xml_lang = LANGUAGES[lang]["xml_lang"]

        def _segments_to_ssml(segs: list[tuple[str, str]]) -> str:
            parts = []
            for voice, seg_text in segs:
                escaped = _escape_xml(seg_text)
                parts.append(
                    f'<voice name="{voice}">'
                    f'<lang xml:lang="{xml_lang}">'
                    f'<prosody rate="{PROSODY_RATE}">'
                    f"{escaped}"
                    f"</prosody>"
                    f"</lang>"
                    f"</voice>"
                )
            body = "".join(parts)
            return (
                f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{xml_lang}">'
                f"{body}"
                f"</speak>"
            )

        return [
            _segments_to_ssml(segments[:mid]),
            _segments_to_ssml(segments[mid:]),
        ]
    else:
        chunks = chunk_text(text)
        return [build_single_voice_ssml(c, lang) for c in chunks]
