"""Tests for Jinja-based prompt templates."""

import pytest

from book_processing.prompt_templates import available_prompt_templates, render_prompt


def test_prompt_templates_are_discoverable():
    names = [path.name for path in available_prompt_templates()]

    assert names == [
        "podcast_section_system.j2",
        "podcast_section_user.j2",
        "simple_summary_system.j2",
        "simple_summary_user.j2",
        "tts_chunk_system.j2",
        "tts_chunk_user.j2",
    ]


def test_simple_summary_prompt_renders_variables():
    system_prompt = render_prompt(
        "simple_summary_system.j2",
        lang_label="English",
        target_words=320,
    )
    user_prompt = render_prompt(
        "simple_summary_user.j2",
        description="high-level architectural summary",
        target_words=320,
        source_md="Source text",
    )

    assert "English" in system_prompt
    assert "320" in system_prompt
    assert "ceiling, not a quota" in system_prompt
    assert "high-level architectural summary" in user_prompt
    assert "Source text" in user_prompt
    assert "produce a shorter answer rather than padding it" in user_prompt


def test_podcast_prompt_renders_section_roles():
    opening = render_prompt(
        "podcast_section_user.j2",
        section_num=1,
        total_sections=5,
        section_role="opening",
        words_per_section=1900,
        max_words=2280,
        section_text="Section text",
    )
    closing = render_prompt(
        "podcast_section_user.j2",
        section_num=5,
        total_sections=5,
        section_role="closing",
        words_per_section=1900,
        max_words=2280,
        section_text="Section text",
    )

    assert "OPENING segment" in opening
    assert "CLOSING segment" in closing
    assert "end earlier and keep it tight" in opening


def test_tts_prompt_includes_optional_translation_instruction():
    czech_prompt = render_prompt(
        "tts_chunk_system.j2",
        lang_label="Czech",
        translate_to_czech=True,
    )
    english_prompt = render_prompt(
        "tts_chunk_system.j2",
        lang_label="English",
        translate_to_czech=False,
    )

    assert "Translate to Czech" in czech_prompt
    assert "Translate to Czech" not in english_prompt


def test_render_prompt_requires_expected_variables():
    with pytest.raises(Exception):
        render_prompt("simple_summary_system.j2")