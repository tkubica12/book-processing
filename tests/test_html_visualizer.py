"""Tests for progressive-disclosure HTML visualization helpers."""

from pathlib import Path

from book_processing.config import SOURCE_RAW_NAME, output_text_path
from book_processing.html_visualizer import (
    _parse_outline_json,
    _render_html,
    discover_existing_source_raws,
)


def _sample_outline() -> dict:
    return {
        "title": "Sample Book",
        "subtitle": "A concise explanation of the source.",
        "main_summary": "This is about **key ideas** and careful structure.",
        "segments": [
            {
                "title": "Core Idea",
                "summary": "The first segment explains the main point.",
                "details": [
                    "Sentence one explains the segment.",
                    "Sentence two adds detail.",
                    "Sentence three keeps it concrete.",
                    "Sentence four connects the ideas.",
                    "Sentence five closes the segment.",
                ],
                "subtopics": [
                    {
                        "title": "Specific Mechanism",
                        "summary": "A focused subtopic summary.",
                        "details": [
                            "Subtopic sentence one.",
                            "Subtopic sentence two.",
                            "Subtopic sentence three.",
                        ],
                    }
                ],
            },
            {
                "title": "Second Idea",
                "summary": "The second segment explains another point.",
                "details": ["One.", "Two.", "Three.", "Four.", "Five."],
                "subtopics": [],
            },
            {
                "title": "Third Idea",
                "summary": "The third segment completes the minimum structure.",
                "details": ["One.", "Two.", "Three.", "Four.", "Five."],
                "subtopics": [],
            },
        ],
    }


def test_parse_outline_json_accepts_fenced_json():
    outline = _parse_outline_json(
        '```json\n{"title":"T","segments":[{"title":"A"},{"title":"B"},{"title":"C"}]}\n```'
    )

    assert outline["title"] == "T"
    assert len(outline["segments"]) == 3


def test_render_html_contains_theme_and_collapsed_segments():
    html = _render_html(_sample_outline())

    assert "clawpilotTheme" in html
    assert "--cp-bg: #f7f4ef;" in html
    assert "font-family: \"Segoe UI\", Aptos" in html
    assert "<strong>key ideas</strong>" in html
    assert 'aria-expanded="false"' in html
    assert "hidden" in html
    assert "Specific Mechanism" in html


def test_discover_existing_source_raws(tmp_path: Path):
    source_path = output_text_path("sample_book", SOURCE_RAW_NAME, output_dir=tmp_path)
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Sample\n\nBody", encoding="utf-8")

    assert discover_existing_source_raws(tmp_path) == {"sample_book": source_path}
