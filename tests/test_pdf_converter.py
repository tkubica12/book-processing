"""Tests for stage-1 source normalization."""

from pathlib import Path

import pytest

from book_processing.content_understanding import ContentUnderstandingNoUsableMarkdownError
from book_processing.config import SOURCE_RAW_NAME
from book_processing.pdf_converter import (
    convert_epub_to_markdown,
    convert_pdf_to_markdown,
    find_source_files,
    run,
    validate_unique_book_names,
)


def test_find_source_files_returns_supported_sources(tmp_path: Path):
    (tmp_path / "book_0.epub").write_bytes(b"epub")
    (tmp_path / "book_b.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "book_a.md").write_text("# Book A", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("ignored", encoding="utf-8")

    sources = find_source_files(tmp_path)

    assert [path.name for path in sources] == ["book_0.epub", "book_a.md", "book_b.pdf"]


def test_validate_unique_book_names_rejects_collisions(tmp_path: Path):
    sources = [tmp_path / "My Book.md", tmp_path / "My-Book.pdf"]

    with pytest.raises(ValueError, match="same book name"):
        validate_unique_book_names(sources)


def test_run_copies_markdown_inputs_to_source_raw(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    markdown_path = input_dir / "The Book.md"
    markdown_path.write_text("# Heading\n\nBody text.", encoding="utf-8")

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_book_name = "the_book"
    expected_output = output_dir / expected_book_name / f"{expected_book_name}_{SOURCE_RAW_NAME}.md"
    assert outputs == {expected_book_name: expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# Heading\n\nBody text."


def test_run_processes_pdf_via_converter(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pdf_path = input_dir / "The Book.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        "book_processing.pdf_converter.convert_pdf_to_markdown",
        lambda path: "<!-- PageNumber=\"1\" -->\n# Converted\n\nBody",
    )

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_output = output_dir / "the_book" / "the_book_source_raw.md"
    assert outputs == {"the_book": expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# Converted\n\nBody"


def test_run_processes_epub_via_markitdown(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    epub_path = input_dir / "The Book.epub"
    epub_path.write_bytes(b"epub")

    monkeypatch.setattr(
        "book_processing.pdf_converter.convert_epub_to_markdown",
        lambda path: "# EPUB Title\n\nEPUB body",
    )

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_output = output_dir / "the_book" / "the_book_source_raw.md"
    assert outputs == {"the_book": expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# EPUB Title\n\nEPUB body"


def test_convert_epub_to_markdown_uses_markitdown(monkeypatch, tmp_path: Path):
    epub_path = tmp_path / "The Book.epub"
    epub_path.write_bytes(b"epub")

    class FakeResult:
        text_content = "# Converted EPUB\n\nBody"

    class FakeMarkItDown:
        def convert(self, path: str) -> FakeResult:
            assert path == str(epub_path)
            return FakeResult()

    monkeypatch.setattr("book_processing.pdf_converter.MarkItDown", FakeMarkItDown)

    markdown = convert_epub_to_markdown(epub_path)

    assert markdown == "# Converted EPUB\n\nBody"


def test_convert_pdf_to_markdown_falls_back_to_rendered_pages(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "The Book.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    def fail_pdf_analysis(_path: Path) -> str:
        raise ContentUnderstandingNoUsableMarkdownError("placeholder markdown")

    monkeypatch.setattr("book_processing.pdf_converter.analyze_pdf_to_markdown", fail_pdf_analysis)
    monkeypatch.setattr(
        "book_processing.pdf_converter._render_pdf_pages_to_png",
        lambda _path: [("The Book_page_1.png", b"page-1"), ("The Book_page_2.png", b"page-2")],
    )

    def fake_image_analysis(name: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        return f"# {name}\n\n{image_bytes.decode('ascii')}\n\n({mime_type})"

    monkeypatch.setattr("book_processing.pdf_converter.analyze_image_to_markdown", fake_image_analysis)

    markdown = convert_pdf_to_markdown(pdf_path)

    assert markdown == (
        "# The Book_page_1.png\n\npage-1\n\n(image/png)\n\n"
        "# The Book_page_2.png\n\npage-2\n\n(image/png)"
    )


def test_convert_pdf_to_markdown_falls_back_to_local_text(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "The Book.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    def fail_pdf_analysis(_path: Path) -> str:
        raise ContentUnderstandingNoUsableMarkdownError("placeholder markdown")

    def fail_image_analysis(_name: str, _image_bytes: bytes, mime_type: str = "image/png") -> str:
        raise ContentUnderstandingNoUsableMarkdownError(f"placeholder markdown ({mime_type})")

    monkeypatch.setattr("book_processing.pdf_converter.analyze_pdf_to_markdown", fail_pdf_analysis)
    monkeypatch.setattr(
        "book_processing.pdf_converter._render_pdf_pages_to_png",
        lambda _path: [("The Book_page_1.png", b"page-1")],
    )
    monkeypatch.setattr("book_processing.pdf_converter.analyze_image_to_markdown", fail_image_analysis)
    monkeypatch.setattr(
        "book_processing.pdf_converter._extract_pdf_text_locally",
        lambda _path: "## Page 1\n\nRecovered local text",
    )

    markdown = convert_pdf_to_markdown(pdf_path)

    assert markdown == "## Page 1\n\nRecovered local text"
