"""Tests for stage-1 source normalization."""

from pathlib import Path

import httpx
import pytest

from book_processing.content_understanding import ContentUnderstandingNoUsableMarkdownError
from book_processing.config import SOURCE_RAW_NAME, wiki_text_path
from book_processing.audio_transcriber import InvalidAudioSourceError
from book_processing.metadata import read_metadata
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
    (tmp_path / "book_a_text.txt").write_text("Book text", encoding="utf-8")
    (tmp_path / "book_c.mp3").write_bytes(b"mp3")
    (tmp_path / "book_d.m4b").write_bytes(b"m4b")
    (tmp_path / "ignore.docx").write_text("ignored", encoding="utf-8")

    sources = find_source_files(tmp_path)

    assert [path.name for path in sources] == [
        "book_0.epub",
        "book_a.md",
        "book_a_text.txt",
        "book_b.pdf",
        "book_c.mp3",
        "book_d.m4b",
    ]


def test_find_source_files_treats_arxiv_pdfs_as_papers_and_ignores_arxiv_folders(tmp_path: Path):
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "Paper 2.pdf").write_bytes(b"%PDF-1.7")
    nested = arxiv_dir / "Ignored Folder"
    nested.mkdir()
    (nested / "Paper 1.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "Book.pdf").write_bytes(b"%PDF-1.7")

    sources = find_source_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in sources] == ["arxiv/Paper 2.pdf", "Book.pdf"]


def test_find_source_files_includes_audio_directories_with_supported_audio_only(tmp_path: Path):
    audio_dir = tmp_path / "Podcast 10"
    audio_dir.mkdir()
    (audio_dir / "Track 10.mp3").write_bytes(b"mp3")
    (audio_dir / "Track 2.m4b").write_bytes(b"m4b")
    (audio_dir / "cover.jpg").write_bytes(b"jpg")
    (audio_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    ignored_dir = tmp_path / "Scans"
    ignored_dir.mkdir()
    (ignored_dir / "page01.jpg").write_bytes(b"jpg")

    sources = find_source_files(tmp_path)

    assert [path.name for path in sources] == ["Podcast 10"]


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
    expected_wiki = wiki_text_path(expected_book_name, output_dir=output_dir)
    assert outputs == {expected_book_name: expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# Heading\n\nBody text."
    assert expected_wiki.read_text(encoding="utf-8") == "# Heading\n\nBody text."
    metadata = read_metadata(output_dir / expected_book_name)
    assert metadata is not None
    assert metadata.source_path == "The Book.md"
    assert metadata.document_type == "book"
    assert metadata.source_medium == "text"
    assert metadata.labels


def test_run_writes_paper_metadata_for_arxiv_pdf(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    arxiv_dir = input_dir / "arxiv"
    arxiv_dir.mkdir(parents=True)
    pdf_path = arxiv_dir / "HyDRA.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        "book_processing.pdf_converter.convert_pdf_to_markdown",
        lambda path: "# HyDRA\n\nAbstract\n\nAI routing paper.\n\nReferences\n\nOng et al.",
    )

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    assert set(outputs) == {"hydra"}
    metadata = read_metadata(output_dir / "hydra")
    assert metadata is not None
    assert metadata.source_path == "arxiv\\HyDRA.pdf"
    assert metadata.document_type == "paper"
    assert metadata.source_medium == "PDF"
    assert "AI" in metadata.labels


def test_run_copies_text_inputs_to_source_raw_and_wiki(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    text_path = input_dir / "The Book.txt"
    text_path.write_text("Plain body text.", encoding="utf-8")

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_book_name = "the_book"
    expected_output = output_dir / expected_book_name / f"{expected_book_name}_{SOURCE_RAW_NAME}.md"
    expected_wiki = wiki_text_path(expected_book_name, output_dir=output_dir)
    assert outputs == {expected_book_name: expected_output}
    assert expected_output.read_text(encoding="utf-8") == "Plain body text."
    assert expected_wiki.read_text(encoding="utf-8") == "Plain body text."


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
    expected_wiki = wiki_text_path("the_book", output_dir=output_dir)
    assert outputs == {"the_book": expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# Converted\n\nBody"
    assert expected_wiki.read_text(encoding="utf-8") == "# Converted\n\nBody"


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
    expected_wiki = wiki_text_path("the_book", output_dir=output_dir)
    assert outputs == {"the_book": expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# EPUB Title\n\nEPUB body"
    assert expected_wiki.read_text(encoding="utf-8") == "# EPUB Title\n\nEPUB body"


def test_run_processes_audio_via_transcriber(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    expected_output_dir = tmp_path / "output"
    input_dir.mkdir()
    audio_path = input_dir / "The Book.mp3"
    audio_path.write_bytes(b"ID3")

    def fake_convert(
        path: Path,
        output_dir: Path,
        *,
        book_name: str | None = None,
        artifact_stem: str | None = None,
    ) -> str:
        assert path == audio_path
        assert output_dir == expected_output_dir
        assert book_name is None
        assert artifact_stem is None
        return "# Audio Title\n\nTranscript body"

    monkeypatch.setattr("book_processing.pdf_converter.convert_audio_to_markdown", fake_convert)

    outputs = run(input_dir=input_dir, output_dir=expected_output_dir)

    expected_output = expected_output_dir / "the_book" / "the_book_source_raw.md"
    expected_wiki = wiki_text_path("the_book", output_dir=expected_output_dir)
    assert outputs == {"the_book": expected_output}
    assert expected_output.read_text(encoding="utf-8") == "# Audio Title\n\nTranscript body"
    assert expected_wiki.read_text(encoding="utf-8") == "# Audio Title\n\nTranscript body"


def test_run_processes_audio_directory_into_single_source_raw(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir = input_dir / "My Podcast"
    source_dir.mkdir(parents=True)
    (source_dir / "Track 10.mp3").write_bytes(b"mp3")
    (source_dir / "Track 2.mp3").write_bytes(b"mp3")
    (source_dir / "cover.jpg").write_bytes(b"jpg")
    (source_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    disc_dir = source_dir / "Disc 2"
    disc_dir.mkdir()
    (disc_dir / "Track 1.m4b").write_bytes(b"m4b")

    calls: list[tuple[Path, Path, str | None, str | None]] = []

    def fake_convert(
        path: Path,
        output_dir: Path,
        *,
        book_name: str | None = None,
        artifact_stem: str | None = None,
    ) -> str:
        calls.append((path, output_dir, book_name, artifact_stem))
        return f"# {path.stem}\n\nTranscript for {path.name}"

    monkeypatch.setattr("book_processing.pdf_converter.convert_audio_to_markdown", fake_convert)

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_book_name = "my_podcast"
    expected_output = output_dir / expected_book_name / f"{expected_book_name}_{SOURCE_RAW_NAME}.md"
    expected_wiki = wiki_text_path(expected_book_name, output_dir=output_dir)

    assert outputs == {expected_book_name: expected_output}
    assert [call[0].relative_to(source_dir).as_posix() for call in calls] == [
        "Disc 2/Track 1.m4b",
        "Track 2.mp3",
        "Track 10.mp3",
    ]
    assert all(call[1] == output_dir for call in calls)
    assert all(call[2] == expected_book_name for call in calls)
    assert [call[3] for call in calls] == [
        "my_podcast_track0001_disc_2_track_1",
        "my_podcast_track0002_track_2",
        "my_podcast_track0003_track_10",
    ]
    expected_markdown = (
        "# Track 1\n\nTranscript for Track 1.m4b\n\n"
        "# Track 2\n\nTranscript for Track 2.mp3\n\n"
        "# Track 10\n\nTranscript for Track 10.mp3"
    )
    assert expected_output.read_text(encoding="utf-8") == expected_markdown
    assert expected_wiki.read_text(encoding="utf-8") == expected_markdown


def test_run_processes_audio_directory_skips_invalid_tracks(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir = input_dir / "My Podcast"
    source_dir.mkdir(parents=True)
    (source_dir / "Track 1.mp3").write_bytes(b"mp3")
    (source_dir / "Track 2.mp3").write_bytes(b"mp3")

    def fake_convert(
        path: Path,
        output_dir: Path,
        *,
        book_name: str | None = None,
        artifact_stem: str | None = None,
    ) -> str:
        if path.name == "Track 1.mp3":
            raise InvalidAudioSourceError("empty file")
        return f"# {path.stem}\n\nTranscript for {path.name}"

    monkeypatch.setattr("book_processing.pdf_converter.convert_audio_to_markdown", fake_convert)

    outputs = run(input_dir=input_dir, output_dir=output_dir)

    expected_book_name = "my_podcast"
    expected_output = output_dir / expected_book_name / f"{expected_book_name}_{SOURCE_RAW_NAME}.md"
    expected_wiki = wiki_text_path(expected_book_name, output_dir=output_dir)
    expected_markdown = (
        "## Skipped audio track\n\n"
        "Could not transcribe `Track 1.mp3` because the source audio was invalid: empty file\n\n"
        "# Track 2\n\nTranscript for Track 2.mp3"
    )

    assert outputs == {expected_book_name: expected_output}
    assert expected_output.read_text(encoding="utf-8") == expected_markdown
    assert expected_wiki.read_text(encoding="utf-8") == expected_markdown


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


def test_convert_pdf_to_markdown_falls_back_to_local_text_on_connect_error(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "The Book.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    request = httpx.Request("POST", "https://example.test/analyze")

    def fail_pdf_analysis(_path: Path) -> str:
        raise httpx.ConnectError("dns failed", request=request)

    monkeypatch.setattr("book_processing.pdf_converter.analyze_pdf_to_markdown", fail_pdf_analysis)
    monkeypatch.setattr(
        "book_processing.pdf_converter._extract_pdf_text_locally",
        lambda _path: "## Page 1\n\nRecovered local text",
    )

    markdown = convert_pdf_to_markdown(pdf_path)

    assert markdown == "## Page 1\n\nRecovered local text"
