"""Tests for the static book library generator."""

from pathlib import Path

from book_processing.site_generator import discover_books, generate_site


def test_generate_site_creates_landing_and_book_pages(tmp_path: Path):
    book_dir = tmp_path / "sample_book"
    book_dir.mkdir()
    (book_dir / "sample_book_visual_summary_en.html").write_text(
        "<html><body><h1>Sample Book</h1><p class=\"summary\">Useful <strong>ideas</strong>.</p></body></html>",
        encoding="utf-8",
    )
    (book_dir / "sample_book_summary_5min_en.mp3").write_bytes(b"audio")
    (book_dir / "sample_book_podcast_20min_cs.mp3").write_bytes(b"audio")
    (book_dir / "sample_book_source_raw.md").write_text("# Sample", encoding="utf-8")
    (book_dir / "metadata.yaml").write_text(
        "source_path: Sample.md\ndocument_type: book\nsource_medium: text\nadded_date: 2026-05-01\nlabels:\n- Other\n",
        encoding="utf-8",
    )

    books = generate_site(tmp_path)

    assert len(books) == 1
    assert (tmp_path / "index.html").exists()
    assert (book_dir / "index.html").exists()
    landing = (tmp_path / "index.html").read_text(encoding="utf-8")
    detail = (book_dir / "index.html").read_text(encoding="utf-8")
    assert "Book maps and recordings" in landing
    assert "[hidden] { display: none !important; }" in landing
    assert 'type="search"' in landing
    assert "1 book" in landing
    assert "0 papers" in landing
    assert 'data-filter="Other"' in landing
    assert 'data-filter="Book"' in landing
    assert 'data-sort' in landing
    assert 'data-added-date="2026-05-01"' in landing
    assert "data-book-card" in landing
    assert "Sample Book" in landing
    assert "sample_book/index.html" in landing
    assert "Open visual summary" in detail
    assert "sample_book_visual_summary_en.html" in detail
    assert "5-minute summary - English" in detail
    assert "20-minute podcast - Czech" in detail
    assert 'preload="none"' in detail
    assert "Added 01 May 2026" in detail
    assert "Other" in detail
    assert "Book" in detail


def test_discover_books_skips_directories_without_publishable_assets(tmp_path: Path):
    ignored_dir = tmp_path / "notes_only"
    ignored_dir.mkdir()
    (ignored_dir / "notes_only_source_raw.md").write_text("# Notes", encoding="utf-8")

    assert discover_books(tmp_path) == []


def test_discover_books_infers_topic_labels(tmp_path: Path):
    book_dir = tmp_path / "ai_brain_book"
    book_dir.mkdir()
    (book_dir / "ai_brain_book_visual_summary_en.html").write_text(
        "<html><body><h1>AI and the Brain</h1><p class=\"summary\">Artificial intelligence and neuroscience.</p></body></html>",
        encoding="utf-8",
    )

    books = discover_books(tmp_path)

    assert books[0].document_type == "book"
    assert books[0].labels == ("Book", "AI")


def test_discover_books_marks_arxiv_papers(tmp_path: Path):
    book_dir = tmp_path / "hybrid_dynamic_routing"
    book_dir.mkdir()
    (book_dir / "hybrid_dynamic_routing_visual_summary_en.html").write_text(
        "<html><body><h1>Hybrid Dynamic Routing</h1><p class=\"summary\">Routing for heterogeneous LLM pools.</p></body></html>",
        encoding="utf-8",
    )
    (book_dir / "hybrid_dynamic_routing_source_raw.md").write_text(
        "# Paper\n\narXiv:2601.12345\n\nThis paper studies AI routing.",
        encoding="utf-8",
    )
    (book_dir / "metadata.yaml").write_text(
        "source_path: arxiv/Hybrid Dynamic Routing.pdf\n"
        "document_type: paper\n"
        "source_medium: PDF\n"
        "added_date: 2026-07-07\n"
        "labels:\n"
        "- AI\n"
        "- computers\n",
        encoding="utf-8",
    )

    books = discover_books(tmp_path)

    assert books[0].document_type == "paper"
    assert books[0].labels == ("Paper", "AI")
    assert books[0].added_date == "2026-07-07"
