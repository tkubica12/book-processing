"""Normalize input sources into per-book raw Markdown files."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from markitdown import MarkItDown
import pymupdf

from book_processing.content_understanding import (
    ContentUnderstandingNoUsableMarkdownError,
    analyze_image_to_markdown,
    analyze_pdf_to_markdown,
)
from book_processing.config import (
    BOOK_MAX_WORKERS,
    INPUT_DIR,
    OUTPUT_DIR,
    SOURCE_RAW_NAME,
    book_name_from_source,
    book_name_from_pdf,
    output_text_path,
)

logger = logging.getLogger(__name__)


def find_source_files(input_dir: Path = INPUT_DIR) -> list[Path]:
    """Find all supported source files in the input directory, sorted alphabetically."""
    sources = sorted([*input_dir.glob("*.epub"), *input_dir.glob("*.md"), *input_dir.glob("*.pdf")])
    logger.info("Found %d source file(s) in %s", len(sources), input_dir)
    return sources


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a single PDF file to Markdown text."""
    logger.info("Converting %s to Markdown...", pdf_path.name)
    try:
        text = analyze_pdf_to_markdown(pdf_path)
    except ContentUnderstandingNoUsableMarkdownError:
        logger.warning(
            "Content Understanding returned no usable PDF markdown for %s; retrying via rendered page images",
            pdf_path.name,
        )
        try:
            text = _convert_pdf_pages_to_markdown(pdf_path)
        except ContentUnderstandingNoUsableMarkdownError:
            logger.warning(
                "Rendered page fallback also returned no usable markdown for %s; using local PDF text extraction",
                pdf_path.name,
            )
            text = _extract_pdf_text_locally(pdf_path)
    logger.info("Converted %s — %d characters", pdf_path.name, len(text))
    return text


def convert_epub_to_markdown(epub_path: Path) -> str:
    """Convert a single EPUB file to Markdown text."""
    logger.info("Converting %s to Markdown via MarkItDown...", epub_path.name)
    result = MarkItDown().convert(str(epub_path))
    markdown = result.text_content.strip()
    if not markdown:
        raise RuntimeError(f"MarkItDown returned no markdown for {epub_path.name}")
    logger.info("Converted %s via MarkItDown — %d characters", epub_path.name, len(markdown))
    return markdown


def _render_pdf_pages_to_png(pdf_path: Path) -> list[tuple[str, bytes]]:
    """Render PDF pages to PNG bytes for image-based fallback analysis."""
    page_images: list[tuple[str, bytes]] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=150, alpha=False)
            page_images.append((f"{pdf_path.stem}_page_{page_number}.png", pixmap.tobytes("png")))
    return page_images


def _convert_pdf_pages_to_markdown(pdf_path: Path) -> str:
    """Retry PDF conversion by rendering each page to an image for analysis."""
    page_markdown: list[str] = []
    for image_name, image_bytes in _render_pdf_pages_to_png(pdf_path):
        page_markdown.append(analyze_image_to_markdown(image_name, image_bytes).strip())

    combined_markdown = "\n\n".join(markdown for markdown in page_markdown if markdown)
    if not combined_markdown:
        raise ContentUnderstandingNoUsableMarkdownError(
            f"Rendered page fallback returned no markdown for {pdf_path.name}"
        )
    return combined_markdown


def _extract_pdf_text_locally(pdf_path: Path) -> str:
    """Extract plain text from a PDF locally as a last-resort fallback."""
    pages: list[str] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append(f"## Page {page_number}\n\n{page_text}")

    combined_text = "\n\n".join(pages).strip()
    if not combined_text:
        raise RuntimeError(f"Local PDF text extraction returned no text for {pdf_path.name}")
    return combined_text


def clean_raw_markdown(text: str) -> str:
    """Post-process Content Understanding Markdown for downstream use.

    Keeps the rich Markdown structure intact while removing noisy page metadata
    comments and collapsing excess blank lines.
    """
    text = re.sub(r"<!--\s*Page(?:Number|Header|Footer)=.*?-->", "", text)
    text = re.sub(r"<!--\s*PageBreak\s*-->", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    logger.info("Cleaned markdown: removed page metadata comments and collapsed blank lines")
    return text.strip()


def validate_unique_book_names(source_paths: list[Path]) -> None:
    """Ensure source files map to unique sanitized book names."""
    seen: dict[str, Path] = {}
    for source_path in source_paths:
        book_name = book_name_from_source(source_path)
        previous = seen.get(book_name)
        if previous is not None:
            raise ValueError(
                "Multiple input files resolve to the same book name "
                f"'{book_name}': {previous.name} and {source_path.name}"
            )
        seen[book_name] = source_path


def _raw_output_path(book_name: str, output_dir: Path) -> Path:
    """Return the per-book source_raw path for one normalized source."""
    return output_text_path(book_name, SOURCE_RAW_NAME, output_dir=output_dir)


def _process_pdf(pdf_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Convert, clean, and save one PDF as its own raw markdown file."""
    book_name = book_name_from_pdf(pdf_path)
    md_text = convert_pdf_to_markdown(pdf_path)
    cleaned_md = clean_raw_markdown(md_text)
    output_path = _raw_output_path(book_name, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_md, encoding="utf-8")
    logger.info("Saved raw Markdown for %s to %s (%d chars)", book_name, output_path, len(cleaned_md))
    return book_name, output_path


def _process_markdown(md_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Copy one Markdown source into the standard raw-markdown output location."""
    book_name = book_name_from_source(md_path)
    md_text = md_path.read_text(encoding="utf-8")
    output_path = _raw_output_path(book_name, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Copied Markdown source for %s to %s (%d chars)", book_name, output_path, len(md_text))
    return book_name, output_path


def _process_epub(epub_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Convert one EPUB source into the standard raw-markdown output location."""
    book_name = book_name_from_source(epub_path)
    md_text = convert_epub_to_markdown(epub_path)
    output_path = _raw_output_path(book_name, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Converted EPUB source for %s to %s (%d chars)", book_name, output_path, len(md_text))
    return book_name, output_path


def _process_source(source_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Normalize one supported source file into the raw-markdown output location."""
    suffix = source_path.suffix.lower()
    if suffix == ".epub":
        return _process_epub(source_path, output_dir)
    if suffix == ".pdf":
        return _process_pdf(source_path, output_dir)
    if suffix == ".md":
        return _process_markdown(source_path, output_dir)
    raise ValueError(f"Unsupported source file type: {source_path}")


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Normalize all supported input files into per-book raw-markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = find_source_files(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No supported input files found in {input_dir}")
    validate_unique_book_names(source_files)

    outputs: dict[str, Path] = {}
    max_workers = min(BOOK_MAX_WORKERS, len(source_files))
    logger.info("Processing %d source file(s) as independent books with %d workers", len(source_files), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_source, source_path, output_dir): source_path
            for source_path in source_files
        }
        for future in as_completed(futures):
            book_name, output_path = future.result()
            outputs[book_name] = output_path

    logger.info("Stage 1 produced %d per-book raw markdown file(s)", len(outputs))
    return outputs
