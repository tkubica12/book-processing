"""Normalize input sources into per-book raw Markdown files."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
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
    sanitize_book_name,
    wiki_text_path,
)

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_SUFFIXES = (".m4b", ".mp3")
SUPPORTED_TEXT_SUFFIXES = (".md", ".txt")
SUPPORTED_SOURCE_SUFFIXES = (".epub", ".pdf", *SUPPORTED_TEXT_SUFFIXES, *SUPPORTED_AUDIO_SUFFIXES)
_NATURAL_SORT_PATTERN = re.compile(r"\d+|\D+")
_SKIPPED_AUDIO_TRACK_TEMPLATE = (
    "## Skipped audio track\n\n"
    "Could not transcribe `{audio_name}` because the source audio was invalid: {reason}"
)


def _natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Return a case-insensitive natural sort key that keeps numeric tracks ordered."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in _NATURAL_SORT_PATTERN.findall(value)
    )


def _audio_sort_key(audio_path: Path, source_dir: Path) -> tuple[tuple[tuple[int, int | str], ...], ...]:
    """Return a deterministic natural sort key for one audio path relative to its source directory."""
    return tuple(_natural_sort_key(part) for part in audio_path.relative_to(source_dir).parts)


def _find_audio_files_in_directory(source_dir: Path) -> list[Path]:
    """Return all supported audio files inside a source directory in deterministic order."""
    audio_files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(audio_files, key=lambda path: _audio_sort_key(path, source_dir))


def find_source_files(input_dir: Path = INPUT_DIR) -> list[Path]:
    """Find supported top-level sources, including audio folders, in deterministic order."""
    sources = sorted(
        (
            path
            for path in input_dir.iterdir()
            if (
                path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
            )
            or (
                path.is_dir() and bool(_find_audio_files_in_directory(path))
            )
        ),
        key=lambda path: _natural_sort_key(path.name),
    )
    logger.info("Found %d source item(s) in %s", len(sources), input_dir)
    return sources


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a single PDF file to Markdown text."""
    logger.info("Converting %s to Markdown...", pdf_path.name)
    try:
        text = analyze_pdf_to_markdown(pdf_path)
    except (ContentUnderstandingNoUsableMarkdownError, httpx.HTTPError, RuntimeError) as error:
        logger.warning(
            "Content Understanding PDF analysis failed for %s (%s); retrying via rendered page images",
            pdf_path.name,
            error,
        )
        try:
            text = _convert_pdf_pages_to_markdown(pdf_path)
        except (ContentUnderstandingNoUsableMarkdownError, httpx.HTTPError, RuntimeError) as fallback_error:
            logger.warning(
                "Rendered page fallback failed for %s (%s); using local PDF text extraction",
                pdf_path.name,
                fallback_error,
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


def convert_audio_to_markdown(
    audio_path: Path,
    output_dir: Path = OUTPUT_DIR,
    *,
    book_name: str | None = None,
    artifact_stem: str | None = None,
) -> str:
    """Convert a single audio file to Markdown text via the audio transcription module."""
    logger.info("Transcribing %s to Markdown...", audio_path.name)
    from book_processing.audio_transcriber import convert_audio_to_markdown as transcribe_audio_to_markdown

    markdown = transcribe_audio_to_markdown(
        audio_path,
        output_dir=output_dir,
        book_name=book_name,
        artifact_stem=artifact_stem,
    )
    if not markdown.strip():
        raise RuntimeError(f"Audio transcription returned no markdown for {audio_path.name}")
    logger.info("Transcribed %s — %d characters", audio_path.name, len(markdown))
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


def _write_stage1_markdown(book_name: str, markdown: str, output_dir: Path) -> Path:
    """Persist Stage 1 raw Markdown to both the per-book output and repo wiki copy."""
    output_path = _raw_output_path(book_name, output_dir)
    wiki_path = wiki_text_path(book_name, output_dir=output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wiki_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    wiki_path.write_text(markdown, encoding="utf-8")
    logger.info(
        "Saved Stage 1 raw Markdown for %s to %s and %s (%d chars)",
        book_name,
        output_path,
        wiki_path,
        len(markdown),
    )
    return output_path


def _process_pdf(pdf_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Convert, clean, and save one PDF as its own raw markdown file."""
    book_name = book_name_from_pdf(pdf_path)
    md_text = convert_pdf_to_markdown(pdf_path)
    cleaned_md = clean_raw_markdown(md_text)
    output_path = _write_stage1_markdown(book_name, cleaned_md, output_dir)
    return book_name, output_path


def _process_text_source(text_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Copy one Markdown or text source into the standard raw-markdown output location."""
    book_name = book_name_from_source(text_path)
    md_text = text_path.read_text(encoding="utf-8")
    output_path = _write_stage1_markdown(book_name, md_text, output_dir)
    return book_name, output_path


def _process_epub(epub_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Convert one EPUB source into the standard raw-markdown output location."""
    book_name = book_name_from_source(epub_path)
    md_text = convert_epub_to_markdown(epub_path)
    output_path = _write_stage1_markdown(book_name, md_text, output_dir)
    return book_name, output_path


def _process_audio(audio_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Transcribe one audio source into the standard raw-markdown output location."""
    book_name = book_name_from_source(audio_path)
    md_text = convert_audio_to_markdown(audio_path, output_dir=output_dir)
    output_path = _write_stage1_markdown(book_name, md_text, output_dir)
    return book_name, output_path


def _process_audio_directory(source_dir: Path, output_dir: Path) -> tuple[str, Path]:
    """Transcribe one audio directory into a single combined raw-markdown output."""
    from book_processing.audio_transcriber import InvalidAudioSourceError

    book_name = book_name_from_source(source_dir)
    audio_files = _find_audio_files_in_directory(source_dir)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {source_dir}")

    transcripts: list[str] = []
    skipped_audio_files: list[Path] = []
    for index, audio_path in enumerate(audio_files, start=1):
        relative_stem = sanitize_book_name(str(audio_path.relative_to(source_dir).with_suffix("")))
        artifact_stem = f"{book_name}_track{index:04d}_{relative_stem}"
        try:
            transcript = convert_audio_to_markdown(
                audio_path,
                output_dir=output_dir,
                book_name=book_name,
                artifact_stem=artifact_stem,
            )
        except InvalidAudioSourceError as error:
            logger.warning(
                "Skipping invalid audio track %s while processing %s: %s",
                audio_path,
                source_dir,
                error,
            )
            skipped_audio_files.append(audio_path)
            transcripts.append(
                _SKIPPED_AUDIO_TRACK_TEMPLATE.format(audio_name=audio_path.name, reason=error)
            )
            continue
        transcripts.append(transcript)

    md_text = "\n\n".join(text.strip() for text in transcripts if text.strip()).strip()
    if not md_text:
        raise RuntimeError(f"Audio directory transcription returned no markdown for {source_dir.name}")

    output_path = _write_stage1_markdown(book_name, md_text, output_dir)
    logger.info(
        "Transcribed audio directory source for %s from %d file(s) (%d chars)",
        book_name,
        len(audio_files),
        len(md_text),
    )
    if skipped_audio_files:
        logger.warning(
            "Skipped %d invalid audio track(s) while processing %s",
            len(skipped_audio_files),
            source_dir,
        )
    return book_name, output_path


def _process_source(source_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Normalize one supported source path into the raw-markdown output location."""
    book_name = book_name_from_source(source_path)
    existing_raw = _raw_output_path(book_name, output_dir)
    if existing_raw.exists() and existing_raw.stat().st_size > 0:
        wiki_path = wiki_text_path(book_name, output_dir=output_dir)
        if not wiki_path.exists():
            wiki_path.parent.mkdir(parents=True, exist_ok=True)
            wiki_path.write_text(existing_raw.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Wrote missing wiki copy for %s", book_name)
        logger.info("Skipping Stage 1 for %s — source_raw already exists (%d bytes)",
                    book_name, existing_raw.stat().st_size)
        return book_name, existing_raw

    if source_path.is_dir():
        return _process_audio_directory(source_path, output_dir)
    suffix = source_path.suffix.lower()
    if suffix == ".epub":
        return _process_epub(source_path, output_dir)
    if suffix == ".pdf":
        return _process_pdf(source_path, output_dir)
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return _process_text_source(source_path, output_dir)
    if suffix in SUPPORTED_AUDIO_SUFFIXES:
        return _process_audio(source_path, output_dir)
    raise ValueError(f"Unsupported source file type: {source_path}")


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Normalize all supported input sources into per-book raw-markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = find_source_files(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No supported input files found in {input_dir}")
    validate_unique_book_names(source_files)

    outputs: dict[str, Path] = {}
    max_workers = min(BOOK_MAX_WORKERS, len(source_files))
    logger.info("Processing %d source item(s) as independent books with %d workers", len(source_files), max_workers)
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
