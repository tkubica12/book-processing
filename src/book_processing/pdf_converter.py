"""Convert PDF files to Markdown using markitdown, with post-processing."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from markitdown import MarkItDown

from book_processing.config import (
    BOOK_MAX_WORKERS,
    INPUT_DIR,
    OUTPUT_DIR,
    SOURCE_RAW_NAME,
    book_name_from_pdf,
    output_text_path,
)

logger = logging.getLogger(__name__)


def find_pdfs(input_dir: Path = INPUT_DIR) -> list[Path]:
    """Find all PDF files in the input directory, sorted alphabetically."""
    pdfs = sorted(input_dir.glob("*.pdf"))
    logger.info("Found %d PDF(s) in %s", len(pdfs), input_dir)
    return pdfs


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a single PDF file to Markdown text."""
    logger.info("Converting %s to Markdown...", pdf_path.name)
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    text = result.text_content
    logger.info("Converted %s — %d characters", pdf_path.name, len(text))
    return text


def clean_raw_markdown(text: str) -> str:
    """Post-process markitdown output to produce clean Markdown.

    Fixes common PDF-to-markdown artifacts:
    - Removes front matter (title page, copyright, ISBN)
    - Removes Table of Contents (dotted page number lines)
    - Removes inline page headers/footers ("80  Chapter 3: Hardware")
    - Fixes broken section headings split across lines
    - Collapses excessive blank lines
    """
    lines = text.split("\n")

    # --- Phase 1: Find and remove TOC block ---
    # TOC lines have patterns like "Preface ...........9" or "Chapter 0: Inference ...15"
    toc_start = None
    toc_end = None
    for i, line in enumerate(lines):
        if re.match(r"^Table of Contents\s*$", line.strip()):
            toc_start = i
        if toc_start is not None and re.search(r"\.{3,}\s*\d+\s*$", line):
            toc_end = i
    # Also catch "X  Table of Contents" page header repeats
    toc_pattern = re.compile(r"^\d+\s+Table of Contents\s*$|^Table of Contents\s+\d+\s*$")

    # --- Phase 2: Remove front matter (before first chapter/preface heading) ---
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for the first real chapter heading or "Preface" as standalone heading
        if re.match(r"^(Preface|Chapter\s+\d+)", stripped) and not re.search(r"\.{3,}", stripped):
            content_start = i
            break

    # --- Phase 3: Build cleaned output ---
    # Page header/footer patterns: "80  Chapter 3: Hardware" or "Chapter 3: Hardware  80"
    page_header_re = re.compile(
        r"^\d{1,3}\s{2,}(Chapter\s+\d+|Preface|Table of Contents)"
        r"|^(Chapter\s+\d+|Preface|Table of Contents)[:\s].*\s{2,}\d{1,3}\s*$"
    )

    cleaned: list[str] = []
    for i, line in enumerate(lines):
        # Skip everything before content start
        if i < content_start:
            continue
        # Skip TOC block
        if toc_start is not None and toc_start <= i <= (toc_end or toc_start):
            continue
        # Skip TOC page header repeats
        if toc_pattern.match(line.strip()):
            continue
        # Skip page headers/footers
        if page_header_re.match(line.strip()):
            continue
        # Skip standalone page numbers (just a number on its own line)
        if re.match(r"^\s*\d{1,3}\s*$", line) and i > 0:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # --- Phase 4: Fix broken headings ---
    # Pattern: "2.3\n\nImage Generation Inference Mechanics" → "## 2.3 Image Generation..."
    text = re.sub(
        r"\n(\d+\.\d+(?:\.\d+)?)\s*\n\n\s*([A-Z])",
        r"\n\1 \2",
        text,
    )

    # --- Phase 5: Collapse excessive blank lines ---
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # --- Phase 6: Convert section headings to proper markdown ---
    # "Chapter X: Title" → "# Chapter X: Title"
    text = re.sub(r"^(Chapter\s+\d+:.*)$", r"# \1", text, flags=re.MULTILINE)
    # "X.Y  Title" (section) → "## X.Y Title"
    text = re.sub(r"^(\d+\.\d+)\s{2,}(.+)$", r"## \1 \2", text, flags=re.MULTILINE)
    # "X.Y.Z  Title" (subsection) → "### X.Y.Z Title"
    text = re.sub(r"^(\d+\.\d+\.\d+)\s{2,}(.+)$", r"### \1 \2", text, flags=re.MULTILINE)
    # "Preface" → "# Preface"
    text = re.sub(r"^(Preface)\s*$", r"# \1", text, flags=re.MULTILINE)

    logger.info("Cleaned markdown: removed front matter and TOC, fixed headings")
    return text.strip()


def _process_pdf(pdf_path: Path, output_dir: Path) -> tuple[str, Path]:
    """Convert, clean, and save one PDF as its own raw markdown file."""
    book_name = book_name_from_pdf(pdf_path)
    md_text = convert_pdf_to_markdown(pdf_path)
    cleaned_md = clean_raw_markdown(md_text)
    output_path = output_text_path(book_name, SOURCE_RAW_NAME)
    output_path.write_text(cleaned_md, encoding="utf-8")
    logger.info("Saved raw Markdown for %s to %s (%d chars)", book_name, output_path, len(cleaned_md))
    return book_name, output_path


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Run the PDF conversion stage for all books, returning per-book raw paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = find_pdfs(input_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    outputs: dict[str, Path] = {}
    max_workers = min(BOOK_MAX_WORKERS, len(pdfs))
    logger.info("Processing %d PDF(s) as independent books with %d workers", len(pdfs), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_pdf, pdf, output_dir): pdf for pdf in pdfs}
        for future in as_completed(futures):
            book_name, output_path = future.result()
            outputs[book_name] = output_path

    logger.info("Stage 1 produced %d per-book raw markdown file(s)", len(outputs))
    return outputs
