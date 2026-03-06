"""Convert PDF files to Markdown using markitdown, with post-processing."""

import logging
import re
from pathlib import Path

from markitdown import MarkItDown

from book_processing.config import INPUT_DIR, OUTPUT_DIR, SOURCE_RAW_NAME

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


def convert_all_pdfs(input_dir: Path = INPUT_DIR) -> str:
    """Convert all PDFs in input_dir to Markdown and concatenate them.

    Returns the full concatenated Markdown string.
    """
    pdfs = find_pdfs(input_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    sections: list[str] = []
    for pdf in pdfs:
        md_text = convert_pdf_to_markdown(pdf)
        sections.append(md_text)

    combined = "\n\n---\n\n".join(sections)
    logger.info("Combined %d PDF(s) into %d characters of Markdown", len(pdfs), len(combined))
    return combined


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> Path:
    """Run the full PDF conversion stage. Returns path to the output file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_md = convert_all_pdfs(input_dir)
    cleaned_md = clean_raw_markdown(combined_md)

    output_path = output_dir / f"{SOURCE_RAW_NAME}.md"
    output_path.write_text(cleaned_md, encoding="utf-8")
    logger.info("Saved raw Markdown to %s (%d chars)", output_path, len(cleaned_md))
    return output_path
