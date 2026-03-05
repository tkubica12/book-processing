"""Convert PDF files to Markdown using markitdown."""

import logging
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

    output_path = output_dir / f"{SOURCE_RAW_NAME}.md"
    output_path.write_text(combined_md, encoding="utf-8")
    logger.info("Saved raw Markdown to %s", output_path)
    return output_path
