"""Main CLI entry point — orchestrates the full book processing pipeline."""

import logging
import sys
from pathlib import Path

from book_processing.config import INPUT_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    """Run the complete book processing pipeline.

    Stage 1: PDF → Markdown
    Stage 2: LLM summaries and TTS preprocessing
    Stage 3: Text-to-Speech audio generation
    """
    logger.info("=" * 60)
    logger.info("Book Processing Pipeline")
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # --- Stage 1: PDF to Markdown ---
    logger.info("STAGE 1: PDF → Markdown")
    from book_processing.pdf_converter import run as run_pdf

    source_md_path = run_pdf(input_dir, output_dir)
    logger.info("Stage 1 complete: %s", source_md_path)

    # --- Stage 2: LLM Processing ---
    logger.info("STAGE 2: LLM Summaries & TTS Preprocessing")
    from book_processing.llm_processor import run as run_llm

    text_outputs = run_llm(source_md_path, output_dir)
    logger.info("Stage 2 complete: %d text files generated", len(text_outputs))
    for name, path in text_outputs.items():
        logger.info("  %s: %s", name, path.name)

    # --- Stage 3: Text-to-Speech ---
    logger.info("STAGE 3: Text-to-Speech Audio Generation")
    from book_processing.tts_processor import run as run_tts

    audio_outputs = run_tts(output_dir)
    logger.info("Stage 3 complete: %d audio files generated", len(audio_outputs))
    for name, path in audio_outputs.items():
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info("  %s: %s (%.1f MB)", name, path.name, size_mb)

    # --- Done ---
    logger.info("=" * 60)
    logger.info("Pipeline complete! All outputs in %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
