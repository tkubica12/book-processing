"""Main CLI entry point - orchestrates the full book processing pipeline.

Stages 2 (LLM) and 3 (TTS) run overlapped: TTS jobs are submitted as soon
as each text file is ready, so audio synthesis happens concurrently with
remaining LLM work.
"""

import logging
import threading
import time
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

    Stage 1: PDF -> Markdown (with post-processing cleanup)
    Stage 2+3: LLM text generation and TTS audio generation run overlapped.
    """
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("Book Processing Pipeline")
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # --- Stage 1: PDF to Markdown ---
    logger.info("STAGE 1: PDF -> Markdown")
    t0 = time.time()
    from book_processing.pdf_converter import run as run_pdf

    source_md_path = run_pdf(input_dir, output_dir)
    logger.info("Stage 1 complete in %.0fs: %s", time.time() - t0, source_md_path)

    # --- Stage 2+3: Overlapped LLM + TTS ---
    logger.info("STAGE 2+3: LLM + TTS (overlapped)")
    t0 = time.time()

    from book_processing.llm_processor import run as run_llm
    from book_processing.tts_processor import TtsJobTracker

    # Create TTS tracker and start its polling loop in a background thread
    tracker = TtsJobTracker()
    tts_thread = threading.Thread(target=tracker.poll_loop, name="tts-poll", daemon=True)
    tts_thread.start()

    def on_file_ready(name: str, lang: str, path: Path, is_podcast: bool) -> None:
        """Callback: immediately queue text file for TTS synthesis."""
        logger.info("Text ready -> queuing TTS: %s_%s", name, lang)
        tracker.enqueue(name, lang, path, is_podcast)

    # Run LLM (blocking); each completed file triggers on_file_ready -> TTS
    text_outputs = run_llm(source_md_path, output_dir, on_file_ready=on_file_ready)
    llm_elapsed = time.time() - t0
    logger.info("LLM complete in %.0fs: %d text files", llm_elapsed, len(text_outputs))

    # Queue any text files that already existed (skipped by LLM) but have no audio
    from book_processing.config import SUMMARY_TYPES, LANGUAGES, SOURCE_TTS_NAME, output_audio_path, output_text_path
    for stype, spec in SUMMARY_TYPES.items():
        for lang in LANGUAGES:
            tp = output_text_path(stype, lang)
            ap = output_audio_path(stype, lang)
            if tp.exists() and not (ap.exists() and ap.stat().st_size > 1000):
                logger.info("Queuing existing text for TTS: %s_%s", stype, lang)
                tracker.enqueue(stype, lang, tp, spec["is_podcast"])
    for lang in LANGUAGES:
        tp = output_text_path(SOURCE_TTS_NAME, lang)
        ap = output_audio_path(SOURCE_TTS_NAME, lang)
        if tp.exists() and not (ap.exists() and ap.stat().st_size > 1000):
            logger.info("Queuing existing text for TTS: %s_%s", SOURCE_TTS_NAME, lang)
            tracker.enqueue(SOURCE_TTS_NAME, lang, tp, False)

    # Signal TTS that no more files are coming, wait for remaining jobs
    tracker.finalize()
    logger.info("Waiting for remaining TTS jobs to complete...")
    tracker.wait()
    tts_thread.join(timeout=10)
    audio_outputs = tracker.get_outputs()

    tts_elapsed = time.time() - t0 - llm_elapsed
    logger.info("TTS complete (%.0fs after LLM). %d audio files.", tts_elapsed, len(audio_outputs))

    for name, path in sorted(audio_outputs.items()):
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info("  %s: %.1f MB", name, size_mb)

    # --- Done ---
    total = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.0fs (%.1f min)!", total, total / 60)
    logger.info("Text files: %d | Audio files: %d", len(text_outputs), len(audio_outputs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
