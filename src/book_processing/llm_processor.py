"""LLM-powered summary generation and TTS preprocessing using Azure OpenAI.

All tasks are flattened into atomic units and run in a single ThreadPoolExecutor
for maximum parallelism. Multi-part outputs (podcasts, TTS sources) are assembled
as their constituent parts complete.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from book_processing.config import (
    AZURE_COGNITIVE_SCOPE,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL,
    LANGUAGES,
    LLM_MAX_WORKERS,
    OUTPUT_DIR,
    PODCAST_SPEAKERS,
    SOURCE_TTS_NAME,
    SUMMARY_TYPES,
    output_text_path,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 8

# Callback signature: (content_name, lang, path, is_podcast)
FileReadyCallback = Callable[[str, str, Path, bool], None]


def _get_client() -> AzureOpenAI:
    """Create an Azure OpenAI client with Entra authentication."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        AZURE_COGNITIVE_SCOPE,
    )
    return AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        timeout=1200,
        max_retries=0,
    )


def _call_llm(client: AzureOpenAI, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
    """Make a single chat completion call with retry logic."""
    effective_max = max(max_tokens, 8000)
    logger.info("Calling LLM (max_completion_tokens=%d)...", effective_max)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=effective_max,
                temperature=0.7,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            finish = choice.finish_reason
            usage = response.usage
            logger.info(
                "LLM response: %d chars, ~%d words (finish=%s, prompt_tokens=%s, completion_tokens=%s)",
                len(content), len(content.split()), finish,
                getattr(usage, 'prompt_tokens', '?'),
                getattr(usage, 'completion_tokens', '?'),
            )
            if not content and finish == "length":
                logger.warning("Response was empty - likely ran out of completion tokens.")
            return content
        except Exception as e:
            wait = 30 * (attempt + 1)
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Atomic task functions - each does exactly ONE LLM call
# ---------------------------------------------------------------------------

def _do_simple_summary(client: AzureOpenAI, source_md: str, summary_type: str, lang: str) -> str:
    """Generate a single summary (2min, 5min, or 20min)."""
    spec = SUMMARY_TYPES[summary_type]
    target_words = spec["target_words"]
    description = spec["description"]
    lang_label = LANGUAGES[lang]["label"]

    system_prompt = (
        f"You are a technical writer creating content for a senior software architect. "
        f"Write in {lang_label}. "
        f"Be deeply technical - cover architectural principles, design patterns, technology trade-offs, "
        f"and implementation details. The reader is not afraid of complexity. "
        f"Write in a clear, flowing style suitable for reading aloud. "
        f"Do not use bullet points or lists - use well-structured paragraphs. "
        f"Do not include any markdown formatting, headers, or special characters. "
        f"IMPORTANT: Keep your output to approximately {target_words} words. Do not exceed this significantly."
    )
    user_prompt = (
        f"Create a {description.lower()} of approximately {target_words} words "
        f"based on the following source material.\n\n"
        f"SOURCE MATERIAL:\n\n{source_md[:40000]}"
    )
    return _call_llm(client, system_prompt, user_prompt, max_tokens=8000)


def _do_podcast_section(
    client: AzureOpenAI, section_text: str,
    section_idx: int, total_sections: int, lang: str,
) -> str:
    """Generate a single podcast section with crash recovery via partial files."""
    speakers = PODCAST_SPEAKERS[lang]
    lang_label = LANGUAGES[lang]["label"]
    target_words = SUMMARY_TYPES["podcast_60min"]["target_words"]
    words_per_section = target_words // total_sections
    section_num = section_idx + 1

    partial_dir = OUTPUT_DIR / "_partial"
    partial_dir.mkdir(exist_ok=True)
    partial_path = partial_dir / f"podcast_60min_{lang}_section{section_num}.md"

    if partial_path.exists() and partial_path.stat().st_size > 100:
        logger.info("Loading cached podcast section %d/%d for %s", section_num, total_sections, lang)
        return partial_path.read_text(encoding="utf-8")

    system_prompt = (
        f"You are a scriptwriter for a highly technical podcast. "
        f"Write in {lang_label}. "
        f"The podcast has two hosts: {speakers['male']} (male, lead host) and {speakers['female']} (female, co-host). "
        f"They are both deeply technical, geeky, and passionate about software architecture and engineering. "
        f"The conversation should be entertaining, fun, and sometimes geeky - use humor, analogies, "
        f"and show genuine excitement - but always remain accurate and deeply technical. "
        f"Format each line as [{speakers['male']}]: or [{speakers['female']}]: followed by the spoken text. "
        f"No stage directions, no parenthetical notes - only spoken dialogue. "
        f"Cover the material comprehensively - principles, architecture, deep technical details. "
        f"CRITICAL: Strictly respect the word count limit given in each prompt."
    )

    if section_num == 1:
        intro = (
            f"Create part {section_num} of {total_sections} of a technical podcast. "
            f"This is the OPENING segment - include a welcoming intro. "
        )
    elif section_num == total_sections:
        intro = (
            f"Create part {section_num} of {total_sections} of a technical podcast. "
            f"This is the CLOSING segment - wrap up and say goodbye to listeners. "
        )
    else:
        intro = (
            f"Create part {section_num} of {total_sections} of a technical podcast. "
            f"Jump straight into the content - no intro needed. "
        )

    user_prompt = (
        f"{intro}"
        f"Write EXACTLY approximately {words_per_section} words of dialogue. "
        f"Do NOT exceed {int(words_per_section * 1.2)} words. "
        f"Cover the key topics in the source material below.\n\n"
        f"SOURCE MATERIAL:\n\n{section_text}"
    )
    logger.info("Generating podcast section %d/%d in %s (~%d words)...",
                section_num, total_sections, lang_label, words_per_section)
    result = _call_llm(client, system_prompt, user_prompt, max_tokens=8000)
    if result:
        partial_path.write_text(result, encoding="utf-8")
    return result


def _do_tts_chunk(
    client: AzureOpenAI, chunk_text: str,
    chunk_idx: int, total_chunks: int, lang: str,
) -> str:
    """Process a single TTS source chunk with crash recovery."""
    lang_label = LANGUAGES[lang]["label"]
    chunk_num = chunk_idx + 1

    partial_dir = OUTPUT_DIR / "_partial"
    partial_dir.mkdir(exist_ok=True)
    partial_path = partial_dir / f"source_tts_{lang}_chunk{chunk_num}.md"

    if partial_path.exists() and partial_path.stat().st_size > 100:
        logger.info("Loading cached TTS chunk %d/%d for %s", chunk_num, total_chunks, lang)
        return partial_path.read_text(encoding="utf-8")

    system_prompt = (
        f"You are preprocessing a technical book for text-to-speech narration. "
        f"Output language: {lang_label}. "
        f"Your task is to transform the text so it sounds natural when read aloud by a human narrator. "
        f"Rules:\n"
        f"- Remove all page numbers, page headers/footers, and pagination markers\n"
        f"- Remove all URLs and email addresses (mention the service/site name instead)\n"
        f"- For TABLES: Do NOT read tables row by row. Instead, briefly describe what the table shows "
        f"and highlight the 2-3 most notable or interesting data points conversationally. "
        f"For example: 'The table compares GPU specifications. Notably, the H100 delivers 1979 teraFLOPS "
        f"with 80GB of memory, while the newer B200 nearly doubles that at 4500 teraFLOPS.'\n"
        f"- For FIGURES and DIAGRAMS: Describe the key insight the figure illustrates, "
        f"don't just read the caption\n"
        f"- Remove all markdown formatting (headers, bold, italic, code blocks)\n"
        f"- Expand abbreviations on first use\n"
        f"- Remove reference numbers and footnote markers\n"
        f"- Keep ALL technical content - do not summarize or omit anything\n"
        f"- Make it flow naturally as continuous narration\n"
        f"- Skip table of contents and copyright/legal notices\n"
        f"{'- Translate to Czech while preserving all technical meaning' if lang == 'cs' else ''}"
    )

    logger.info("Processing TTS source chunk %d/%d for lang=%s...", chunk_num, total_chunks, lang)
    user_prompt = (
        f"Preprocess the following section of the book for TTS narration. "
        f"This is section {chunk_num} of {total_chunks}.\n\n"
        f"TEXT:\n\n{chunk_text}"
    )
    result = _call_llm(client, system_prompt, user_prompt, max_tokens=16000)
    if result:
        partial_path.write_text(result, encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Orchestration - all tasks in a single pool
# ---------------------------------------------------------------------------

def run(
    source_md_path: Path,
    output_dir: Path = OUTPUT_DIR,
    on_file_ready: FileReadyCallback | None = None,
) -> dict[str, Path]:
    """Run all LLM tasks in a single parallel pool for maximum throughput.

    Every atomic unit of work (each summary, each podcast section, each TTS
    chunk) is submitted independently.  Multi-part outputs are assembled as
    their constituents complete.

    Args:
        source_md_path: Path to the raw markdown source.
        output_dir: Directory for output files.
        on_file_ready: Optional callback invoked when a complete text file
            is written, enabling overlapped TTS submission.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    source_md = source_md_path.read_text(encoding="utf-8")
    client = _get_client()
    outputs: dict[str, Path] = {}

    # Prepare podcast sections
    source_trimmed = source_md[:120000]
    section_size = 20000
    podcast_sections = [source_trimmed[i:i + section_size]
                        for i in range(0, len(source_trimmed), section_size)]

    # Prepare TTS chunks
    chunk_size = 20000
    tts_chunks = [source_md[i:i + chunk_size]
                  for i in range(0, len(source_md), chunk_size)]

    # Build ALL atomic tasks
    all_tasks: list[dict] = []
    podcast_needed: dict[str, int] = {}   # lang -> total sections needed
    tts_needed: dict[str, int] = {}       # lang -> total chunks needed

    for stype, spec in SUMMARY_TYPES.items():
        if spec["is_podcast"]:
            continue
        for lang in LANGUAGES:
            path = output_text_path(stype, lang)
            key = f"{stype}_{lang}"
            if path.exists() and path.stat().st_size > 100:
                logger.info("Skipping %s (exists, %d bytes)", path.name, path.stat().st_size)
                outputs[key] = path
                continue
            all_tasks.append({
                "type": "simple_summary",
                "summary_type": stype,
                "lang": lang,
                "path": path,
                "key": key,
            })

    for lang in LANGUAGES:
        path = output_text_path("podcast_60min", lang)
        key = f"podcast_60min_{lang}"
        if path.exists() and path.stat().st_size > 100:
            logger.info("Skipping %s (exists, %d bytes)", path.name, path.stat().st_size)
            outputs[key] = path
            continue
        podcast_needed[lang] = len(podcast_sections)
        for idx in range(len(podcast_sections)):
            all_tasks.append({
                "type": "podcast_section",
                "section_idx": idx,
                "total_sections": len(podcast_sections),
                "section_text": podcast_sections[idx],
                "lang": lang,
                "key": f"podcast_{lang}_s{idx + 1}",
            })

    for lang in LANGUAGES:
        path = output_text_path(SOURCE_TTS_NAME, lang)
        key = f"{SOURCE_TTS_NAME}_{lang}"
        if path.exists() and path.stat().st_size > 100:
            logger.info("Skipping %s (exists, %d bytes)", path.name, path.stat().st_size)
            outputs[key] = path
            continue
        tts_needed[lang] = len(tts_chunks)
        for idx in range(len(tts_chunks)):
            all_tasks.append({
                "type": "tts_chunk",
                "chunk_idx": idx,
                "total_chunks": len(tts_chunks),
                "chunk_text": tts_chunks[idx],
                "lang": lang,
                "key": f"tts_{lang}_c{idx + 1}",
            })

    if not all_tasks:
        logger.info("All text files already exist - nothing to do")
        return outputs

    logger.info("Submitting %d atomic LLM tasks to pool of %d workers", len(all_tasks), LLM_MAX_WORKERS)

    # Thread-safe assembly tracking
    lock = threading.Lock()
    podcast_parts: dict[str, dict[int, str]] = {lang: {} for lang in podcast_needed}
    tts_parts: dict[str, dict[int, str]] = {lang: {} for lang in tts_needed}

    def _handle_result(task: dict, text: str) -> None:
        """Process a completed task, assembling multi-part outputs as ready."""
        ttype = task["type"]

        if ttype == "simple_summary":
            task["path"].write_text(text, encoding="utf-8")
            key = task["key"]
            with lock:
                outputs[key] = task["path"]
            logger.info("-> %s (%d words)", key, len(text.split()))
            if on_file_ready:
                on_file_ready(task["summary_type"], task["lang"], task["path"], False)

        elif ttype == "podcast_section":
            lang = task["lang"]
            total = task["total_sections"]
            with lock:
                podcast_parts[lang][task["section_idx"]] = text
                done_count = len(podcast_parts[lang])
            logger.info("-> %s (%d words) [%d/%d sections]",
                        task["key"], len(text.split()), done_count, total)
            if done_count == total:
                assembled = "\n\n".join(podcast_parts[lang][i] for i in range(total))
                path = output_text_path("podcast_60min", lang)
                path.write_text(assembled, encoding="utf-8")
                key = f"podcast_60min_{lang}"
                with lock:
                    outputs[key] = path
                logger.info("=> Assembled %s (%d words from %d sections)",
                            key, len(assembled.split()), total)
                if on_file_ready:
                    on_file_ready("podcast_60min", lang, path, True)

        elif ttype == "tts_chunk":
            lang = task["lang"]
            total = task["total_chunks"]
            with lock:
                tts_parts[lang][task["chunk_idx"]] = text
                done_count = len(tts_parts[lang])
            logger.info("-> %s (%d words) [%d/%d chunks]",
                        task["key"], len(text.split()), done_count, total)
            if done_count == total:
                assembled = "\n\n".join(tts_parts[lang][i] for i in range(total))
                path = output_text_path(SOURCE_TTS_NAME, lang)
                path.write_text(assembled, encoding="utf-8")
                key = f"{SOURCE_TTS_NAME}_{lang}"
                with lock:
                    outputs[key] = path
                logger.info("=> Assembled %s (%d words from %d chunks)",
                            key, len(assembled.split()), total)
                if on_file_ready:
                    on_file_ready(SOURCE_TTS_NAME, lang, path, False)

    # Execute all tasks in a single pool
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as pool:
        futures: dict = {}
        for task in all_tasks:
            if task["type"] == "simple_summary":
                future = pool.submit(
                    _do_simple_summary, client, source_md,
                    task["summary_type"], task["lang"],
                )
            elif task["type"] == "podcast_section":
                future = pool.submit(
                    _do_podcast_section, client, task["section_text"],
                    task["section_idx"], task["total_sections"], task["lang"],
                )
            elif task["type"] == "tts_chunk":
                future = pool.submit(
                    _do_tts_chunk, client, task["chunk_text"],
                    task["chunk_idx"], task["total_chunks"], task["lang"],
                )
            futures[future] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                text = future.result()
                _handle_result(task, text)
            except Exception as e:
                logger.error("FAILED: %s - %s", task["key"], e)
                raise

    return outputs
