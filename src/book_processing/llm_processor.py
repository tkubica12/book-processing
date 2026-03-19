"""LLM-powered summary generation and TTS preprocessing using Azure OpenAI.

All tasks are flattened into atomic units and run in a single ThreadPoolExecutor
for maximum parallelism. Multi-part outputs (podcasts, TTS sources) are assembled
as their constituent parts complete.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from openai import AzureOpenAI

from book_processing.auth import get_cognitive_token
from book_processing.config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL,
    LANGUAGES,
    LLM_MAX_WORKERS,
    OUTPUT_DIR,
    PODCAST_SPEAKERS,
    SOURCE_TTS_NAME,
    SUMMARY_TYPES,
    book_output_dir,
    output_text_path,
)
from book_processing.prompt_templates import render_prompt

logger = logging.getLogger(__name__)


class ContentFilterError(RuntimeError):
    """Raised when Azure OpenAI rejects a prompt due to content filtering."""


class LlmRequestTimeoutError(RuntimeError):
    """Raised when an LLM request times out and should be adaptively split."""


MAX_RETRIES = 8
LLM_REQUEST_TIMEOUT_SECONDS = 300
FILTER_RECOVERY_MAX_DEPTH = 6
FILTER_RECOVERY_MIN_CHARS = 400
FILTER_RECOVERY_MIN_WORDS = 60
FILTER_REDACTIONS = {
    r"\bsex(?:ual|ually)?\b": "intimate",
    r"\bporn(?:ography)?\b": "adult material",
    r"\berotic\b": "suggestive",
    r"\bnud(?:e|ity)\b": "undressed",
    r"\bgenitals?\b": "body parts",
    r"\bbreasts?\b": "chest",
    r"\bpenis\b": "body part",
    r"\bvagina\b": "body part",
    r"\borgasm(?:ic)?\b": "climax",
    r"\bmasturbat(?:e|es|ed|ing|ion)\b": "self-stimulate",
    r"\bincest\b": "abuse",
    r"\brape\b": "assault",
}

# Callback signature: (book_name, content_name, lang, path, is_podcast)
FileReadyCallback = Callable[[str, str, str, Path, bool], None]


def _get_client() -> AzureOpenAI:
    """Create an Azure OpenAI client with Entra authentication."""
    return AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=get_cognitive_token,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _is_content_filter_error(error: Exception) -> bool:
    """Return True when Azure rejected the prompt due to content filtering."""
    text = str(error)
    return "content_filter" in text or "ResponsibleAIPolicyViolation" in text


def _sanitize_filtered_prompt(prompt: str) -> str:
    """Replace likely trigger terms with neutral wording for Azure prompt retries."""
    sanitized = prompt
    for pattern, replacement in FILTER_REDACTIONS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _is_timeout_error(error: Exception) -> bool:
    """Return True when the client timed out waiting for Azure OpenAI."""
    return "timed out" in str(error).lower()


def _split_parts_near_half(parts: list[str], joiner: str) -> tuple[str, str] | None:
    """Split a sequence into two roughly balanced, non-empty halves."""
    if len(parts) < 2:
        return None

    total_length = sum(len(part) for part in parts)
    best_index = None
    best_diff = None
    running_length = 0
    for index, part in enumerate(parts[:-1], start=1):
        running_length += len(part)
        diff = abs((total_length - running_length) - running_length)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_index = index

    if best_index is None:
        return None

    left = joiner.join(parts[:best_index]).strip()
    right = joiner.join(parts[best_index:]).strip()
    if not left or not right:
        return None
    return left, right


def _split_text_for_filter_recovery(text: str) -> tuple[str, str] | None:
    """Split text for adaptive content-filter recovery, preferring natural boundaries."""
    normalized = text.strip()
    if not normalized:
        return None

    paragraph_parts = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    paragraph_split = _split_parts_near_half(paragraph_parts, "\n\n")
    if paragraph_split is not None:
        return paragraph_split

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    sentence_split = _split_parts_near_half(sentence_parts, " ")
    if sentence_split is not None:
        return sentence_split

    word_parts = normalized.split()
    return _split_parts_near_half(word_parts, " ")


def _filtered_fragment_placeholder(lang: str) -> str:
    """Return a spoken placeholder for a tiny fragment omitted due to content filtering."""
    if lang == "cs":
        return "Zde je vynechána krátká pasáž kvůli bezpečnostnímu filtrování obsahu."
    return "A short passage is omitted here due to content safety filtering."


def _recover_filtered_text(
    client: AzureOpenAI,
    system_prompt: str,
    render_user_prompt: Callable[[str], str],
    source_text: str,
    lang: str,
    partial_dir: Path,
    cache_prefix: str,
    max_tokens: int,
    depth: int = 0,
    fragment_path: str = "root",
) -> str:
    """Recursively recover from content-filtered prompts by splitting source text."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    cache_path = partial_dir / f"{cache_prefix}_recovery_{fragment_path}.md"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")

    user_prompt = render_user_prompt(source_text)
    try:
        result = _call_llm(
            client,
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            split_on_timeout=True,
        )
    except (ContentFilterError, LlmRequestTimeoutError) as error:
        is_timeout = isinstance(error, LlmRequestTimeoutError)
        split_text = _split_text_for_filter_recovery(source_text)
        is_irreducible = (
            depth >= FILTER_RECOVERY_MAX_DEPTH
            or len(source_text) <= FILTER_RECOVERY_MIN_CHARS
            or len(source_text.split()) <= FILTER_RECOVERY_MIN_WORDS
            or split_text is None
        )
        if is_irreducible:
            if is_timeout:
                raise
            placeholder = _filtered_fragment_placeholder(lang)
            logger.warning(
                "Content filter persisted for %s fragment %s; inserting placeholder (%d chars, ~%d words).",
                cache_prefix,
                fragment_path,
                len(source_text),
                len(source_text.split()),
            )
            cache_path.write_text(placeholder, encoding="utf-8")
            return placeholder

        left_text, right_text = split_text
        logger.warning(
            "%s hit %s fragment %s; splitting at depth %d into %d and %d chars.",
            "LLM timeout" if is_timeout else "Content filter",
            cache_prefix,
            fragment_path,
            depth,
            len(left_text),
            len(right_text),
        )
        left_result = _recover_filtered_text(
            client,
            system_prompt,
            render_user_prompt,
            left_text,
            lang,
            partial_dir,
            cache_prefix,
            max_tokens,
            depth + 1,
            f"{fragment_path}a",
        )
        right_result = _recover_filtered_text(
            client,
            system_prompt,
            render_user_prompt,
            right_text,
            lang,
            partial_dir,
            cache_prefix,
            max_tokens,
            depth + 1,
            f"{fragment_path}b",
        )
        result = "\n\n".join(part for part in (left_result.strip(), right_result.strip()) if part)

    cache_path.write_text(result, encoding="utf-8")
    return result


def _call_llm(
    client: AzureOpenAI,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 16000,
    split_on_timeout: bool = False,
) -> str:
    """Make a single chat completion call with retry logic."""
    effective_max = max(max_tokens, 8000)
    current_user_prompt = user_prompt
    used_sanitized_prompt = False
    logger.info("Calling LLM (max_completion_tokens=%d)...", effective_max)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt},
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
            if _is_content_filter_error(e):
                if not used_sanitized_prompt:
                    sanitized_prompt = _sanitize_filtered_prompt(current_user_prompt)
                    if sanitized_prompt != current_user_prompt:
                        current_user_prompt = sanitized_prompt
                        used_sanitized_prompt = True
                        logger.warning(
                            "LLM prompt hit content filter; retrying immediately with sanitized source text."
                        )
                        continue
                raise ContentFilterError("LLM prompt was blocked by Azure content filtering") from e
            if split_on_timeout and _is_timeout_error(e):
                raise LlmRequestTimeoutError("LLM request timed out while generating a chunk") from e
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

    system_prompt = render_prompt(
        "simple_summary_system.j2",
        lang_label=lang_label,
        target_words=target_words,
    )
    user_prompt = render_prompt(
        "simple_summary_user.j2",
        description=description.lower(),
        target_words=target_words,
        source_md=source_md[:40000],
    )
    return _call_llm(client, system_prompt, user_prompt, max_tokens=8000)


def _do_podcast_section(
    client: AzureOpenAI, section_text: str,
    section_idx: int, total_sections: int, lang: str, book_name: str, output_dir: Path,
) -> str:
    """Generate a single podcast section with crash recovery via partial files."""
    speakers = PODCAST_SPEAKERS[lang]
    lang_label = LANGUAGES[lang]["label"]
    target_words = SUMMARY_TYPES["podcast_60min"]["target_words"]
    words_per_section = target_words // total_sections
    section_num = section_idx + 1

    partial_dir = book_output_dir(book_name, output_dir) / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"{book_name}_podcast_60min_{lang}_section{section_num}.md"

    if partial_path.exists() and partial_path.stat().st_size > 100:
        logger.info("Loading cached podcast section %d/%d for %s", section_num, total_sections, lang)
        return partial_path.read_text(encoding="utf-8")

    system_prompt = render_prompt(
        "podcast_section_system.j2",
        lang_label=lang_label,
        male_speaker=speakers["male"],
        female_speaker=speakers["female"],
    )

    if section_num == 1:
        section_role = "opening"
    elif section_num == total_sections:
        section_role = "closing"
    else:
        section_role = "middle"

    user_prompt = render_prompt(
        "podcast_section_user.j2",
        section_num=section_num,
        total_sections=total_sections,
        section_role=section_role,
        words_per_section=words_per_section,
        max_words=int(words_per_section * 1.2),
        section_text=section_text,
    )
    logger.info("Generating podcast section %d/%d in %s (~%d words)...",
                section_num, total_sections, lang_label, words_per_section)
    result = _call_llm(client, system_prompt, user_prompt, max_tokens=8000)
    if result:
        partial_path.write_text(result, encoding="utf-8")
    return result


def _do_tts_chunk(
    client: AzureOpenAI, chunk_text: str,
    chunk_idx: int, total_chunks: int, lang: str, book_name: str, output_dir: Path,
) -> str:
    """Process a single TTS source chunk with crash recovery."""
    lang_label = LANGUAGES[lang]["label"]
    chunk_num = chunk_idx + 1

    partial_dir = book_output_dir(book_name, output_dir) / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"{book_name}_source_tts_{lang}_chunk{chunk_num}.md"

    if partial_path.exists() and partial_path.stat().st_size > 100:
        logger.info("Loading cached TTS chunk %d/%d for %s", chunk_num, total_chunks, lang)
        return partial_path.read_text(encoding="utf-8")

    system_prompt = render_prompt(
        "tts_chunk_system.j2",
        lang_label=lang_label,
        translate_to_czech=(lang == "cs"),
    )

    logger.info("Processing TTS source chunk %d/%d for lang=%s...", chunk_num, total_chunks, lang)
    cache_prefix = f"{book_name}_source_tts_{lang}_chunk{chunk_num}"
    result = _recover_filtered_text(
        client=client,
        system_prompt=system_prompt,
        render_user_prompt=lambda text: render_prompt(
            "tts_chunk_user.j2",
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            chunk_text=text,
        ),
        source_text=chunk_text,
        lang=lang,
        partial_dir=partial_dir,
        cache_prefix=cache_prefix,
        max_tokens=16000,
    )
    if result:
        partial_path.write_text(result, encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Orchestration - all tasks in a single pool
# ---------------------------------------------------------------------------

def run(
    book_name: str,
    source_md_path: Path,
    output_dir: Path = OUTPUT_DIR,
    on_file_ready: FileReadyCallback | None = None,
) -> dict[str, Path]:
    """Run all LLM tasks in a single parallel pool for maximum throughput.

    Every atomic unit of work (each summary, each podcast section, each TTS
    chunk) is submitted independently.  Multi-part outputs are assembled as
    their constituents complete.

    Args:
        book_name: Sanitized identifier of the source PDF/book.
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
            path = output_text_path(book_name, stype, lang, output_dir=output_dir)
            key = f"{book_name}_{stype}_{lang}"
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
        path = output_text_path(book_name, "podcast_60min", lang, output_dir=output_dir)
        key = f"{book_name}_podcast_60min_{lang}"
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
        path = output_text_path(book_name, SOURCE_TTS_NAME, lang, output_dir=output_dir)
        key = f"{book_name}_{SOURCE_TTS_NAME}_{lang}"
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
            task["path"].parent.mkdir(parents=True, exist_ok=True)
            task["path"].write_text(text, encoding="utf-8")
            key = task["key"]
            with lock:
                outputs[key] = task["path"]
            logger.info("-> %s (%d words)", key, len(text.split()))
            if on_file_ready:
                on_file_ready(book_name, task["summary_type"], task["lang"], task["path"], False)

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
                path = output_text_path(book_name, "podcast_60min", lang, output_dir=output_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(assembled, encoding="utf-8")
                key = f"{book_name}_podcast_60min_{lang}"
                with lock:
                    outputs[key] = path
                logger.info("=> Assembled %s (%d words from %d sections)",
                            key, len(assembled.split()), total)
                if on_file_ready:
                    on_file_ready(book_name, "podcast_60min", lang, path, True)

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
                path = output_text_path(book_name, SOURCE_TTS_NAME, lang, output_dir=output_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(assembled, encoding="utf-8")
                key = f"{book_name}_{SOURCE_TTS_NAME}_{lang}"
                with lock:
                    outputs[key] = path
                logger.info("=> Assembled %s (%d words from %d chunks)",
                            key, len(assembled.split()), total)
                if on_file_ready:
                    on_file_ready(book_name, SOURCE_TTS_NAME, lang, path, False)

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
                    task["section_idx"], task["total_sections"], task["lang"], book_name, output_dir,
                )
            elif task["type"] == "tts_chunk":
                future = pool.submit(
                    _do_tts_chunk, client, task["chunk_text"],
                    task["chunk_idx"], task["total_chunks"], task["lang"], book_name, output_dir,
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
