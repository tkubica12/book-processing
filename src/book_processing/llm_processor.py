"""LLM-powered summary generation and TTS preprocessing using Azure OpenAI."""

import logging
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from book_processing.config import (
    AZURE_COGNITIVE_SCOPE,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL,
    LANGUAGES,
    OUTPUT_DIR,
    PODCAST_SPEAKERS,
    SOURCE_TTS_NAME,
    SUMMARY_TYPES,
    output_text_path,
)

logger = logging.getLogger(__name__)


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
        timeout=1200,  # 20 minute timeout for long generations
        max_retries=0,  # We handle retries ourselves in _call_llm
    )


MAX_RETRIES = 8


def _call_llm(client: AzureOpenAI, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
    """Make a single chat completion call and return the content."""
    effective_max = max(max_tokens, 8000)  # Ensure minimum 8000 for reasoning headroom
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
                logger.warning("Response was empty — likely ran out of completion tokens.")
            return content
        except Exception as e:
            wait = 30 * (attempt + 1)
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, MAX_RETRIES, e, wait)
            import time
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts")


def _call_llm_long(client: AzureOpenAI, system_prompt: str, user_prompt: str, target_words: int) -> str:
    """Generate long-form content using a single LLM call.

    For the podcast, the generate_summary function handles sectioning
    the source material into independent segments. This function simply
    makes one call and returns the result.
    """
    return _call_llm(client, system_prompt, user_prompt, max_tokens=8000)


def generate_summary(client: AzureOpenAI, source_md: str, summary_type: str, lang: str) -> str:
    """Generate a single summary of a given type and language.

    Args:
        client: Azure OpenAI client.
        source_md: Full source markdown text.
        summary_type: Key from SUMMARY_TYPES (e.g. 'summary_2min').
        lang: Language code ('en' or 'cs').

    Returns:
        Generated summary text.
    """
    spec = SUMMARY_TYPES[summary_type]
    target_words = spec["target_words"]
    description = spec["description"]
    is_podcast = spec["is_podcast"]
    lang_label = LANGUAGES[lang]["label"]

    if is_podcast:
        speakers = PODCAST_SPEAKERS[lang]
        system_prompt = (
            f"You are a scriptwriter for a highly technical podcast. "
            f"Write in {lang_label}. "
            f"The podcast has two hosts: {speakers['male']} (male, lead host) and {speakers['female']} (female, co-host). "
            f"They are both deeply technical, geeky, and passionate about software architecture and engineering. "
            f"The conversation should be entertaining, fun, and sometimes geeky — use humor, analogies, "
            f"and show genuine excitement — but always remain accurate and deeply technical. "
            f"Format each line as [{speakers['male']}]: or [{speakers['female']}]: followed by the spoken text. "
            f"No stage directions, no parenthetical notes — only spoken dialogue. "
            f"Cover the material comprehensively — principles, architecture, deep technical details."
        )
        # Split source into smaller sections to avoid API timeouts
        # 25K chars per section keeps input well under the timeout threshold
        section_size = 25000
        source_trimmed = source_md[:200000]
        sections = [source_trimmed[i : i + section_size] for i in range(0, len(source_trimmed), section_size)]

        words_per_section = target_words // len(sections)
        total_sections = len(sections)

        # Use partial files to save progress and resume on failure
        partial_dir = OUTPUT_DIR / "_partial"
        partial_dir.mkdir(exist_ok=True)
        podcast_segments: list[str] = []

        for idx, section in enumerate(sections):
            section_num = idx + 1
            partial_path = partial_dir / f"{summary_type}_{lang}_section{section_num}.md"

            # Resume: skip already-completed sections
            if partial_path.exists() and partial_path.stat().st_size > 100:
                logger.info("Loading cached podcast section %d/%d from %s", section_num, total_sections, partial_path.name)
                podcast_segments.append(partial_path.read_text(encoding="utf-8"))
                continue

            if section_num == 1:
                intro = (
                    f"Create part {section_num} of {total_sections} of a technical podcast. "
                    f"This is the OPENING segment — include a welcoming intro. "
                )
            elif section_num == total_sections:
                intro = (
                    f"Create part {section_num} of {total_sections} of a technical podcast. "
                    f"This is the CLOSING segment — wrap up and say goodbye to listeners. "
                )
            else:
                intro = (
                    f"Create part {section_num} of {total_sections} of a technical podcast. "
                    f"Jump straight into the content — no intro needed. "
                )

            user_prompt = (
                f"{intro}"
                f"Write approximately {words_per_section} words of dialogue. "
                f"Cover ALL the topics in the source material below in deep technical detail.\n\n"
                f"SOURCE MATERIAL:\n\n{section}"
            )
            logger.info("Generating podcast section %d/%d in %s...", section_num, total_sections, lang_label)
            segment = _call_llm(client, system_prompt, user_prompt, max_tokens=8000)
            if segment:
                partial_path.write_text(segment, encoding="utf-8")
                podcast_segments.append(segment)

        return "\n\n".join(podcast_segments)
    else:
        system_prompt = (
            f"You are a technical writer creating content for a senior software architect. "
            f"Write in {lang_label}. "
            f"Be deeply technical — cover architectural principles, design patterns, technology trade-offs, "
            f"and implementation details. The reader is not afraid of complexity. "
            f"Write in a clear, flowing style suitable for reading aloud. "
            f"Do not use bullet points or lists — use well-structured paragraphs. "
            f"Do not include any markdown formatting, headers, or special characters."
        )
        user_prompt = (
            f"Create a {description.lower()} of approximately {target_words} words "
            f"based on the following source material.\n\n"
            f"SOURCE MATERIAL:\n\n{source_md[:40000]}"
        )

    return _call_llm(client, system_prompt, user_prompt, max_tokens=8000)


def generate_tts_source(client: AzureOpenAI, source_md: str, lang: str) -> str:
    """Preprocess the full source markdown for TTS consumption.

    Cleans up tables, charts, page numbers, URLs, and other elements that
    don't work well when read aloud. For Czech, also translates the content.

    Args:
        client: Azure OpenAI client.
        source_md: Full raw source markdown.
        lang: Language code ('en' or 'cs').

    Returns:
        Cleaned/translated text ready for TTS.
    """
    lang_label = LANGUAGES[lang]["label"]

    system_prompt = (
        f"You are preprocessing a technical book for text-to-speech narration. "
        f"Output language: {lang_label}. "
        f"Your task is to transform the text so it sounds natural when read aloud. "
        f"Rules:\n"
        f"- Remove all page numbers, page headers/footers, and pagination markers\n"
        f"- Remove all URLs and email addresses (mention the service/site name instead)\n"
        f"- Convert tables into flowing spoken descriptions\n"
        f"- Convert charts/diagrams descriptions into spoken explanations\n"
        f"- Remove all markdown formatting (headers, bold, italic, code blocks)\n"
        f"- Expand abbreviations on first use\n"
        f"- Remove reference numbers and footnote markers\n"
        f"- Keep ALL technical content — do not summarize or omit anything\n"
        f"- Make it flow naturally as continuous narration\n"
        f"{'- Translate to Czech while preserving all technical meaning' if lang == 'cs' else ''}"
    )

    # Process in chunks — 20K chars keeps each call well under timeout threshold
    chunk_size = 20000
    chunks = [source_md[i : i + chunk_size] for i in range(0, len(source_md), chunk_size)]

    # Use partial files to save progress and resume on failure
    partial_dir = OUTPUT_DIR / "_partial"
    partial_dir.mkdir(exist_ok=True)
    processed_parts: list[str] = []

    for idx, chunk in enumerate(chunks):
        chunk_num = idx + 1
        partial_path = partial_dir / f"source_tts_{lang}_chunk{chunk_num}.md"

        # Resume: skip already-completed chunks
        if partial_path.exists() and partial_path.stat().st_size > 100:
            logger.info("Loading cached TTS source chunk %d/%d from %s", chunk_num, len(chunks), partial_path.name)
            processed_parts.append(partial_path.read_text(encoding="utf-8"))
            continue

        logger.info("Processing TTS source chunk %d/%d for lang=%s...", chunk_num, len(chunks), lang)
        user_prompt = (
            f"Preprocess the following section of the book for TTS narration. "
            f"This is section {chunk_num} of {len(chunks)}.\n\n"
            f"TEXT:\n\n{chunk}"
        )
        processed = _call_llm(client, system_prompt, user_prompt, max_tokens=16000)
        if processed:
            partial_path.write_text(processed, encoding="utf-8")
            processed_parts.append(processed)

    return "\n\n".join(processed_parts)


def run(source_md_path: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Run the full LLM processing stage.

    Args:
        source_md_path: Path to the raw source markdown file.
        output_dir: Directory to write output files.

    Returns:
        Dictionary mapping output names to their file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    source_md = source_md_path.read_text(encoding="utf-8")
    client = _get_client()
    outputs: dict[str, Path] = {}

    # Generate summaries for each type and language
    for summary_type in SUMMARY_TYPES:
        for lang in LANGUAGES:
            path = output_text_path(summary_type, lang)
            if path.exists() and path.stat().st_size > 100:
                logger.info("Skipping %s (already exists with %d bytes)", path.name, path.stat().st_size)
                outputs[f"{summary_type}_{lang}"] = path
                continue

            logger.info("Generating %s in %s...", summary_type, LANGUAGES[lang]["label"])
            text = generate_summary(client, source_md, summary_type, lang)

            path.write_text(text, encoding="utf-8")
            outputs[f"{summary_type}_{lang}"] = path
            logger.info("Saved %s (%d words)", path.name, len(text.split()))

    # Generate TTS-preprocessed source for each language
    for lang in LANGUAGES:
        path = output_text_path(SOURCE_TTS_NAME, lang)
        if path.exists() and path.stat().st_size > 100:
            logger.info("Skipping %s (already exists with %d bytes)", path.name, path.stat().st_size)
            outputs[f"{SOURCE_TTS_NAME}_{lang}"] = path
            continue

        logger.info("Generating TTS source in %s...", LANGUAGES[lang]["label"])
        text = generate_tts_source(client, source_md, lang)

        path.write_text(text, encoding="utf-8")
        outputs[f"{SOURCE_TTS_NAME}_{lang}"] = path
        logger.info("Saved %s (%d words)", path.name, len(text.split()))

    return outputs
