"""Generate progressive-disclosure HTML visualizations for processed books."""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from book_processing.config import (
    OUTPUT_DIR,
    SOURCE_RAW_NAME,
    VISUAL_SUMMARY_NAME,
    book_output_dir,
    output_html_path,
    output_text_path,
)
from book_processing.llm_processor import _call_llm, _get_client, _recover_filtered_text
from book_processing.prompt_templates import render_prompt

logger = logging.getLogger(__name__)

_CHUNK_CHARS = 50_000
_DIRECT_SOURCE_CHARS = 70_000
_MAX_FINAL_NOTES_CHARS = 120_000
_OUTLINE_MAX_ATTEMPTS = 3


def discover_existing_source_raws(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Return source_raw Markdown files already present under the output directory."""
    if not output_dir.exists():
        return {}

    sources: dict[str, Path] = {}
    for book_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        book_name = book_dir.name
        path = output_text_path(book_name, SOURCE_RAW_NAME, output_dir=output_dir)
        if path.exists() and path.stat().st_size > 0:
            sources[book_name] = path
    return sources


def run(book_name: str, source_md_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Generate the English progressive-disclosure HTML visualization for one book."""
    html_path = output_html_path(book_name, VISUAL_SUMMARY_NAME, output_dir=output_dir)
    if html_path.exists() and html_path.stat().st_size > 1000:
        logger.info("Skipping %s (exists, %d bytes)", html_path.name, html_path.stat().st_size)
        return html_path

    source_md = source_md_path.read_text(encoding="utf-8")
    client = _get_client()
    outline = _generate_outline(client, book_name, source_md, output_dir)
    html_text = _render_html(outline)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    logger.info("Generated HTML visualization: %s (%d bytes)", html_path, html_path.stat().st_size)
    return html_path


def _generate_outline(client: AzureOpenAI, book_name: str, source_md: str, output_dir: Path) -> dict[str, Any]:
    if len(source_md) <= _DIRECT_SOURCE_CHARS:
        source_for_outline = source_md
    else:
        notes = _generate_chunk_notes(client, book_name, source_md, output_dir)
        source_for_outline = notes[:_MAX_FINAL_NOTES_CHARS]

    system_prompt = render_prompt("visual_outline_system.j2")
    base_user_prompt = render_prompt(
        "visual_outline_user.j2",
        book_name=book_name,
        source_md=source_for_outline,
    )
    user_prompt = base_user_prompt
    last_error: RuntimeError | None = None
    for attempt in range(1, _OUTLINE_MAX_ATTEMPTS + 1):
        response = _call_llm(client, system_prompt, user_prompt, max_tokens=16000)
        try:
            return _parse_outline_json(response)
        except RuntimeError as error:
            last_error = error
            logger.warning(
                "Invalid HTML outline for %s on attempt %d/%d: %s",
                book_name,
                attempt,
                _OUTLINE_MAX_ATTEMPTS,
                error,
            )
            user_prompt = (
                f"{base_user_prompt}\n\n"
                "The previous response was invalid. Return strict JSON only, with a top-level "
                '"segments" array containing between 3 and 20 segment objects. '
                f"Validation error: {error}"
            )

    raise RuntimeError(f"Could not generate a valid HTML visualization outline for {book_name}") from last_error


def _generate_chunk_notes(client: AzureOpenAI, book_name: str, source_md: str, output_dir: Path) -> str:
    chunks = _split_text(source_md, _CHUNK_CHARS)
    system_prompt = render_prompt("visual_chunk_system.j2")
    partial_dir = book_output_dir(book_name, output_dir) / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        partial_path = partial_dir / f"{book_name}_{VISUAL_SUMMARY_NAME}_chunk{index}.md"
        if partial_path.exists() and partial_path.stat().st_size > 100:
            logger.info("Loading cached visual notes for %s chunk %d/%d", book_name, index, len(chunks))
            notes.append(partial_path.read_text(encoding="utf-8"))
            continue

        logger.info("Extracting visual notes for %s chunk %d/%d", book_name, index, len(chunks))
        cache_prefix = f"{book_name}_{VISUAL_SUMMARY_NAME}_chunk{index}"
        result = _recover_filtered_text(
            client=client,
            system_prompt=system_prompt,
            render_user_prompt=lambda text: render_prompt(
                "visual_chunk_user.j2",
                chunk_num=index,
                total_chunks=len(chunks),
                chunk_text=text,
            ),
            source_text=chunk,
            lang="en",
            partial_dir=partial_dir,
            cache_prefix=cache_prefix,
            max_tokens=8000,
        )
        if result:
            partial_path.write_text(result, encoding="utf-8")
        notes.append(result)
    return "\n\n".join(notes)


def _split_text(text: str, max_chars: int) -> list[str]:
    paragraphs = re.split(r"(\n\s*\n+)", text)
    chunks: list[str] = []
    current = ""
    for part in paragraphs:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current.strip():
            chunks.append(current.strip())
        current = part
        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars:]
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text.strip()]


def _parse_outline_json(response: str) -> dict[str, Any]:
    text = response.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("HTML visualization LLM response was not valid JSON") from error

    if not isinstance(data, dict):
        raise RuntimeError("HTML visualization outline must be a JSON object")
    segments = data.get("segments")
    if not isinstance(segments, list) or not 3 <= len(segments) <= 20:
        raise RuntimeError("HTML visualization outline must contain 3-20 segments")
    return data


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def _safe_inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _details_to_paragraphs(value: Any) -> str:
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
    else:
        parts = re.split(r"(?<=[.!?])\s+", _as_text(value))
    return "\n".join(f"<p>{_safe_inline_markdown(part)}</p>" for part in parts if part)


def _render_subtopics(subtopics: Any, segment_index: int) -> str:
    if not isinstance(subtopics, list) or not subtopics:
        return ""

    rendered: list[str] = ['<div class="subtopics" aria-label="Subtopics">']
    for sub_index, subtopic in enumerate(subtopics[:5], start=1):
        if not isinstance(subtopic, dict):
            continue
        title = _as_text(subtopic.get("title")) or f"Subtopic {sub_index}"
        summary = _as_text(subtopic.get("summary"))
        details = _details_to_paragraphs(subtopic.get("details", ""))
        sub_id = f"segment-{segment_index}-subtopic-{sub_index}"
        rendered.append(
            f"""
            <article class="subtopic">
              <button class="subtopic-toggle" type="button" aria-expanded="false" aria-controls="{sub_id}">
                <span>
                  <strong>{html.escape(title)}</strong>
                  <small>{html.escape(summary)}</small>
                </span>
                <span class="chevron" aria-hidden="true">+</span>
              </button>
              <div class="subtopic-body" id="{sub_id}" hidden>
                {details}
              </div>
            </article>
            """
        )
    rendered.append("</div>")
    return "\n".join(rendered)


def _render_segments(segments: Any) -> str:
    rendered: list[str] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        title = _as_text(segment.get("title")) or f"Segment {index}"
        summary = _as_text(segment.get("summary"))
        details = _details_to_paragraphs(segment.get("details", ""))
        subtopics = _render_subtopics(segment.get("subtopics"), index)
        segment_id = f"segment-{index}"
        rendered.append(
            f"""
            <article class="segment-card">
              <button class="segment-toggle" type="button" aria-expanded="false" aria-controls="{segment_id}">
                <span class="segment-number">{index:02d}</span>
                <span class="segment-heading">
                  <strong>{html.escape(title)}</strong>
                  <small>{html.escape(summary)}</small>
                </span>
                <span class="chevron" aria-hidden="true">+</span>
              </button>
              <div class="segment-body" id="{segment_id}" hidden>
                <div class="details">
                  {details}
                </div>
                {subtopics}
              </div>
            </article>
            """
        )
    return "\n".join(rendered)


def _render_html(outline: dict[str, Any]) -> str:
    title = _as_text(outline.get("title")) or "Book visual summary"
    subtitle = _as_text(outline.get("subtitle"))
    summary = _as_text(outline.get("main_summary"))
    segments = _render_segments(outline.get("segments"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script>
  (() => {{
    const param = new URLSearchParams(window.location.search).get("clawpilotTheme");
    const theme =
      param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  }})();
  </script>
  <title>{html.escape(title)}</title>
  <style>
:root {{
  color-scheme: light;
  --cp-bg: #f7f4ef;
  --cp-bg-elevated: #fcfbf8;
  --cp-surface: #ffffff;
  --cp-surface-soft: #f5f5f5;
  --cp-border: #dedede;
  --cp-border-strong: #919191;
  --cp-text: #242424;
  --cp-text-muted: #5c5c5c;
  --cp-text-soft: #6f6f6f;
  --cp-accent: #b11f4b;
  --cp-accent-hover: #9a1a41;
  --cp-accent-soft: rgba(177, 31, 75, 0.08);
  --cp-accent-fg: #ffffff;
  --cp-success: #16a34a;
  --cp-danger: #dc2626;
  --cp-warning: #f59e0b;
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
  --cp-overlay: rgba(255, 255, 255, 0.8);
  --cp-panel: rgba(255, 255, 255, 0.86);
  --cp-panel-strong: rgba(255, 255, 255, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.55);
  --cp-highlight: rgba(177, 31, 75, 0.12);
}}
html[data-theme="dark"] {{
  color-scheme: dark;
  --cp-bg: #3d3b3a;
  --cp-bg-elevated: #343231;
  --cp-surface: #292929;
  --cp-surface-soft: #2e2e2e;
  --cp-border: #474747;
  --cp-border-strong: #5f5f5f;
  --cp-text: #dedede;
  --cp-text-muted: #919191;
  --cp-text-soft: #b0b0b0;
  --cp-accent: #fd8ea1;
  --cp-accent-hover: #fb7b91;
  --cp-accent-soft: rgba(253, 142, 161, 0.14);
  --cp-accent-fg: #1a1a1a;
  --cp-success: #4ade80;
  --cp-danger: #f87171;
  --cp-warning: #fbbf24;
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --cp-overlay: rgba(41, 41, 41, 0.88);
  --cp-panel: rgba(41, 41, 41, 0.72);
  --cp-panel-strong: rgba(41, 41, 41, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.04);
  --cp-highlight: rgba(253, 142, 161, 0.12);
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  background: var(--cp-bg);
  color: var(--cp-text);
  font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.65;
}}
.page {{
  width: min(960px, calc(100% - 32px));
  margin: 0 auto;
  padding: 56px 0 80px;
}}
.hero {{
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--cp-border);
}}
.eyebrow {{
  color: var(--cp-accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
h1 {{
  margin: 12px 0;
  font-size: clamp(2rem, 5vw, 4.25rem);
  line-height: 1.05;
  letter-spacing: -0.04em;
}}
.subtitle {{
  margin: 0 0 20px;
  color: var(--cp-text-muted);
  font-size: 1.1rem;
}}
.summary {{
  max-width: 760px;
  margin: 0;
  font-size: 1.22rem;
}}
strong {{
  color: var(--cp-accent);
}}
.controls {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 24px 0;
}}
.control-button {{
  border: 1px solid var(--cp-border);
  border-radius: 0.625rem;
  background: var(--cp-surface);
  color: var(--cp-text);
  cursor: pointer;
  font: inherit;
  padding: 8px 14px;
}}
.control-button:hover {{
  border-color: var(--cp-border-strong);
}}
.segments {{
  display: grid;
  gap: 16px;
}}
.segment-card, .subtopic {{
  border: 1px solid var(--cp-border);
  border-radius: 16px;
  background: var(--cp-surface);
  box-shadow: var(--cp-shadow);
  overflow: hidden;
}}
.segment-toggle, .subtopic-toggle {{
  width: 100%;
  border: 0;
  background: var(--cp-surface);
  color: var(--cp-text);
  cursor: pointer;
  display: grid;
  gap: 16px;
  grid-template-columns: auto 1fr auto;
  align-items: start;
  padding: 20px;
  text-align: left;
  font: inherit;
}}
.subtopic-toggle {{
  background: var(--cp-surface-soft);
  padding: 16px;
}}
.segment-toggle:hover, .subtopic-toggle:hover {{
  background: var(--cp-accent-soft);
}}
.segment-number {{
  color: var(--cp-accent);
  font-family: Consolas, "Courier New", Courier, monospace;
  font-weight: 700;
}}
.segment-heading strong, .subtopic-toggle strong {{
  display: block;
  color: var(--cp-text);
  font-size: 1.08rem;
  line-height: 1.35;
}}
.segment-heading small, .subtopic-toggle small {{
  display: block;
  color: var(--cp-text-muted);
  font-size: 0.98rem;
  margin-top: 4px;
}}
.chevron {{
  color: var(--cp-accent);
  font-weight: 700;
  line-height: 1.35;
}}
.segment-body, .subtopic-body {{
  border-top: 1px solid var(--cp-border);
  padding: 4px 20px 20px 56px;
}}
.details {{
  max-width: 760px;
}}
.details p, .subtopic-body p {{
  margin: 16px 0 0;
}}
.subtopics {{
  display: grid;
  gap: 12px;
  margin-top: 20px;
}}
.subtopic {{
  box-shadow: none;
}}
.subtopic-body {{
  padding-left: 20px;
}}
@media (max-width: 640px) {{
  .page {{
    width: min(100% - 20px, 960px);
    padding-top: 32px;
  }}
  .segment-toggle, .subtopic-toggle {{
    grid-template-columns: 1fr auto;
  }}
  .segment-number {{
    display: none;
  }}
  .segment-body {{
    padding-left: 20px;
  }}
}}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Progressive book map</div>
      <h1>{html.escape(title)}</h1>
      {f'<p class="subtitle">{html.escape(subtitle)}</p>' if subtitle else ''}
      <p class="summary">{_safe_inline_markdown(summary)}</p>
    </header>
    <div class="controls" aria-label="Expansion controls">
      <button class="control-button" type="button" data-action="expand">Expand all</button>
      <button class="control-button" type="button" data-action="collapse">Collapse all</button>
    </div>
    <section class="segments" aria-label="Key ideas">
      {segments}
    </section>
  </main>
  <script>
    const setExpanded = (button, expanded) => {{
      const panel = document.getElementById(button.getAttribute("aria-controls"));
      button.setAttribute("aria-expanded", String(expanded));
      const icon = button.querySelector(".chevron");
      if (icon) icon.textContent = expanded ? "−" : "+";
      if (panel) panel.hidden = !expanded;
    }};

    document.querySelectorAll("[aria-controls]").forEach((button) => {{
      button.addEventListener("click", () => {{
        setExpanded(button, button.getAttribute("aria-expanded") !== "true");
      }});
    }});

    document.querySelector('[data-action="expand"]').addEventListener("click", () => {{
      document.querySelectorAll("[aria-controls]").forEach((button) => setExpanded(button, true));
    }});

    document.querySelector('[data-action="collapse"]').addEventListener("click", () => {{
      document.querySelectorAll("[aria-controls]").forEach((button) => setExpanded(button, false));
    }});
  </script>
</body>
</html>
"""
