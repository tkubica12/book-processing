"""Generate a static web catalog for processed book outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import html
import logging
import re
from pathlib import Path
from urllib.parse import quote

from book_processing.config import LANGUAGES, OUTPUT_DIR, VISUAL_SUMMARY_NAME
from book_processing.metadata import (
    DEFAULT_ADDED_DATE,
    DOCUMENT_BOOK,
    DOCUMENT_PAPER,
    SourceMetadata,
    classify_labels,
    display_document_label,
    infer_document_type,
    read_metadata,
)

logger = logging.getLogger(__name__)

_AUDIO_TYPES = {
    "summary_5min": ("5-minute summary", 1),
    "summary_20min": ("20-minute summary", 2),
    "podcast_20min": ("20-minute podcast", 3),
    "podcast_60min": ("60-minute podcast", 4),
    "source_tts": ("Full audiobook", 5),
}
_INDEX_NAME = "index.html"


@dataclass(frozen=True)
class AudioAsset:
    """Metadata for one generated audio file."""

    path: Path
    type_key: str
    type_label: str
    language: str
    language_label: str
    size_bytes: int
    sort_order: int


@dataclass(frozen=True)
class BookPage:
    """Metadata for one generated per-book page."""

    book_name: str
    title: str
    summary: str
    directory: Path
    page_path: Path
    visual_summary_path: Path | None
    audio_assets: tuple[AudioAsset, ...]
    document_type: str
    labels: tuple[str, ...]
    added_date: str
    total_size_bytes: int
    file_count: int


def generate_site(output_dir: Path = OUTPUT_DIR) -> list[BookPage]:
    """Create a static landing page and per-book index pages under the output directory."""

    books = discover_books(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for book in books:
        book.page_path.write_text(_render_book_page(book), encoding="utf-8")
        logger.info("Generated book page: %s", book.page_path)

    index_path = output_dir / _INDEX_NAME
    index_path.write_text(_render_landing_page(books), encoding="utf-8")
    logger.info("Generated landing page: %s", index_path)
    return books


def discover_books(output_dir: Path = OUTPUT_DIR) -> list[BookPage]:
    """Return processed books discovered in the output directory."""

    if not output_dir.exists():
        return []

    books: list[BookPage] = []
    for directory in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        book_name = directory.name
        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            continue

        source_text = _source_text_for_labels(book_name, directory)
        visual_summary = directory / f"{book_name}_{VISUAL_SUMMARY_NAME}.html"
        visual_path = visual_summary if visual_summary.exists() else None
        title, summary = _book_text(book_name, visual_path)
        metadata = read_metadata(directory)
        document_type = _document_type(book_name, source_text, metadata)
        audio_assets = tuple(_discover_audio_assets(book_name, directory, document_type))
        if visual_path is None and not audio_assets:
            continue

        total_size_bytes = sum(path.stat().st_size for path in files)
        labels = _labels_for_book(book_name, title, summary, source_text, document_type, metadata)
        added_date = metadata.added_date if metadata else DEFAULT_ADDED_DATE
        books.append(
            BookPage(
                book_name=book_name,
                title=title,
                summary=summary,
                directory=directory,
                page_path=directory / _INDEX_NAME,
                visual_summary_path=visual_path,
                audio_assets=audio_assets,
                document_type=document_type,
                labels=labels,
                added_date=added_date,
                total_size_bytes=total_size_bytes,
                file_count=len(files),
            )
        )

    return sorted(
        sorted(books, key=lambda book: book.title.casefold()),
        key=lambda book: book.added_date,
        reverse=True,
    )


def main() -> None:
    """Generate the static web catalog from the default output directory."""

    generate_site()


def _discover_audio_assets(book_name: str, directory: Path, document_type: str) -> list[AudioAsset]:
    pattern = re.compile(rf"^{re.escape(book_name)}_(.+)_(en|cs)\.mp3$", re.IGNORECASE)
    assets: list[AudioAsset] = []
    for path in sorted(directory.glob("*.mp3")):
        match = pattern.match(path.name)
        if not match:
            continue
        type_key, language = match.groups()
        if type_key not in _AUDIO_TYPES or language not in LANGUAGES:
            continue
        if document_type == DOCUMENT_PAPER and type_key in {"podcast_60min", "source_tts"}:
            continue
        type_label, sort_order = _AUDIO_TYPES[type_key]
        assets.append(
            AudioAsset(
                path=path,
                type_key=type_key,
                type_label=type_label,
                language=language,
                language_label=LANGUAGES[language]["label"],
                size_bytes=path.stat().st_size,
                sort_order=sort_order,
            )
        )
    return sorted(assets, key=lambda asset: (asset.sort_order, asset.language))


def _book_text(book_name: str, visual_summary_path: Path | None) -> tuple[str, str]:
    fallback_title = _humanize_book_name(book_name)
    if visual_summary_path is None:
        return fallback_title, ""

    text = visual_summary_path.read_text(encoding="utf-8")
    title = _extract_tag_text(text, "h1") or fallback_title
    summary = _extract_class_text(text, "summary")
    return title, _truncate(summary, 240)


def _source_text_for_labels(book_name: str, directory: Path) -> str:
    source_path = directory / f"{book_name}_source_raw.md"
    if not source_path.exists():
        return ""
    return source_path.read_text(encoding="utf-8", errors="ignore")[:250_000]


def _document_type(book_name: str, source_text: str, metadata: SourceMetadata | None) -> str:
    return infer_document_type(
        book_name,
        source_text,
        explicit_document_type=metadata.document_type if metadata else None,
    )


def _labels_for_book(
    book_name: str,
    title: str,
    summary: str,
    source_text: str,
    document_type: str,
    metadata: SourceMetadata | None,
) -> tuple[str, ...]:
    labels = metadata.labels if metadata else classify_labels(book_name, title, summary, source_text)
    return tuple([display_document_label(document_type), *labels])


def _extract_tag_text(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return _strip_html(match.group(1)) if match else ""


def _extract_class_text(text: str, class_name: str) -> str:
    match = re.search(
        rf'<[^>]+class="[^"]*\b{re.escape(class_name)}\b[^"]*"[^>]*>(.*?)</[^>]+>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _strip_html(match.group(1)) if match else ""


def _strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", without_tags)).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _humanize_book_name(book_name: str) -> str:
    return " ".join(part.capitalize() for part in book_name.split("_") if part)


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _format_date(value: str) -> str:
    try:
        return date.fromisoformat(value).strftime("%d %b %Y")
    except ValueError:
        return value


def _count_label(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count with the right singular/plural label."""

    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _href(path: Path) -> str:
    return quote(path.as_posix(), safe="/._-")


def _theme_script() -> str:
    return """<script>
  (() => {
    const param = new URLSearchParams(window.location.search).get("clawpilotTheme");
    const theme =
      param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  })();
  </script>"""


def _style() -> str:
    return """<style>
:root {
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
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
}
html[data-theme="dark"] {
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
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
  margin: 0;
  background: var(--cp-bg);
  color: var(--cp-text);
  font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.65;
}
a { color: var(--cp-link); }
.page {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 56px 0 80px;
}
.hero {
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--cp-border);
}
.eyebrow {
  color: var(--cp-accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h1 {
  margin: 12px 0;
  font-size: clamp(2rem, 5vw, 4.25rem);
  line-height: 1.05;
  letter-spacing: -0.04em;
}
.summary {
  max-width: 780px;
  margin: 0;
  color: var(--cp-text-muted);
  font-size: 1.16rem;
}
.stats, .actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 22px;
}
.filters {
  display: grid;
  gap: 14px;
  margin: 28px 0;
}
.search-input {
  width: 100%;
  border: 1px solid var(--cp-border);
  border-radius: 14px;
  background: var(--cp-surface);
  color: var(--cp-text);
  font: inherit;
  padding: 12px 14px;
}
.toolbar {
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(0, 1fr) auto;
}
.sort-select {
  border: 1px solid var(--cp-border);
  border-radius: 14px;
  background: var(--cp-surface);
  color: var(--cp-text);
  font: inherit;
  padding: 12px 14px;
}
.label-filters {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.pill, .button {
  border: 1px solid var(--cp-border);
  border-radius: 999px;
  background: var(--cp-surface);
  color: var(--cp-text);
  padding: 7px 12px;
  text-decoration: none;
}
.pill[data-filter] {
  cursor: pointer;
}
.pill[data-active="true"] {
  border-color: var(--cp-accent);
  background: var(--cp-accent);
  color: var(--cp-accent-fg);
}
.empty-state {
  display: none;
  margin-top: 24px;
  color: var(--cp-text-muted);
}
.button {
  border-color: var(--cp-accent);
  background: var(--cp-accent);
  color: var(--cp-accent-fg);
  font-weight: 700;
}
.button.secondary {
  border-color: var(--cp-border);
  background: var(--cp-surface);
  color: var(--cp-text);
}
.grid {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.card, .audio-card {
  border: 1px solid var(--cp-border);
  border-radius: 16px;
  background: var(--cp-surface);
  box-shadow: var(--cp-shadow);
}
.card {
  display: flex;
  min-height: 100%;
  padding: 20px;
  text-decoration: none;
  color: var(--cp-text);
  flex-direction: column;
}
.card:hover { border-color: var(--cp-border-strong); background: var(--cp-accent-soft); }
.card h2, .audio-card h2 {
  margin: 0 0 8px;
  font-size: 1.1rem;
  line-height: 1.35;
}
.card p, .audio-card p {
  margin: 0;
  color: var(--cp-text-muted);
}
.card footer {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: auto;
  padding-top: 18px;
}
.section-title {
  margin: 36px 0 14px;
  color: var(--cp-accent);
  font-size: 0.9rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.audio-list {
  display: grid;
  gap: 16px;
}
.audio-card {
  padding: 18px;
}
audio {
  display: block;
  width: 100%;
  margin-top: 12px;
}
.meta {
  color: var(--cp-text-soft);
  font-size: 0.95rem;
}
@media (max-width: 640px) {
  .page { width: min(100% - 20px, 1120px); padding-top: 32px; }
  .toolbar { grid-template-columns: 1fr; }
}
</style>"""


def _document_head(title: str) -> str:
    return f"""<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {_theme_script()}
  <title>{html.escape(title)}</title>
  {_style()}
</head>"""


def _render_landing_page(books: list[BookPage]) -> str:
    book_count = sum(1 for book in books if book.document_type == DOCUMENT_BOOK)
    paper_count = sum(1 for book in books if book.document_type == DOCUMENT_PAPER)
    total_audio = sum(len(book.audio_assets) for book in books)
    total_size = sum(book.total_size_bytes for book in books)
    cards = "\n".join(_render_book_card(book) for book in books)
    labels = sorted({label for book in books for label in book.labels})
    label_filters = "\n".join(
        f'<button class="pill" type="button" data-filter="{html.escape(label)}">{html.escape(label)}</button>'
        for label in labels
    )
    return f"""<!doctype html>
<html lang="en">
{_document_head("Book library")}
<body>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Private book library</div>
      <h1>Book maps and recordings</h1>
      <p class="summary">One private place for generated visual summaries, short summaries, podcasts, and full audiobook recordings.</p>
      <div class="stats" aria-label="Library statistics">
        <span class="pill">{_count_label(book_count, "book")}</span>
        <span class="pill">{_count_label(paper_count, "paper")}</span>
        <span class="pill">{_count_label(total_audio, "recording")}</span>
        <span class="pill">{_format_size(total_size)}</span>
      </div>
    </header>
    <section class="filters" aria-label="Library filters">
      <div class="toolbar">
        <input class="search-input" type="search" placeholder="Search materials..." aria-label="Search materials" data-search>
        <select class="sort-select" aria-label="Sort materials" data-sort>
          <option value="newest">Newest first</option>
          <option value="title">Title A-Z</option>
        </select>
      </div>
      <div class="label-filters" aria-label="Topic filters">
        <button class="pill" type="button" data-filter="" data-active="true">All</button>
        {label_filters}
      </div>
    </section>
    <section class="grid" aria-label="Books" data-book-grid>
      {cards}
    </section>
    <p class="empty-state" data-empty-state>No materials match the current search.</p>
  </main>
  {_landing_script()}
</body>
</html>
"""


def _render_book_card(book: BookPage) -> str:
    page_href = _href(Path(book.book_name) / _INDEX_NAME)
    summary = f"<p>{html.escape(book.summary)}</p>" if book.summary else ""
    labels = " ".join(f'<span class="pill">{html.escape(label)}</span>' for label in book.labels)
    search_text = f"{book.title} {book.summary} {display_document_label(book.document_type)} {' '.join(book.labels)}"
    return f"""<a class="card" href="{page_href}" data-book-card data-labels="{html.escape('|'.join(book.labels))}" data-search-text="{html.escape(search_text.casefold())}" data-title="{html.escape(book.title.casefold())}" data-added-date="{html.escape(book.added_date)}">
  <article>
    <h2>{html.escape(book.title)}</h2>
    {summary}
    <footer>
      <span class="pill">Added {_format_date(book.added_date)}</span>
      {labels}
    </footer>
  </article>
</a>"""


def _render_book_page(book: BookPage) -> str:
    visual_link = ""
    if book.visual_summary_path is not None:
        visual_link = (
            f'<a class="button" href="{_href(Path(book.visual_summary_path.name))}">'
            "Open visual summary</a>"
        )
    audio_cards = "\n".join(_render_audio_card(asset) for asset in book.audio_assets)
    summary = f"<p class=\"summary\">{html.escape(book.summary)}</p>" if book.summary else ""
    labels = "\n".join(f'<span class="pill">{html.escape(label)}</span>' for label in book.labels)
    return f"""<!doctype html>
<html lang="en">
{_document_head(book.title)}
<body>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">{html.escape(display_document_label(book.document_type))} detail</div>
      <h1>{html.escape(book.title)}</h1>
      {summary}
      <div class="stats" aria-label="Book statistics">
        <span class="pill">{len(book.audio_assets)} recordings</span>
        <span class="pill">{book.file_count} files</span>
        <span class="pill">{_format_size(book.total_size_bytes)}</span>
        <span class="pill">Added {_format_date(book.added_date)}</span>
        {labels}
      </div>
      <nav class="actions" aria-label="Book actions">
        <a class="button secondary" href="../{_INDEX_NAME}">Back to library</a>
        {visual_link}
      </nav>
    </header>
    <h2 class="section-title">Recordings</h2>
    <section class="audio-list" aria-label="Recordings">
      {audio_cards}
    </section>
  </main>
</body>
</html>
"""


def _render_audio_card(asset: AudioAsset) -> str:
    href = _href(Path(asset.path.name))
    title = f"{asset.type_label} - {asset.language_label}"
    return f"""<article class="audio-card">
  <h2>{html.escape(title)}</h2>
  <p class="meta">{_format_size(asset.size_bytes)}</p>
  <audio controls preload="none" src="{href}"></audio>
  <p><a href="{href}" download>Download MP3</a></p>
</article>"""


def _landing_script() -> str:
    return """<script>
  (() => {
    const search = document.querySelector("[data-search]");
    const sort = document.querySelector("[data-sort]");
    const grid = document.querySelector("[data-book-grid]");
    const cards = Array.from(document.querySelectorAll("[data-book-card]"));
    const filters = Array.from(document.querySelectorAll("[data-filter]"));
    const empty = document.querySelector("[data-empty-state]");
    let selectedLabel = "";

    const applyFilters = () => {
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      const sortedCards = [...cards].sort((left, right) => {
        if (sort.value === "title") return left.dataset.title.localeCompare(right.dataset.title);
        return right.dataset.addedDate.localeCompare(left.dataset.addedDate)
          || left.dataset.title.localeCompare(right.dataset.title);
      });
      sortedCards.forEach((card) => grid.appendChild(card));
      cards.forEach((card) => {
        const textMatch = !query || card.dataset.searchText.includes(query);
        const labels = (card.dataset.labels || "").split("|");
        const labelMatch = !selectedLabel || labels.includes(selectedLabel);
        const show = textMatch && labelMatch;
        card.hidden = !show;
        if (show) visible += 1;
      });
      if (empty) empty.style.display = visible ? "none" : "block";
    };

    search.addEventListener("input", applyFilters);
    sort.addEventListener("change", applyFilters);
    filters.forEach((filter) => {
      filter.addEventListener("click", () => {
        selectedLabel = filter.dataset.filter || "";
        filters.forEach((button) => button.dataset.active = String(button === filter));
        applyFilters();
      });
    });
  })();
</script>"""
