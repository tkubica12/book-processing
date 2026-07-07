"""Metadata helpers for processed book and paper outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any

import yaml

METADATA_NAME = "metadata.yaml"
DOCUMENT_BOOK = "book"
DOCUMENT_PAPER = "paper"
SOURCE_MEDIUM_AUDIO = "audio"
SOURCE_MEDIUM_EPUB = "ePub"
SOURCE_MEDIUM_PDF = "PDF"
SOURCE_MEDIUM_TEXT = "text"
SOURCE_MEDIUM_UNKNOWN = "unknown"
DEFAULT_ADDED_DATE = "2026-05-01"

ARXIV_MARKERS = ("arxiv", "arxiv.org", "arxiv:")
PAPER_EARLY_MARKERS = ("abstract", "introduction")
LABEL_KEYWORDS = {
    "AI": (
        "ai",
        "artificial intelligence",
        "ai agent",
        "ai agents",
        "copilot",
        "foundation model",
        "llm",
        "large language model",
        "machine learning",
        "model routing",
        "openai",
        "prompt",
    ),
    "ComputerScience": (
        "algorithm",
        "attack",
        "auth",
        "cloud",
        "computation",
        "computer",
        "cryptography",
        "data engineering",
        "database",
        "debugging",
        "exploit",
        "kubernetes",
        "llm pool",
        "malware",
        "programming",
        "routing",
        "security",
        "software",
        "threat",
        "vulnerability",
    ),
    "Science": (
        "anatomy",
        "anxiety",
        "astrobiology",
        "biology",
        "brain",
        "cognition",
        "consciousness",
        "cosmology",
        "emotion",
        "entropy",
        "evolution",
        "gene",
        "genetic",
        "human body",
        "mental health",
        "mitochondria",
        "neuroscience",
        "physics",
        "psychology",
        "quantum",
        "social brain",
        "thermodynamics",
        "universe",
    ),
    "Technology": (
        "automation",
        "drone",
        "innovation",
        "robot",
        "technology",
    ),
}
LEGACY_LABELS = {
    "ai": "AI",
    "computers": "ComputerScience",
    "security": "ComputerScience",
    "biology": "Science",
    "physics": "Science",
    "psychology": "Science",
    "technology": "Technology",
    "other": "Other",
}
DOCUMENT_LABELS = {"book", "paper", "arxiv"}
MAX_LABELS = 1


@dataclass(frozen=True)
class SourceMetadata:
    """Stable metadata written next to each processed output."""

    source_path: str
    document_type: str
    source_medium: str
    added_date: str
    labels: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return YAML-serializable metadata."""

        return {
            "source_path": self.source_path,
            "document_type": self.document_type,
            "source_medium": self.source_medium,
            "added_date": self.added_date,
            "labels": self.labels,
        }


def metadata_path(book_dir: Path) -> Path:
    """Return the metadata path for one output directory."""

    return book_dir / METADATA_NAME


def read_metadata(book_dir: Path) -> SourceMetadata | None:
    """Read metadata.yaml from an output directory when present."""

    path = metadata_path(book_dir)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = data.get("labels") or ["other"]
    return SourceMetadata(
        source_path=str(data.get("source_path") or book_dir.name),
        document_type=str(data.get("document_type") or DOCUMENT_BOOK),
        source_medium=str(data.get("source_medium") or SOURCE_MEDIUM_UNKNOWN),
        added_date=str(data.get("added_date") or DEFAULT_ADDED_DATE),
        labels=normalize_labels([str(label) for label in labels]),
    )


def write_metadata(book_dir: Path, metadata: SourceMetadata) -> Path:
    """Write metadata.yaml for one output directory."""

    book_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(book_dir)
    path.write_text(
        yaml.safe_dump(metadata.as_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def classify_labels(book_name: str, title: str = "", summary: str = "", source_text: str = "") -> list[str]:
    """Classify a source into one or more stable topical labels."""

    primary = f"{book_name} {title} {summary}".casefold()
    source_excerpt = source_text[:30_000].casefold()
    scores: dict[str, int] = {}
    for label, keywords in LABEL_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword.casefold())}\b"
            score += 4 * len(re.findall(pattern, primary))
            score += len(re.findall(pattern, source_excerpt))
        if score:
            scores[label] = score

    if not scores:
        return ["Other"]
    if scores.get("AI"):
        return ["AI"]
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].casefold()))
    return [label for label, _score in ranked[:MAX_LABELS]]


def normalize_labels(labels: list[str]) -> list[str]:
    """Return simplified topic labels from current or legacy metadata labels."""

    normalized: list[str] = []
    for label in labels:
        key = label.strip()
        if not key or key.casefold() in DOCUMENT_LABELS:
            continue
        mapped = LEGACY_LABELS.get(key.casefold(), key)
        if mapped not in LABEL_KEYWORDS and mapped != "Other":
            mapped = "Other"
        if mapped not in normalized:
            normalized.append(mapped)

    if "AI" in normalized:
        return ["AI"]
    if not normalized:
        return ["Other"]
    return normalized[:MAX_LABELS]


def today_added_date() -> str:
    """Return the ISO date to write for newly processed sources."""

    return date.today().isoformat()


def is_paper_text(book_name: str, source_text: str) -> bool:
    """Return True when text appears to be a research paper."""

    haystack = f"{book_name} {source_text[:5000]}".casefold()
    normalized_source = re.sub(r"\s+", " ", source_text.casefold())
    early_source = normalized_source[:5000]
    has_paper_shape = (
        len(source_text) < 250_000
        and all(marker in early_source for marker in PAPER_EARLY_MARKERS)
        and "references" in normalized_source
        and re.search(r"\b(?:et al\.|abstract)\b", normalized_source) is not None
    )
    return any(marker in haystack for marker in ARXIV_MARKERS) or has_paper_shape


def infer_document_type(book_name: str, source_text: str, explicit_document_type: str | None = None) -> str:
    """Return the document type, honoring explicit source layout metadata first."""

    if explicit_document_type in {DOCUMENT_BOOK, DOCUMENT_PAPER}:
        return explicit_document_type
    return DOCUMENT_PAPER if is_paper_text(book_name, source_text) else DOCUMENT_BOOK


def display_document_label(document_type: str) -> str:
    """Return the label shown in the web catalog for a document type."""

    return "Paper" if document_type == DOCUMENT_PAPER else "Book"
