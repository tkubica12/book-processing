"""Jinja-based prompt rendering for LLM tasks."""

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from book_processing.config import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts"


@lru_cache(maxsize=1)
def _get_environment() -> Environment:
    """Return the shared Jinja environment for prompt rendering."""
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )


def available_prompt_templates() -> list[Path]:
    """Return prompt template files from the prompts directory."""
    return sorted(PROMPTS_DIR.glob("*.j2"))


def render_prompt(template_name: str, **context: object) -> str:
    """Render a named prompt template with strict variable checking."""
    template = _get_environment().get_template(template_name)
    return template.render(**context).strip()