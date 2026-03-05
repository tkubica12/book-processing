# AGENTS.md

Coding conventions and tooling guidance for AI agents working in this repository.

## Python Code style

- Use `uv` for all package management. No `requirements.txt` — dependencies live in `pyproject.toml`.
- Docstrings on all public functions and classes; skip obvious inline comments.
- No premature abstractions — solve the problem at hand, refactor when a pattern repeats.
- Keep modules focused and small; flat structure beats deep nesting.
- Always search for latest stable versions of libraries and other dependencies 

## Agent coding

- Start with planning and make sure you understand the task and data before coding.
- Where possible create tests so you can validate your progress and fix issues early.
- Work autonomously, but don't hesitate to stop and offer options for me to choose, when you feel there is need to significantly deviate from architecture or approach.