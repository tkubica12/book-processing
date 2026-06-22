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

## Book-processing workflow

- Treat `input\arxiv\*.pdf` as arXiv papers. There are never folders under `input\arxiv\`; ignore anything there that is not a direct PDF file.
- Treat every other supported top-level file or folder under `input\` as a book. Single-file books can be PDF, EPUB, Markdown/text, or audio. A top-level folder is one book only when it contains supported audio files.
- Stage 1 writes `output\<book_name>\<book_name>_source_raw.md`, mirrors it to `wiki\<book_name>.md`, and writes `output\<book_name>\metadata.yaml`.
- `metadata.yaml` is the source of truth for `source_path`, `document_type`, `source_medium`, and labels. Valid `document_type` values are `book` and `paper`; valid labels are `AI`, `security`, `computers`, `biology`, `physics`, `technology`, `psychology`, and `other`.
- Papers generate only `summary_5min`, `summary_20min`, and `podcast_20min`. Do not generate 60-minute podcasts or full `source_tts` for papers.
- Run the processing pipeline with `uv run --no-sync book-processing` when refreshing content. It resumes existing outputs and regenerates the static catalog at the end.
- Run only the static web catalog generator with `uv run --no-sync book-processing-site` when source outputs are already current and only `output\index.html` / per-book `index.html` need refreshing.
- Run the full test suite with `uv run --no-sync pytest -q` before committing code changes.

## Private web publishing workflow

- The private site is served from Azure Container Apps at `https://books.tomasonline.net/`.
- Static files and MP3s live in private Azure Blob Storage account `booksite673af34d6b`, container `books`, with Blob access tier set to Cold.
- ACA runs `ghcr.io/tkubica12/book-processing/book-processing-site:latest`; do not use ACR unless bootstrapping an emergency replacement image.
- Browser authentication is handled by custom in-app GitHub OAuth. Do not enable ACA Easy Auth. Blob authentication is handled server-side by the ACA system-assigned managed identity with `Storage Blob Data Reader`.
- Do not expose blobs publicly, do not use SAS links in generated HTML, and do not depend on Azure Files mounts; this tenant disables shared-key access and ACA Azure Files mounts need shared keys.
- To deploy or refresh infrastructure/content, prefer `.\scripts\deploy-web.ps1`. The script regenerates the site, creates/updates Azure resources, uploads `output\`, and enforces Cold tier.
- If doing a manual content-only refresh, use AzCopy with Entra auth:

```powershell
azcopy login --tenant-id <tenant-id>
azcopy sync output https://booksite673af34d6b.blob.core.windows.net/books --recursive=true --delete-destination=true
azcopy set-properties https://booksite673af34d6b.blob.core.windows.net/books --block-blob-tier=Cold --recursive=true
```

- After deployment, smoke-test unauthenticated access with `curl.exe -I -L --max-redirs 0 https://books.tomasonline.net/`; expected result is `302 Found` redirecting to GitHub OAuth.
