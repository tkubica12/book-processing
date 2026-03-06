# book-processing

PDF-to-Markdown, technical summarization, translation, and Azure TTS pipeline for long-form books.

## What it does

- Treats every PDF in `input\` as a separate book, converts it with `markitdown`, and writes a cleaned per-book source file to `output\`.
- Uses Azure OpenAI (`gpt-5.2`) to generate English and Czech outputs:
  - `summary_2min`
  - `summary_5min`
  - `summary_20min`
  - `podcast_60min`
  - `source_tts`
- Uses Azure Speech Batch Synthesis to create matching MP3 files with Dragon HD voices.

## Output files

Each output filename includes the sanitized source PDF name. For example, `Inference Engineering.pdf` becomes `inference_engineering_*`.

Text outputs per book:

- `output\<book_name>_source_raw.md`
- `output\<book_name>_summary_2min_en.md`, `output\<book_name>_summary_2min_cs.md`
- `output\<book_name>_summary_5min_en.md`, `output\<book_name>_summary_5min_cs.md`
- `output\<book_name>_summary_20min_en.md`, `output\<book_name>_summary_20min_cs.md`
- `output\<book_name>_podcast_60min_en.md`, `output\<book_name>_podcast_60min_cs.md`
- `output\<book_name>_source_tts_en.md`, `output\<book_name>_source_tts_cs.md`

Audio outputs per book:

- `output\<book_name>_summary_2min_en.mp3`, `output\<book_name>_summary_2min_cs.mp3`
- `output\<book_name>_summary_5min_en.mp3`, `output\<book_name>_summary_5min_cs.mp3`
- `output\<book_name>_summary_20min_en.mp3`, `output\<book_name>_summary_20min_cs.mp3`
- `output\<book_name>_podcast_60min_en.mp3`, `output\<book_name>_podcast_60min_cs.mp3`
- `output\<book_name>_source_tts_en.mp3`, `output\<book_name>_source_tts_cs.mp3`

## Requirements

- Python 3.11+
- `uv`
- Azure login that works with `DefaultAzureCredential`
- Access to:
  - Azure OpenAI endpoint: `https://sw-v2-project-resource.cognitiveservices.azure.com`
  - Azure Speech endpoint: `https://sw-v2-project-resource.cognitiveservices.azure.com`

Login before running:

```powershell
az login
```

## Install

```powershell
uv sync
```

## Run

Place one or more PDF files into `input\`, then run:

```powershell
uv run book-processing
```

Or:

```powershell
uv run python -m book_processing.main
```

## Parallelism and reliability

- PDFs are treated as independent books and processed in parallel batches.
- PDF conversion runs first, then LLM generation and TTS run overlapped.
- LLM work is flattened into independent tasks and processed concurrently within each book.
- Full-length TTS files are split into smaller chunks and synthesized in parallel.
- TTS access tokens are cached to avoid repeated Azure CLI calls during polling.
- Failed Azure batch TTS chunks are retried individually instead of aborting the entire file immediately.
- Existing text and audio outputs are reused per book, so reruns resume work instead of regenerating everything.

## Development

Run tests:

```powershell
uv run pytest -q
```
