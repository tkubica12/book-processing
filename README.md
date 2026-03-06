# book-processing

PDF-to-Markdown, technical summarization, translation, and Azure TTS pipeline for long-form books.

## What it does

- Converts every PDF in `input\` to Markdown with `markitdown`, concatenates them, and writes a cleaned source file to `output\source_raw.md`.
- Uses Azure OpenAI (`gpt-5.2`) to generate English and Czech outputs:
  - `summary_2min`
  - `summary_5min`
  - `summary_20min`
  - `podcast_60min`
  - `source_tts`
- Uses Azure Speech Batch Synthesis to create matching MP3 files with Dragon HD voices.

## Output files

Text outputs:

- `output\source_raw.md`
- `output\summary_2min_en.md`, `output\summary_2min_cs.md`
- `output\summary_5min_en.md`, `output\summary_5min_cs.md`
- `output\summary_20min_en.md`, `output\summary_20min_cs.md`
- `output\podcast_60min_en.md`, `output\podcast_60min_cs.md`
- `output\source_tts_en.md`, `output\source_tts_cs.md`

Audio outputs:

- `output\summary_2min_en.mp3`, `output\summary_2min_cs.mp3`
- `output\summary_5min_en.mp3`, `output\summary_5min_cs.mp3`
- `output\summary_20min_en.mp3`, `output\summary_20min_cs.mp3`
- `output\podcast_60min_en.mp3`, `output\podcast_60min_cs.mp3`
- `output\source_tts_en.mp3`, `output\source_tts_cs.mp3`

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

- PDF conversion runs first, then LLM generation and TTS run overlapped.
- LLM work is flattened into independent tasks and processed concurrently.
- Full-length TTS files are split into smaller chunks and synthesized in parallel.
- TTS access tokens are cached to avoid repeated Azure CLI calls during polling.
- Failed Azure batch TTS chunks are retried individually instead of aborting the entire file immediately.
- Existing text and audio outputs are reused, so reruns resume work instead of regenerating everything.

## Development

Run tests:

```powershell
uv run pytest -q
```
