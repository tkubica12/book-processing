# book-processing

PDF/Markdown ingestion, technical summarization, translation, and Azure TTS pipeline for long-form books.

## What it does

- Treats every supported source file in `input\` as a separate book.
- Converts PDFs with Azure Content Understanding in Foundry Tools and writes a cleaned per-book source file to `output\`.
- Reuses Markdown inputs as-is by copying them into the standard per-book raw source file in `output\`.
- Uses Azure OpenAI (`gpt-5.2`) to generate English and Czech outputs:
  - `summary_2min`
  - `summary_5min`
  - `summary_20min`
  - `podcast_60min`
  - `source_tts`
- Uses Azure Speech Batch Synthesis to create matching MP3 files with Dragon HD voices.

## Output files

Each output filename includes the sanitized source filename stem. For example, `Inference Engineering.pdf` or `Inference Engineering.md` becomes `inference_engineering_*`.

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
- Azure login that can obtain Cognitive Services Entra tokens with `az login`
- Access to:
  - Azure AI Foundry resource with Content Understanding enabled in a supported region
  - Content Understanding endpoint: `https://<your-resource>.services.ai.azure.com/`
  - Azure OpenAI endpoint: `https://sw-v2-project-resource.cognitiveservices.azure.com`
  - Azure Speech endpoint: `https://sw-v2-project-resource.cognitiveservices.azure.com`

For Content Understanding document-to-Markdown extraction, the Foundry resource should have default model deployments configured for the prebuilt analyzer flow. For `prebuilt-documentSearch`, Microsoft’s samples call out `gpt-4.1-mini` and `text-embedding-3-large`. Enabling autodeployment for required models is the simplest setup.

The principal or user running the pipeline needs access to call the Foundry resource. In practice, grant `Cognitive Services User` on that Foundry resource.

Login before running:

```powershell
az login
```

## Environment

Copy `.env.example` to `.env` and fill in the Content Understanding settings:

```powershell
Copy-Item .env.example .env
```

Required values:

- `CONTENT_UNDERSTANDING_ENDPOINT`: your Foundry endpoint, for example `https://my-foundry.services.ai.azure.com/`

Optional values:

- `CONTENT_UNDERSTANDING_ANALYZER_ID`: defaults to `prebuilt-documentSearch`
- `CONTENT_UNDERSTANDING_API_VERSION`: defaults to `2025-11-01`
- `CONTENT_UNDERSTANDING_PROCESSING_LOCATION`: optional `global`, `geography`, or `dataZone`
- `CONTENT_UNDERSTANDING_API_KEY`: only if you want key auth instead of Entra auth

The default and recommended path is Entra auth via `az login`, with no API key in `.env`.

If you want guaranteed figure descriptions, chart extraction, or Mermaid/table figure analysis in the Markdown output, configure those options on the analyzer itself in Content Understanding and then point `CONTENT_UNDERSTANDING_ANALYZER_ID` at that copied/custom analyzer. The client in this repo simply calls the analyzer you specify and reads `result.contents[0].markdown`.

## Install

```powershell
uv sync
```

## Run

Place one or more `.pdf` or `.md` files into `input\`, then run:

```powershell
uv run book-processing
```

Or:

```powershell
uv run python -m book_processing.main
```

## Parallelism and reliability

- Input files are treated as independent books and processed in parallel batches.
- PDF conversion runs first through Content Understanding, then LLM generation and TTS run overlapped.
- LLM work is flattened into independent tasks and processed concurrently within each book.
- Full-length TTS files are split into smaller chunks and synthesized in parallel.
- TTS access tokens are cached to avoid repeated Azure CLI calls during polling.
- Failed Azure batch TTS chunks are retried individually instead of aborting the entire file immediately.
- Existing text and audio outputs are reused per book, so reruns resume work instead of regenerating everything.
- Generated MP3 files include ID3 metadata with book title, content type, and language code.

## Azure Setup Summary

You need to deploy or configure:

1. An Azure AI Foundry resource in a Content Understanding-supported region.
2. Default model deployments for Content Understanding on that resource.
3. A Content Understanding analyzer to call: either `prebuilt-documentSearch` or your own copied/custom analyzer if you want explicit document-processing settings.
4. Access for your identity, at minimum `Cognitive Services User` on the Foundry resource.

I need from you:

1. The Foundry endpoint URI, for example `https://my-foundry.services.ai.azure.com/`.
2. Confirmation that `prebuilt-documentSearch` is acceptable, or the analyzer ID of your copied/custom analyzer if you want locked behavior and explicit figure-analysis settings.
3. Confirmation that your identity can call the resource with `az login`, or an API key if you want to use key auth instead.

## Development

LLM prompts live in `prompts\*.j2`, so prompt tuning can happen without hunting through Python code.

Run tests:

```powershell
uv run pytest -q
```
