# book-processing

Personal pipeline for turning long PDFs or Markdown files into a spiral learning workflow across reading and listening formats.

The main goal is not generic document processing. It is to help me move through a book in layers so I can both save time and understand the material better by getting the high-level context first, before diving into details and losing the forest for the trees.

## Why this exists

I want to take an eBook, PDF, or Markdown source and automatically create AI-generated text and audio versions that support progressive understanding.

For every input book, the pipeline can produce:

- 2-minute quick summary
- 5-minute summary
- 20-minute deeper summary
- 60-minute deep technical podcast
- full-length audiobook
- cleaned-up Markdown version suitable as LLM context

The idea is spiral learning through reading and listening.

I usually start with a 2-minute or 5-minute summary to understand the overall shape of the material. If the topic looks promising, I move to the 20-minute version to get the important details. From there I either stop, continue to the 60-minute podcast for a deeper guided walkthrough, or go directly to the full-length version when the book is clearly worth the time.

That cleaned-up Markdown output serves a different purpose: it becomes a high-quality context file for AI tools such as Copilot notebooks or Perplexity spaces, so I can move from reading or listening straight into clarifying questions, follow-up exploration, or voice chat about the same source material.

## What the pipeline does

Each file in `input\` is treated as a separate book.

- PDFs are extracted into Markdown with Azure Content Understanding, including document cleanup and richer structure from the source.
- Markdown inputs are normalized into the same raw source flow so everything downstream behaves the same way.
- LLM processing generates summaries, translations, podcast-style scripts, and full-length text adapted for speech.
- Text-to-speech generates matching audio files for the same hierarchy of outputs.

In practice, this means a single source document becomes multiple entry points into the same content, from extremely short overview to near-complete coverage.

## Output structure

Each output filename uses the sanitized source filename stem. For example, `Inference Engineering.pdf` becomes `inference_engineering_*`.

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

## Very short technical summary

This project uses:

- Azure Content Understanding for PDF to Markdown extraction and cleanup
- Azure OpenAI for summarization, translation, and podcast/text generation
- Azure Speech Batch Synthesis for audio generation
- `uv` for Python environment and dependency management

## Very short setup

Requirements:

- Python 3.11+
- `uv`
- `az login`
- access to an Azure AI Foundry resource with Content Understanding enabled

Create `.env` from `.env.example` and set at least:

- `CONTENT_UNDERSTANDING_ENDPOINT`

Optional settings include:

- `CONTENT_UNDERSTANDING_ANALYZER_ID` default: `prebuilt-documentSearch`
- `CONTENT_UNDERSTANDING_API_VERSION` default: `2025-11-01`
- `CONTENT_UNDERSTANDING_PROCESSING_LOCATION`
- `CONTENT_UNDERSTANDING_API_KEY` if you prefer key auth over Entra auth

Install dependencies:

```powershell
uv sync
```

## Very short run

Put one or more `.pdf` or `.md` files into `input\`, then run:

```powershell
uv run book-processing
```

Alternative entry point:

```powershell
uv run python -m book_processing.main
```

## Notes

- Existing outputs are reused, so reruns resume instead of regenerating everything.
- Work is parallelized across books and across many per-book generation steps.
- Full-length TTS is chunked and retried to make long audio generation more reliable.
- Generated MP3 files include ID3 metadata with book title, content type, and language code.

## Development

Prompt templates live in `prompts\*.j2`.

Run tests with:

```powershell
uv run pytest -q
```
