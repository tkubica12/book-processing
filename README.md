# Book processing

Personal pipeline for turning long PDFs, EPUBs, Markdown files, or supported audio sources into a spiral learning workflow across reading and listening formats.

The main goal is not generic document processing. It is to help me move through a book in layers so I can both save time and understand the material better by getting the high-level context first, before diving into details and losing the forest for the trees.

## Why this exists

I want to take an eBook, PDF, Markdown source, or audiobook recording and automatically create AI-generated text and audio versions that support progressive understanding.

For every input book, the pipeline can produce:

- 5-minute summary
- 20-minute deeper summary
- 20-minute conversational podcast
- 60-minute deep podcast
- full-length audiobook
- cleaned-up Markdown version suitable as LLM context
- single-file English HTML visual map with progressive disclosure

The idea is spiral learning through reading and listening.

I usually start with a 5-minute summary to understand the overall shape of the material. If the topic looks promising, I move to the 20-minute version to get the important details. From there I either stop, continue to the 20-minute or 60-minute podcast for a more guided walkthrough, or go directly to the full-length version when the book is clearly worth the time.

That cleaned-up Markdown output serves a different purpose: it becomes a high-quality context file for AI tools such as Copilot notebooks or Perplexity spaces, so I can move from reading or listening straight into clarifying questions, follow-up exploration, or voice chat about the same source material.

## What the pipeline does

Each supported file in `input\` is treated as a separate book, and each top-level subdirectory with supported audio files is treated as one combined audio book/podcast source.

- PDFs are extracted into Markdown with Azure Content Understanding, including document cleanup and richer structure from the source.
- EPUB inputs are converted into Markdown with MarkItDown and then fed into the same raw source flow.
- Markdown inputs are normalized into the same raw source flow so everything downstream behaves the same way.
- Supported audio inputs (`.mp3`, `.m4b`) are transcribed to Markdown in Stage 1 and saved into the same raw source flow as document inputs. A folder such as `input\mybook\*` is treated as one logical audio source, only audio files inside it are processed, and those files are combined in deterministic filename order into one raw output.
- LLM processing generates summaries, translations, podcast-style scripts, and full-length text adapted for speech.
- Text-to-speech generates matching audio files for the same hierarchy of outputs.

In practice, this means a single source document becomes multiple entry points into the same content, from extremely short overview to near-complete coverage.

## Output structure

Each book gets its own subfolder named after the sanitized source filename stem. For example, `Inference Engineering.pdf` becomes `output\inference_engineering\...`.

Text outputs per book:

- `output\<book_name>\<book_name>_source_raw.md`
- `output\<book_name>\<book_name>_summary_5min_en.md`, `output\<book_name>\<book_name>_summary_5min_cs.md`
- `output\<book_name>\<book_name>_summary_20min_en.md`, `output\<book_name>\<book_name>_summary_20min_cs.md`
- `output\<book_name>\<book_name>_podcast_20min_en.md`, `output\<book_name>\<book_name>_podcast_20min_cs.md`
- `output\<book_name>\<book_name>_podcast_60min_en.md`, `output\<book_name>\<book_name>_podcast_60min_cs.md`
- `output\<book_name>\<book_name>_source_tts_en.md`, `output\<book_name>\<book_name>_source_tts_cs.md`
- `output\<book_name>\<book_name>_visual_summary_en.html`

Audio outputs per book:

- `output\<book_name>\<book_name>_summary_5min_en.mp3`, `output\<book_name>\<book_name>_summary_5min_cs.mp3`
- `output\<book_name>\<book_name>_summary_20min_en.mp3`, `output\<book_name>\<book_name>_summary_20min_cs.mp3`
- `output\<book_name>\<book_name>_podcast_20min_en.mp3`, `output\<book_name>\<book_name>_podcast_20min_cs.mp3`
- `output\<book_name>\<book_name>_podcast_60min_en.mp3`, `output\<book_name>\<book_name>_podcast_60min_cs.mp3`
- `output\<book_name>\<book_name>_source_tts_en.mp3`, `output\<book_name>\<book_name>_source_tts_cs.mp3`

## Very short technical summary

This project uses:

- Azure Content Understanding for PDF to Markdown extraction and cleanup
- MarkItDown for EPUB to Markdown conversion
- speech-to-text transcription for supported audio inputs, normalized into `source_raw`
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

- `AZURE_OPENAI_ENDPOINT` if your chat/completions resource differs from the built-in default
- `AZURE_OPENAI_MODEL` if your Azure OpenAI deployment name differs from the built-in default
- `AZURE_SPEECH_ENDPOINT` if your Speech / fast transcription resource differs from the built-in default
- `AZURE_SPEECH_TRANSCRIPTION_MODEL` default: empty; set `mai-transcribe-1.5` only on endpoints where MAI-Transcribe / LLM Speech is enabled
- `AZURE_SPEECH_TRANSCRIPTION_LOCALES` default: empty auto-detect; set comma-separated locales only if the target STT endpoint accepts them
- `VOICE_MALE` / `VOICE_FEMALE` default to English MAI-Voice-2 voices for English audio generation
- `CONTENT_UNDERSTANDING_ANALYZER_ID` default: `prebuilt-documentSearch`
- `CONTENT_UNDERSTANDING_API_VERSION` default: `2025-11-01`
- `CONTENT_UNDERSTANDING_PROCESSING_LOCATION`
- `CONTENT_UNDERSTANDING_API_KEY` if you prefer key auth over Entra auth

Install dependencies:

```powershell
uv sync
```

## Very short run

Put one or more `.pdf`, `.epub`, `.md`, `.txt`, `.mp3`, or `.m4b` files into `input\`, or place multi-file audio books/podcasts inside their own subdirectory under `input\`, then run:

```powershell
uv run book-processing
```

Alternative entry point:

```powershell
uv run python -m book_processing.main
```

## Notes

- Stage 1 always produces `output\<book_name>\<book_name>_source_raw.md`, whether the input started as a document, a single audio file, or a folder of ordered audio tracks.
- Stage 1 also mirrors that normalized raw Markdown to `wiki\<book_name>.md` at the repo root.
- Existing outputs are reused, so reruns resume instead of regenerating everything.
- The English HTML visual summary is generated from `source_raw` and also runs for existing `output\<book_name>\*_source_raw.md` folders, even when there is no current matching input file.
- A private static catalog is generated into `output\index.html` plus `output\<book_name>\index.html` so the whole output folder can be served directly.
- Work is parallelized across books and across many per-book generation steps.
- Full-length TTS is chunked and retried to make long audio generation more reliable.
- Generated MP3 files include ID3 metadata with book title, content type, and language code.

## Private web site - caveman runbook

Goal: private internet URL, Entra login, only `tomas@tomasonline.net`, big MP3 files work.

Architecture:

- Azure Blob Storage = private storage for `output\`.
- Blob access tier = Cold for lower monthly storage cost.
- Azure Container Apps = runs a tiny Python web server.
- Managed identity = web server reads private blobs.
- GitHub Container Registry = stores the web server image once the package is public.
- Container Apps Easy Auth = login wall.
- Entra app assignment = only Tomas gets in.
- No Azure Storage static website. That is public anonymous.
- No Azure Static Web Apps. Too small for this output.
- No Azure Files. This tenant blocks shared-key access, and Container Apps Azure Files mount needs it.
- ACR was used for bootstrap, but should be removed after ACA is switched to public GHCR image.

Deploy or refresh:

```powershell
azcopy login --tenant-id <tenant-id>
.\scripts\deploy-web.ps1
```

The image is built by GitHub Actions workflow `.github\workflows\publish-web-image.yml`.
Make the GHCR package public before switching ACA to it; the container image excludes `input\`, `output\`, `wiki\`, `.env`, and local logs via `.dockerignore`.

What it does:

1. Runs `uv run book-processing-site`.
2. Creates `output\index.html`.
3. Creates every `output\<book_name>\index.html`.
4. Creates private Blob Storage container.
5. Uses `ghcr.io/tkubica12/book-processing/book-processing-site:latest` as the web server image.
6. Creates Container Apps environment and app.
7. Gives the app managed identity read access to private blobs.
8. Enables Microsoft Entra login.
9. Assigns only `tomas@tomasonline.net`.
10. Creates DNS records and binds `books.tomasonline.net` with managed TLS certificate.
11. Uploads `output\` to private Blob Storage.
12. Prints URL.

Useful knobs:

```powershell
.\scripts\deploy-web.ps1 -ResourceGroup rg-book-processing-site -Location westeurope
.\scripts\deploy-web.ps1 -CustomHostname books.tomasonline.net -DnsZoneName tomasonline.net -DnsZoneResourceGroup rg-base
.\scripts\deploy-web.ps1 -Image ghcr.io/tkubica12/book-processing/book-processing-site:latest
.\scripts\deploy-web.ps1 -SkipUpload
uv run book-processing-site
```

After GHCR works and ACA is using the public image, delete the old bootstrap ACR:

```powershell
az acr delete -g rg-book-processing-site -n acrbooksite673af34d6b --yes
```

Smoke test:

```powershell
curl.exe -I https://<printed-url>
```

Good result without browser login: HTTP 302 to `/.auth/login/aad`.
Also OK: HTTP 401 with a `www-authenticate` header pointing to Microsoft login.

## Development

Prompt templates live in `prompts\*.j2`.

Run tests with:

```powershell
uv run pytest -q
```
