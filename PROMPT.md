Create a Python pipeline that processes all PDFs in `input\` and produces cleaned Markdown, technical long-form summaries, and audio outputs in `output\`.

Treat each PDF as a separate book/work item. Do not merge PDFs together. Each PDF in `input\` must go through the full pipeline independently, and the pipeline should be able to process multiple books in parallel batches.

Use the `markitdown` library for PDF to Markdown conversion. Convert each PDF into its own Markdown document and save a per-book raw source Markdown file. The raw Markdown should still be human-usable and LLM-usable: clean obvious PDF artifacts, remove boring front matter and table-of-contents noise, remove page numbers and repeated headers/footers, normalize headings where possible, and preserve meaningful technical content. If tables or diagrams are poorly extracted, improve the Markdown where practical rather than keeping unreadable garbage.

From each book's cleaned source, generate multiple architect-grade outputs with Azure OpenAI using model `gpt-5.2` at `https://sw-v2-project-resource.cognitiveservices.azure.com`. The audience is technical and senior, so focus on principles, architecture, system design, trade-offs, implementation mechanics, inference/runtime behavior, and technological depth. Avoid fluffy business summaries. Generate these text outputs in both English and Czech, and include both the original PDF name and language in the file name.

Assume the output naming pattern should include a sanitized PDF stem, for example:

- `{book_name}_source_raw.md`
- `{book_name}_summary_2min_{lang}.md`
- `{book_name}_summary_5min_{lang}.md`
- `{book_name}_summary_20min_{lang}.md`
- `{book_name}_podcast_60min_{lang}.md`
- `{book_name}_source_tts_{lang}.md`

Treat the requested durations as target listening durations. Use a speaking-speed calibration suitable for approximately `+20%` speech rate, and size outputs accordingly so they are not obviously too long. The 60-minute podcast should still be content-dense, but it must fit the target better than a naive overlong transcript.

The podcast must be written as a dialogue with explicit speaker markers so TTS can switch voices correctly. Use clear tags such as `[Andrew]:` and `[Emma]:` in English, and corresponding Czech speaker tags in Czech. The hosts should sound technically sophisticated and conversational, challenge each other, explain trade-offs, occasionally be playful or geeky, but remain accurate and information-dense.

The `{book_name}_source_tts_{lang}.md` files are not simple copies of the raw source. They must be optimized for listening:

- remove boring or repetitive material such as long contents pages, indexes, navigation junk, or page artifacts
- rewrite tables into concise spoken descriptions of the key findings instead of reading cells row by row
- summarize charts, figures, and hard-to-read visual structures into natural spoken prose
- remove or rewrite text that sounds bad aloud, such as raw URLs, fragmented captions, repeated labels, and broken formatting
- preserve the full technical substance of the source, but make it pleasant and useful for long-form listening

Generate audio for all text outputs with Azure Speech Batch Synthesis using the same resource endpoint `https://sw-v2-project-resource.cognitiveservices.azure.com`. Use Entra authentication via `DefaultAzureCredential` rather than custom tenant-specific configuration. Use Dragon HD voices:

- male/default narration: `en-US-Andrew:DragonHDLatestNeural`
- female/podcast co-host: `en-US-Emma:DragonHDLatestNeural`

These voices are multilingual; switch language via SSML language settings rather than changing the voice identity. Use the male voice for summaries and full-book narration. Use both male and female voices for podcasts. Produce MP3 outputs for every text artifact in both English and Czech, again including the source PDF name in each output filename, for example `{book_name}_summary_2min_en.mp3`.

Design the pipeline for reliability and restartability, not just a happy path:

- support rerunning the pipeline without regenerating outputs that already exist and are valid, on a per-book basis
- handle transient API failures, network issues, and 429 rate limits with retries and backoff
- make long-running work restartable at a granular level so one failed chunk does not force a full restart
- log progress clearly by stage, by book, and by output artifact
- prefer explicit failure over silent data loss

Parallelize aggressively to reduce wall-clock time:

- process multiple books in parallel batches
- within a single book, run independent LLM tasks in parallel rather than sequentially
- break long outputs such as podcast generation or full-book TTS preparation into smaller independent sections when needed
- start TTS work as soon as upstream text artifacts become available instead of waiting for all LLM work to finish
- split large TTS jobs into smaller chunks, synthesize those chunks in parallel, and stitch the audio back together in order
- keep chunking small enough to improve reliability for long-form synthesis

The final deliverable is a complete pipeline that can take multiple PDFs from `input\` and, for each PDF independently, produce:

- cleaned per-book source Markdown
- all requested English and Czech Markdown outputs
- all requested English and Czech MP3 outputs

All filenames should clearly indicate the source book, content type, and language.
