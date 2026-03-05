"""Centralized configuration for the book processing pipeline."""

from pathlib import Path

# === Project Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT = "https://sw-v2-project-resource.cognitiveservices.azure.com"
AZURE_OPENAI_MODEL = "gpt-5.2"
AZURE_OPENAI_API_VERSION = "2025-04-01-preview"

# === Azure Speech (Batch Synthesis API) ===
AZURE_SPEECH_ENDPOINT = "https://sw-v2-project-resource.cognitiveservices.azure.com"
AZURE_SPEECH_API_VERSION = "2024-04-01"
AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# === Voice Configuration ===
# Dragon HD voices are multilingual — same voices for EN and CZ
VOICE_MALE = "en-US-Andrew:DragonHDLatestNeural"
VOICE_FEMALE = "en-US-Emma:DragonHDLatestNeural"

# === SSML / Prosody ===
PROSODY_RATE = "+30%"
AUDIO_OUTPUT_FORMAT = "audio-48khz-192kbitrate-mono-mp3"

# === Language Configuration ===
LANGUAGES = {
    "en": {"xml_lang": "en-US", "label": "English"},
    "cs": {"xml_lang": "cs-CZ", "label": "Czech"},
}

# === Podcast Speaker Tags ===
PODCAST_SPEAKERS = {
    "en": {"male": "Andrew", "female": "Emma"},
    "cs": {"male": "Tomáš", "female": "Kateřina"},
}

# === Word Count Targets (at 130% speaking speed ≈ 195 WPM) ===
WPM_AT_130_PERCENT = 195

SUMMARY_TYPES = {
    "summary_2min": {
        "duration_min": 2,
        "target_words": 2 * WPM_AT_130_PERCENT,  # ~390
        "description": "High-level architectural summary",
        "is_podcast": False,
    },
    "summary_5min": {
        "duration_min": 5,
        "target_words": 5 * WPM_AT_130_PERCENT,  # ~975
        "description": "Architecture patterns, trade-offs, technology choices",
        "is_podcast": False,
    },
    "summary_20min": {
        "duration_min": 20,
        "target_words": 20 * WPM_AT_130_PERCENT,  # ~3900
        "description": "Technical deep-dive with implementation details",
        "is_podcast": False,
    },
    "podcast_60min": {
        "duration_min": 60,
        "target_words": 60 * WPM_AT_130_PERCENT,  # ~11700
        "description": "Two-host technical podcast, entertaining and deeply technical",
        "is_podcast": True,
    },
}

# === File Naming ===
def output_text_path(name: str, lang: str) -> Path:
    """Return the output path for a text file, e.g. output/summary_2min_en.md."""
    return OUTPUT_DIR / f"{name}_{lang}.md"


def output_audio_path(name: str, lang: str) -> Path:
    """Return the output path for an audio file, e.g. output/summary_2min_en.mp3."""
    return OUTPUT_DIR / f"{name}_{lang}.mp3"


# Names for the full-length TTS-preprocessed source
SOURCE_RAW_NAME = "source_raw"
SOURCE_TTS_NAME = "source_tts"
