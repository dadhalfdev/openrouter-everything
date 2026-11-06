# OpenRouter Everything (`openrouter-everything`)

One `OPENROUTER_API_KEY` for every Hermes Agent modality — **Web Search (`openrouter:web_search`)**, **Web Fetch (`openrouter:web_fetch`)**, **Image Generation**, **Video Generation**, **Text-to-Speech (TTS)**, **Speech-to-Text (STT)**, **Probabilistic Decisions**, **Structured Field Extractions**, **Reranking**, and **Embeddings** — with live **model-task adequacy validation** that blocks mismatched models before spending tokens.

## Why

Setting up a multi-modal agent usually requires juggling 6+ separate API keys (one for web search, one for image generation, one for video, one for TTS, one for STT, etc.). OpenRouter exposes all of these capabilities under a single API key. `openrouter-everything` wires all of them into Hermes with curated defaults, per-action model switching, and a live adequacy guard so users cannot accidentally pass an image model to the video tool or a chat model to TTS.

## Curated Defaults

| Capability | Hermes Surface | Default Model / Tool | Endpoint |
|---|---|---|---|
| **Text / Fast Aux** | `text` | `deepseek/deepseek-v4.1-flash-1017` | `POST /api/v1/chat/completions` |
| **Web Search** | `web_search` provider | `openrouter:web_search` (`deepseek/deepseek-v4.1-flash-1017`, `engine=auto`) | `POST /api/v1/chat/completions` (`tools: [openrouter:web_search]`) |
| **Web Extract** | `web_extract` provider | `openrouter:web_fetch` (`deepseek/deepseek-v4.1-flash-1017`, `engine=openrouter`) | `POST /api/v1/chat/completions` (`tools: [openrouter:web_fetch]`) |
| **Image Gen** | `image_generate` provider | `google/gemini-nano-banana-2.1` | `POST /api/v1/chat/completions` / `POST /api/v1/images/generations` |
| **Video Gen** | `video_generate` provider | `alibaba/wan-3.0-prime` | `POST /api/v1/videos` |
| **Text-to-Speech** | `text_to_speech` provider | `fish-audio/drama-3` | `POST /api/v1/audio/speech` |
| **Speech-to-Text** | `transcribe_audio` provider | `openai/whisper-large-v3-turbo` | `POST /api/v1/audio/transcriptions` |
| **Decisions** | `openrouter_decide` tool | `typesafe/jev-1.14` | `POST /api/alpha/decisions` |
| **Extractions** | `openrouter_extract_fields` tool | `cloudflare/clairic` | `POST /api/alpha/extractions` |
| **Rerank** | `openrouter_rerank` tool | `cohere/rerank-v3.5` | `POST /api/v1/rerank` |
| **Embeddings** | `openrouter_embed` tool | `google/gemini-embedding-2` | `POST /api/v1/embeddings` |

## Quick Start

```bash
# 1. Install the plugin
hermes plugins install openrouter-everything
hermes plugins enable openrouter-everything

# 2. One-step setup: set OPENROUTER_API_KEY and activate across web, image_gen, video_gen, tts, and stt
hermes openrouter-everything setup

# 3. Check status and adequacy across all 11 capabilities
hermes openrouter-everything status
```

## Switching Models & Adequacy Validation

You can switch the model for any action via CLI, slash command (`/openrouter-everything`), or the `openrouter_everything_config` agent tool:

```bash
# List compatible OpenRouter models for a capability
hermes openrouter-everything models video_gen
hermes openrouter-everything models stt

# Switch the STT model to Mistral Voxtral Mini Transcribe 3.0
hermes openrouter-everything set-model stt mistralai/voxtral-mini-transcribe-3-0

# Switch the openrouter:web_search engine (auto | exa | parallel | perplexity | firecrawl | native)
hermes openrouter-everything set-engine web_search parallel

# Validate whether a model is adequate for a task
hermes openrouter-everything validate video_gen google/gemini-nano-banana-2.1
```

When a user selects an inadequate model for a task (for example, `google/gemini-nano-banana-2.1` for `video_gen`), the plugin blocks the assignment/request and explains what the model actually supports alongside recommended alternatives.

## Disclosure (Catalog Rule 13)

- **Network calls:** Sends HTTPS requests exclusively to `https://openrouter.ai/api/v1/*` and `https://openrouter.ai/api/alpha/*` (`decisions`, `extractions`) authenticated with `OPENROUTER_API_KEY` resolved via `agent.secret_scope.get_secret("OPENROUTER_API_KEY")`.
- **File reads/writes:** Reads local reference images or audio files explicitly passed by the user/agent to `image_generate`, `video_generate`, or `transcribe_audio`. Writes generated images/videos/audio through Hermes's standard `save_*` provider helpers and updates `config.yaml` when the user invokes `set-model`, `set-engine`, or `setup`.
- **No background processes, no shell execution, no telemetry, no self-updating code.**
