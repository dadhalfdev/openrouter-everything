![OpenRouter Everything logo: a neon lime mark over a globe](assets/openrouter-everything-readme.png)

# 🌐 OpenRouter Everything

Want your Hermes agent to search the web, create images, generate speech and more without setting up a different API key for each service?

**OpenRouter Everything connects these tools to one OpenRouter API key.** Pick a model for each job, switch it when you need to, and check whether a model supports the task before using it.

## 💡 Perfect for...

- **🤖 AI agents:** Give Hermes web search, page fetching, image and video generation, and voice tools through OpenRouter.
- **🔧 Builders:** Start with the included defaults, then choose different models for the jobs you care about.
- **📚 RAG workflows:** Generate embeddings and rerank passages before using them in a retrieval pipeline.
- **📋 Structured workflows:** Use the Decisions and Extractions tools to classify inputs and extract fields from text.

## ✨ What's included

- **🔑 One API key:** The plugin uses `OPENROUTER_API_KEY` for its OpenRouter requests. OpenRouter usage charges still apply.
- **🌐 Search and fetch:** Search with `openrouter:web_search` and fetch pages with `openrouter:web_fetch`. Both have configurable engines.
- **🎨 Images and video:** Use Hermes generation tools with an OpenRouter model selected for each.
- **🎙️ Voice:** Generate speech from text and transcribe audio with separate TTS and STT models.
- **🎯 Model checks:** The plugin checks OpenRouter's model catalog and reports known mismatches, such as an image model selected for video generation.
- **🧩 Extra tools:** Make decisions, extract structured fields, rerank documents and create embeddings.

## 📦 Default models and tools

You can change each model. The search and fetch defaults use DeepSeek to call OpenRouter's server tools.

| Capability | Hermes surface | Default model or tool | OpenRouter endpoint |
| --- | --- | --- | --- |
| Text setting | `text` | `deepseek/deepseek-v4.1-flash-1017` | `POST /api/v1/chat/completions` |
| Web search | `web_search` provider | `openrouter:web_search`, DeepSeek, `engine=auto` | `POST /api/v1/chat/completions` |
| Web fetch | `web_extract` provider | `openrouter:web_fetch`, DeepSeek, `engine=openrouter` | `POST /api/v1/chat/completions` |
| Image generation | `image_generate` provider | `google/gemini-nano-banana-2.1` | `POST /api/v1/chat/completions` or `POST /api/v1/images/generations` |
| Video generation | `video_generate` provider | `alibaba/wan-3.0-prime` | `POST /api/v1/videos` |
| Text to speech | `text_to_speech` provider | `fish-audio/drama-3` | `POST /api/v1/audio/speech` |
| Speech to text | `transcribe_audio` provider | `openai/whisper-large-v3-turbo` | `POST /api/v1/audio/transcriptions` |
| Decisions | `openrouter_decide` tool | `typesafe/jev-1.14` | `POST /api/alpha/decisions` |
| Field extraction | `openrouter_extract_fields` tool | `cloudflare/clairic` | `POST /api/alpha/extractions` |
| Reranking | `openrouter_rerank` tool | `cohere/rerank-v3.5` | `POST /api/v1/rerank` |
| Embeddings | `openrouter_embed` tool | `google/gemini-embedding-2` | `POST /api/v1/embeddings` |

The plugin's `text` setting does not change your main Hermes chat model. Configure that separately in Hermes. Models and their availability can change on OpenRouter.

## 🚀 Quick start

You need [Hermes Agent](https://hermes-agent.nousresearch.com/docs) and an [OpenRouter API key](https://openrouter.ai/settings/keys).

```bash
# Install directly from GitHub and enable the plugin
hermes plugins install https://github.com/dadhalfdev/openrouter-everything
hermes plugins enable openrouter-everything

# Configure the key and activate the OpenRouter Everything providers
hermes openrouter-everything setup

# See the configured models and active providers
hermes openrouter-everything status
```

`setup` changes the active Hermes providers for web, image generation, video generation, TTS and STT. Review those changes before running it if you already use other providers. The interactive setup asks for your OpenRouter key when needed. API requests can incur charges based on the model, tool and usage.

## 🔄 Switch models and search engines

Use the CLI, the `/openrouter-everything` slash command, or the `openrouter_everything_config` agent tool to inspect and change model settings.

```bash
# List compatible models
hermes openrouter-everything models video_gen
hermes openrouter-everything models stt

# Pick a different speech-to-text model
hermes openrouter-everything set-model stt mistralai/voxtral-mini-transcribe-3-0

# Change the web search engine
hermes openrouter-everything set-engine web_search parallel

# Check an image model against the video task
hermes openrouter-everything validate video_gen google/gemini-nano-banana-2.1
```

Search engines: `auto`, `exa`, `parallel`, `perplexity`, `firecrawl` and `native`. Fetch engines: `openrouter`, `auto`, `exa`, `parallel`, `firecrawl` and `native`.

When the catalog is available, an incompatible model produces an explanation and suggested alternatives. If the catalog can't be reached, the plugin lets the request proceed and OpenRouter must validate it. A successful check doesn't guarantee that a model is available to your account or that a request will succeed.

## 🔐 Privacy and disclosure

- **Network:** Requests normally go to OpenRouter's `/api/v1/*` and `/api/alpha/*` endpoints. The API base URL can be overridden. Generated media may also be downloaded from URLs returned by the service.
- **Credentials:** The plugin reads `OPENROUTER_API_KEY` through Hermes's profile-scoped secret access. Interactive setup can save the key through Hermes configuration.
- **Files:** The plugin reads local reference images and audio files passed to its tools. It saves generated media and changes Hermes configuration when you run commands such as `setup`, `set-model` or `set-engine`.
- **Background activity:** The plugin does not add a background process, run shell commands, send telemetry or update itself.

Web fetch is model-mediated: the model receives the fetched content and returns a response, so treat the output as potentially incomplete or altered rather than a guaranteed verbatim copy.
