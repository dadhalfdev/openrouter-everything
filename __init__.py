"""OpenRouter Everything Plugin — One API Key for Web Search/Fetch, Image Gen, Video Gen,
TTS, STT, Decisions, Extractions, Rerank, and Embeddings with model-task adequacy validation."""

from __future__ import annotations

from .catalog import has_credentials
from .cli import handle_cli, handle_slash, setup_cli_parser
from .providers import (
    OpenRouterEverythingImageProvider,
    OpenRouterEverythingSTTProvider,
    OpenRouterEverythingTTSProvider,
    OpenRouterEverythingVideoProvider,
    OpenRouterEverythingWebProvider,
)
from .tools import (
    CONFIG_TOOL_SCHEMA,
    DECIDE_SCHEMA,
    EMBED_SCHEMA,
    EXTRACT_FIELDS_SCHEMA,
    RERANK_SCHEMA,
    handle_config_tool,
    handle_decide,
    handle_embed,
    handle_extract_fields,
    handle_rerank,
)


def register(ctx) -> None:
    """Register all 5 native Hermes capability providers, 5 specialized tools, CLI, and slash commands."""
    # 1. Native Hermes capability providers (selectable in `hermes tools` and config.yaml)
    ctx.register_web_search_provider(OpenRouterEverythingWebProvider())
    ctx.register_image_gen_provider(OpenRouterEverythingImageProvider())
    ctx.register_video_gen_provider(OpenRouterEverythingVideoProvider())
    ctx.register_tts_provider(OpenRouterEverythingTTSProvider())
    ctx.register_transcription_provider(OpenRouterEverythingSTTProvider())

    # 2. Specialized OpenRouter System-1 & Data tools + model switcher/validator
    ctx.register_tool(
        name="openrouter_decide",
        toolset="openrouter_everything",
        schema=DECIDE_SCHEMA,
        handler=handle_decide,
        check_fn=has_credentials,
        requires_env=["OPENROUTER_API_KEY"],
        emoji="🧭",
    )
    ctx.register_tool(
        name="openrouter_extract_fields",
        toolset="openrouter_everything",
        schema=EXTRACT_FIELDS_SCHEMA,
        handler=handle_extract_fields,
        check_fn=has_credentials,
        requires_env=["OPENROUTER_API_KEY"],
        emoji="🧬",
    )
    ctx.register_tool(
        name="openrouter_rerank",
        toolset="openrouter_everything",
        schema=RERANK_SCHEMA,
        handler=handle_rerank,
        check_fn=has_credentials,
        requires_env=["OPENROUTER_API_KEY"],
        emoji="📊",
    )
    ctx.register_tool(
        name="openrouter_embed",
        toolset="openrouter_everything",
        schema=EMBED_SCHEMA,
        handler=handle_embed,
        check_fn=has_credentials,
        requires_env=["OPENROUTER_API_KEY"],
        emoji="🔢",
    )
    ctx.register_tool(
        name="openrouter_everything_config",
        toolset="openrouter_everything",
        schema=CONFIG_TOOL_SCHEMA,
        handler=handle_config_tool,
        check_fn=lambda: True,
        requires_env=[],
        emoji="🎛️",
    )

    # 3. CLI (`hermes openrouter-everything ...`) & TUI slash (`/openrouter-everything ...`) commands
    ctx.register_cli_command(
        name="openrouter-everything",
        help="Unified OpenRouter Everything setup, status, model switcher, and adequacy validator",
        setup_fn=setup_cli_parser,
        handler_fn=handle_cli,
        description="Manage OpenRouter Everything capabilities, models, and adequacy validation under one API key.",
    )
    ctx.register_command(
        name="openrouter-everything",
        handler=handle_slash,
        description="View status, switch models per action, or validate model adequacy (/openrouter-everything status|set|validate|models|activate-all)",
        args_hint="[status|activate-all|set <cap> <model>|validate <cap> <model>|models <cap>]",
    )
