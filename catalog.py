"""Configuration, credentials, live OpenRouter catalog, and model-task adequacy validator."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROVIDER_ID = "openrouter-everything"
DISPLAY_NAME = "OpenRouter Everything"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ALPHA_BASE_URL = "https://openrouter.ai/api/alpha"

# Curated defaults per capability (Marco's specifications + best-in-class defaults)
DEFAULTS: dict[str, str] = {
    "text": "deepseek/deepseek-v4.1-flash-1017",
    "web_search": "deepseek/deepseek-v4.1-flash-1017",  # Drives openrouter:web_search server tool by default
    "web_search_engine": "auto",  # "auto" | "parallel" | "exa" | "perplexity" | "firecrawl" | "native"
    "web_fetch": "deepseek/deepseek-v4.1-flash-1017",  # Drives openrouter:web_fetch server tool by default
    "web_fetch_engine": "openrouter",  # "openrouter" (free) | "exa" | "parallel" | "firecrawl" | "auto" | "native"
    "image_gen": "google/gemini-nano-banana-2.1",
    "video_gen": "alibaba/wan-3.0-prime",
    "tts": "fish-audio/drama-3",
    "tts_voice": "",  # Optional voice id (drama-3 uses default expressive voice or inline [emotion] tags)
    "stt": "openai/whisper-large-v3-turbo",
    "decision": "typesafe/jev-1.14",
    "extraction": "cloudflare/clairic",
    "rerank": "cohere/rerank-v3.5",
    "embeddings": "google/gemini-embedding-2",
}

# Human-readable capability descriptions and required OpenRouter metadata
CAPABILITY_SPECS: dict[str, dict[str, Any]] = {
    "text": {
        "label": "Text / Chat / Fast Aux",
        "required_endpoints": ("/api/v1/chat/completions",),
        "required_output_modalities": ("text",),
        "recommended": [
            "deepseek/deepseek-v4.1-flash-1017",
            "google/gemini-3-flash-preview",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4-mini",
        ],
    },
    "web_search": {
        "label": "Web Search (openrouter:web_search Server Tool)",
        "required_endpoints": ("/api/v1/chat/completions",),
        "required_output_modalities": ("text",),
        "disallow_pure_modalities": ("image", "video", "speech", "embeddings"),
        "recommended": [
            "deepseek/deepseek-v4.1-flash-1017",
            "google/gemini-3-flash-preview",
            "perplexity/sonar",
            "perplexity/sonar-pro",
            "openai/gpt-5.4-mini",
        ],
    },
    "web_fetch": {
        "label": "Web Extract / Fetch (openrouter:web_fetch Server Tool)",
        "required_endpoints": ("/api/v1/chat/completions",),
        "required_output_modalities": ("text",),
        "recommended": [
            "deepseek/deepseek-v4.1-flash-1017",
            "google/gemini-3-flash-preview",
            "openai/gpt-5.4-mini",
        ],
    },
    "image_gen": {
        "label": "Image Generation",
        "required_endpoints": ("/api/v1/images", "/api/v1/images/generations"),
        "required_output_modalities": ("image",),
        "recommended": [
            "google/gemini-nano-banana-2.1",
            "google/gemini-3.1-flash-image",
            "openai/gpt-image-2",
            "openai/gpt-5.4-image-2",
            "bytedance-seed/seedream-5.0-lite",
            "black-forest-labs/flux.2-max",
            "qwen/qwen-image-3-pro",
        ],
    },
    "video_gen": {
        "label": "Video Generation",
        "required_endpoints": ("/api/v1/videos",),
        "required_output_modalities": ("video",),
        "recommended": [
            "alibaba/wan-3.0-prime",
            "alibaba/wan-3.0",
            "minimax/hailuo-3-max",
            "minimax/hailuo-3",
            "google/veo-3.1-lite",
            "klingai/kling-v3-pro",
            "bytedance/seedance-2.0-fast",
            "openai/sora-2",
        ],
    },
    "tts": {
        "label": "Text-to-Speech (TTS)",
        "required_endpoints": ("/api/v1/audio/speech",),
        "required_output_modalities": ("speech",),
        "recommended": [
            "fish-audio/drama-3",
            "fish-audio/s2-pro",
            "openai/gpt-4o-mini-tts",
            "elevenlabs/eleven-v3",
            "elevenlabs/eleven-flash-v2.5",
            "minimax/speech-2.8-hd",
            "cartesia/sonic-3",
            "mistralai/voxtral-mini-tts-2603",
        ],
    },
    "stt": {
        "label": "Speech-to-Text (STT)",
        "required_endpoints": ("/api/v1/audio/transcriptions",),
        "required_input_modalities": ("audio",),
        "recommended": [
            "openai/whisper-large-v3-turbo",
            "mistralai/voxtral-mini-transcribe-3-0",
            "moondream/parakeet-ultra",
            "openai/gpt-4o-mini-transcribe",
            "elevenlabs/scribe-v2",
            "deepgram/nova-3",
            "assemblyai/universal-3-pro",
        ],
    },
    "decision": {
        "label": "Decision / Probabilistic Routing",
        "required_endpoints": ("/api/alpha/decisions",),
        "recommended": [
            "typesafe/jev-1.14",
            "typesafe/jev-1.13",
            "scaledown/scale-decide-v1",
        ],
    },
    "extraction": {
        "label": "Structured Field Extraction",
        "required_endpoints": ("/api/alpha/extractions",),
        "recommended": [
            "cloudflare/clairic",
            "scaledown/scale-extract-v1",
        ],
    },
    "rerank": {
        "label": "Document Reranking",
        "required_endpoints": ("/api/v1/rerank",),
        "recommended": [
            "cohere/rerank-v3.5",
            "cohere/rerank-5-fast",
            "cohere/rerank-5-pro",
            "jina/jina-reranker-v3",
            "zeroentropy/zerank-2",
        ],
    },
    "embeddings": {
        "label": "Vector Embeddings",
        "required_endpoints": ("/api/v1/embeddings",),
        "required_output_modalities": ("embeddings",),
        "recommended": [
            "google/gemini-embedding-2",
            "openai/text-embedding-3-small",
            "qwen/qwen3-embedding-8b",
            "baai/bge-m3",
        ],
    },
}

_CATALOG_TTL_S = 600.0
_CATALOG_CACHE: Optional[Tuple[float, Dict[str, Dict[str, Any]]]] = None
_VIDEO_CATALOG_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_IMAGE_CATALOG_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None


def resolve_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for OpenRouter using Hermes' profile-scoped secret routes (Rule 11)."""
    from agent.secret_scope import get_secret

    api_key = str(get_secret("OPENROUTER_API_KEY") or "").strip()
    base_url = str(get_secret("OPENROUTER_BASE_URL") or "").strip().rstrip("/") or DEFAULT_BASE_URL
    return api_key, base_url


def has_credentials() -> bool:
    api_key, _ = resolve_credentials()
    return bool(api_key)


def build_headers(api_key: Optional[str] = None) -> dict[str, str]:
    if api_key is None:
        api_key, _ = resolve_credentials()
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent (OpenRouter Everything)",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def load_plugin_config() -> dict[str, Any]:
    """Read ``openrouter_everything`` block from config.yaml alongside category overrides."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        return {}
    section = cfg.get("openrouter_everything")
    return dict(section) if isinstance(section, dict) else {}


def save_plugin_config_key(key: str, value: Any) -> None:
    """Persist ``openrouter_everything.<key> = value`` into config.yaml (and sync category key if active)."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    section = cfg.setdefault("openrouter_everything", {})
    if not isinstance(section, dict):
        section = {}
        cfg["openrouter_everything"] = section
    section[key] = value

    # Also sync to native category model keys if openrouter-everything is the active provider there
    if key == "image_gen":
        img_cfg = cfg.setdefault("image_gen", {})
        if isinstance(img_cfg, dict) and img_cfg.get("provider") in (PROVIDER_ID, "openrouter", None):
            img_cfg["model"] = value
    elif key == "video_gen":
        vid_cfg = cfg.setdefault("video_gen", {})
        if isinstance(vid_cfg, dict) and vid_cfg.get("provider") in (PROVIDER_ID, "openrouter", None):
            vid_cfg["model"] = value
    elif key == "tts":
        tts_cfg = cfg.get("tts")
        if isinstance(tts_cfg, dict) and tts_cfg.get("provider") == PROVIDER_ID:
            tts_cfg["model"] = value
            or_tts = tts_cfg.setdefault(PROVIDER_ID, {})
            if isinstance(or_tts, dict):
                or_tts["model"] = value
    elif key == "stt":
        stt_cfg = cfg.setdefault("stt", {})
        if isinstance(stt_cfg, dict):
            or_stt = stt_cfg.setdefault(PROVIDER_ID, {})
            if isinstance(or_stt, dict):
                or_stt["model"] = value

    save_config(cfg)


def get_configured_model(capability: str, explicit: Optional[str] = None) -> str:
    """Resolve the active model for *capability*:
    1. Explicit per-call parameter (if valid and not another provider's default like 'whisper-1')
    2. Environment variable ``OPENROUTER_<CAPABILITY>_MODEL``
    3. ``openrouter_everything.<capability>`` in config.yaml
    4. Native Hermes section override (e.g. ``image_gen.model``, ``video_gen.model``, ``tts.model``)
    5. ``DEFAULTS[capability]``
    """
    # Guard against Hermes built-in default models leaking in when user hasn't set a model
    _FOREIGN_DEFAULTS = frozenset({
        "whisper-1", "base", "gpt-4o-mini-tts", "edge", "fal-ai/flux-2/klein/4b",
    })
    if isinstance(explicit, str) and explicit.strip() and explicit.strip() not in _FOREIGN_DEFAULTS:
        return explicit.strip()

    from agent.secret_scope import get_secret
    env_key = f"OPENROUTER_{capability.upper()}_MODEL"
    env_val = str(get_secret(env_key) or "").strip()
    if env_val:
        return env_val

    plugin_cfg = load_plugin_config()
    cfg_val = plugin_cfg.get(capability)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()

    # Check native Hermes category sections
    try:
        from hermes_cli.config import load_config
        full_cfg = load_config()
        if capability == "image_gen":
            v = (full_cfg.get("image_gen") or {}).get("model")
            if isinstance(v, str) and v.strip() and "/" in v and not v.startswith("fal-ai/"):
                return v.strip()
        elif capability == "video_gen":
            v = (full_cfg.get("video_gen") or {}).get("model")
            if isinstance(v, str) and v.strip() and "/" in v and not v.startswith("fal-ai/"):
                return v.strip()
        elif capability == "tts":
            tts_c = full_cfg.get("tts") or {}
            v = (tts_c.get(PROVIDER_ID) or {}).get("model") or tts_c.get("model")
            if isinstance(v, str) and v.strip() and "/" in v:
                return v.strip()
        elif capability == "stt":
            stt_c = full_cfg.get("stt") or {}
            v = (stt_c.get(PROVIDER_ID) or {}).get("model") or stt_c.get("model")
            if isinstance(v, str) and v.strip() and "/" in v:
                return v.strip()
    except Exception:
        pass

    return DEFAULTS.get(capability, "")


def fetch_full_catalog(force: bool = False) -> dict[str, dict[str, Any]]:
    """Fetch ``GET /api/v1/models?output_modalities=all`` from OpenRouter (~518 models across all modalities).
    Cached for 10 minutes."""
    global _CATALOG_CACHE
    now = time.monotonic()
    if not force and _CATALOG_CACHE is not None and (now - _CATALOG_CACHE[0]) < _CATALOG_TTL_S:
        return _CATALOG_CACHE[1]

    import requests

    _, base_url = resolve_credentials()
    url = f"{base_url}/models?output_modalities=all"
    try:
        resp = requests.get(url, headers=build_headers(), timeout=15.0)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        catalog = {
            entry["id"].strip(): entry
            for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"].strip()
        }
        if catalog:
            _CATALOG_CACHE = (now, catalog)
            return catalog
    except Exception as exc:
        logger.debug("OpenRouter Everything catalog fetch failed: %s", exc)

    return _CATALOG_CACHE[1] if _CATALOG_CACHE is not None else {}


def fetch_video_catalog(force: bool = False) -> list[dict[str, Any]]:
    """Fetch ``GET /api/v1/videos/models`` with per-model durations, resolutions, aspect ratios."""
    global _VIDEO_CATALOG_CACHE
    now = time.monotonic()
    if not force and _VIDEO_CATALOG_CACHE is not None and (now - _VIDEO_CATALOG_CACHE[0]) < _CATALOG_TTL_S:
        return _VIDEO_CATALOG_CACHE[1]

    import requests

    _, base_url = resolve_credentials()
    try:
        resp = requests.get(f"{base_url}/videos/models", headers=build_headers(), timeout=12.0)
        resp.raise_for_status()
        data = [e for e in (resp.json().get("data") or []) if isinstance(e, dict) and e.get("id")]
        if data:
            _VIDEO_CATALOG_CACHE = (now, data)
            return data
    except Exception as exc:
        logger.debug("OpenRouter video catalog fetch failed: %s", exc)
    return _VIDEO_CATALOG_CACHE[1] if _VIDEO_CATALOG_CACHE is not None else []


def fetch_image_catalog(force: bool = False) -> list[dict[str, Any]]:
    """Fetch ``GET /api/v1/images/models`` with per-model supported_parameters."""
    global _IMAGE_CATALOG_CACHE
    now = time.monotonic()
    if not force and _IMAGE_CATALOG_CACHE is not None and (now - _IMAGE_CATALOG_CACHE[0]) < _CATALOG_TTL_S:
        return _IMAGE_CATALOG_CACHE[1]

    import requests

    _, base_url = resolve_credentials()
    try:
        resp = requests.get(f"{base_url}/images/models", headers=build_headers(), timeout=12.0)
        resp.raise_for_status()
        data = [e for e in (resp.json().get("data") or []) if isinstance(e, dict) and e.get("id")]
        if data:
            _IMAGE_CATALOG_CACHE = (now, data)
            return data
    except Exception as exc:
        logger.debug("OpenRouter image catalog fetch failed: %s", exc)
    return _IMAGE_CATALOG_CACHE[1] if _IMAGE_CATALOG_CACHE is not None else []


def describe_model_capabilities(entry: dict[str, Any]) -> list[str]:
    """Return the capability keys that a catalog model entry actually supports."""
    endpoints = set(entry.get("endpoints") or [])
    arch = entry.get("architecture") or {}
    out_mods = set(arch.get("output_modalities") or [])
    in_mods = set(arch.get("input_modalities") or [])

    supported: list[str] = []
    if "/api/v1/videos" in endpoints or "video" in out_mods:
        supported.append("video_gen (Video Generation)")
    if "/api/v1/images" in endpoints or "/api/v1/images/generations" in endpoints or "image" in out_mods:
        supported.append("image_gen (Image Generation)")
    if "/api/v1/audio/speech" in endpoints or "speech" in out_mods:
        supported.append("tts (Text-to-Speech)")
    if "/api/v1/audio/transcriptions" in endpoints:
        supported.append("stt (Speech-to-Text)")
    if "/api/alpha/decisions" in endpoints:
        supported.append("decision (Probabilistic Decision Routing)")
    if "/api/alpha/extractions" in endpoints:
        supported.append("extraction (Structured Field Extraction)")
    if "/api/v1/rerank" in endpoints:
        supported.append("rerank (Document Reranking)")
    if "/api/v1/embeddings" in endpoints or "embeddings" in out_mods:
        supported.append("embeddings (Vector Embeddings)")
    if "/api/v1/chat/completions" in endpoints and "text" in out_mods:
        supported.append("text / web_search / web_fetch")
    return supported or [f"endpoints={sorted(endpoints)}, modality={arch.get('modality', 'unknown')}"]


def validate_model_for_capability(
    model_id: str, capability: str, *, catalog: Optional[dict[str, dict[str, Any]]] = None
) -> tuple[bool, Optional[str], Optional[dict[str, Any]]]:
    """Validate whether *model_id* is adequate for *capability*.

    Returns ``(is_valid, error_message_or_None, model_entry_or_None)``.
    When a user configures or passes a model that is not adequate for the task (e.g., an image
    model for video generation, or a chat model for TTS), returns ``(False, detailed_error, entry)``
    explaining what the model actually supports and listing recommended models for the task.
    """
    model_id = (model_id or "").strip()
    spec = CAPABILITY_SPECS.get(capability)
    if not spec:
        return True, None, None
    if not model_id:
        return False, f"No model specified for {spec['label']}. Default is '{DEFAULTS.get(capability)}'.", None

    if catalog is None:
        catalog = fetch_full_catalog()

    # If offline / catalog unreachable, don't block execution — let the API validate
    if not catalog:
        return True, None, None

    entry = catalog.get(model_id)
    if entry is None:
        # Check close matches to help with typos
        lower = model_id.lower()
        suggestions = [mid for mid in catalog if lower in mid.lower() or mid.split("/")[-1] in lower][:5]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        rec = ", ".join(spec.get("recommended", [])[:4])
        return (
            False,
            f"Model '{model_id}' was not found in the OpenRouter catalog for {spec['label']}.{hint} "
            f"Recommended models for {capability}: {rec} (default: '{DEFAULTS.get(capability)}').",
            None,
        )

    endpoints = set(entry.get("endpoints") or [])
    arch = entry.get("architecture") or {}
    out_mods = set(arch.get("output_modalities") or [])
    in_mods = set(arch.get("input_modalities") or [])
    modality_str = arch.get("modality") or f"{'+'.join(in_mods)}->{'+'.join(out_mods)}"

    # Check required endpoints (any match satisfies)
    req_eps = spec.get("required_endpoints") or ()
    ep_ok = not req_eps or any(ep in endpoints for ep in req_eps)

    # Check required output modalities (any match satisfies)
    req_out = spec.get("required_output_modalities") or ()
    out_ok = not req_out or any(m in out_mods for m in req_out)

    # Check required input modalities
    req_in = spec.get("required_input_modalities") or ()
    in_ok = not req_in or any(m in in_mods for m in req_in)

    # Special check for web_search / text: disallow models that ONLY output image/video/speech/embeddings
    if capability in ("web_search", "web_fetch", "text"):
        if "text" not in out_mods and "/api/v1/chat/completions" not in endpoints:
            out_ok = False
        # Even if an image model like gemini-nano-banana-2.1 has /api/v1/chat/completions, warn if user
        # accidentally uses a dedicated image-gen model for web_search
        if "/api/v1/images" in endpoints and capability in ("web_search", "web_fetch"):
            actual_caps = ", ".join(describe_model_capabilities(entry))
            rec = ", ".join(spec.get("recommended", [])[:4])
            return (
                False,
                f"Inadequate model for {spec['label']}: '{model_id}' ({entry.get('name', model_id)}) is an "
                f"Image Generation model (modality: {modality_str}; suited for: {actual_caps}) and should not "
                f"be used for {spec['label']}. Recommended models for {capability}: {rec} "
                f"(default: '{DEFAULTS.get(capability)}').",
                entry,
            )

    if not (ep_ok and out_ok and in_ok):
        actual_caps = ", ".join(describe_model_capabilities(entry))
        rec = ", ".join(spec.get("recommended", [])[:4])
        return (
            False,
            f"Inadequate model for {spec['label']}: '{model_id}' ({entry.get('name', model_id)}) has "
            f"modality '{modality_str}' and endpoints {sorted(endpoints)}, which is suited for "
            f"[{actual_caps}] — NOT {spec['label']}. "
            f"You cannot use this model for {capability}. "
            f"Recommended models for {capability}: {rec} (default: '{DEFAULTS.get(capability)}').",
            entry,
        )

    return True, None, entry


def list_models_for_capability(capability: str) -> list[dict[str, Any]]:
    """Return all valid models in the OpenRouter catalog for *capability*, with recommended ones first."""
    catalog = fetch_full_catalog()
    spec = CAPABILITY_SPECS.get(capability, {})
    recommended = spec.get("recommended") or []
    rec_rank = {m: i for i, m in enumerate(recommended)}

    rows: list[dict[str, Any]] = []
    for mid, entry in catalog.items():
        ok, _, _ = validate_model_for_capability(mid, capability, catalog=catalog)
        if not ok:
            continue
        arch = entry.get("architecture") or {}
        rows.append({
            "id": mid,
            "display": entry.get("name") or mid,
            "modality": arch.get("modality") or "",
            "endpoints": entry.get("endpoints") or [],
            "recommended": mid in rec_rank,
            "_rank": rec_rank.get(mid, 999),
        })
    rows.sort(key=lambda r: (r["_rank"], r["id"]))
    return rows
