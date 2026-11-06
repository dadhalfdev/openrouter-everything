"""Specialized OpenRouter System-1 & Data tools (Decisions, Extractions, Rerank, Embeddings, Model Switcher/Validator)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from .catalog import (
    CAPABILITY_SPECS, DEFAULTS, PROVIDER_ID,
    build_headers, describe_model_capabilities, fetch_full_catalog,
    get_configured_model, has_credentials, list_models_for_capability,
    load_plugin_config, resolve_credentials, save_plugin_config_key,
    validate_model_for_capability,
)

logger = logging.getLogger(__name__)


def _alpha_base_url(base_url: str) -> str:
    """Derive the alpha API root only from the canonical configured v1 layout."""
    try:
        parsed = urlsplit(base_url)
        # Accessing port also validates malformed values such as ':not-a-port'.
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid OpenRouter API base URL.") from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/api/v1"
    ):
        raise ValueError(
            "OpenRouter alpha APIs require OPENROUTER_BASE_URL to be an HTTP(S) "
            "URL ending in /api/v1, without credentials, query parameters, or fragments."
        )
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/alpha", "", ""))


# ---------------------------------------------------------------------------
# 1. openrouter_decide (typesafe/jev-1.14 via POST /api/alpha/decisions)
# ---------------------------------------------------------------------------
DECIDE_SCHEMA = {
    "name": "openrouter_decide",
    "description": (
        "Run fast System-1 probabilistic decision routing, classification, scoring, or triage using "
        f"OpenRouter's Decisions API (default model: '{DEFAULTS['decision']}'). Returns calibrated "
        "probabilities and confidence scores across your candidate choices or numeric criteria in a "
        "single forward pass (~15ms, ~$0.00001). Validates that the chosen model supports /api/alpha/decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The input context, user query, or document state to evaluate.",
            },
            "options": {
                "type": "object",
                "description": (
                    "Map of option_name -> description/criteria for a single choice question "
                    "(e.g. {'web_search': 'Needs live internet facts', 'code': 'Programming task'}). "
                    "Use either 'options' (simple) or 'questions' (multi-question)."
                ),
            },
            "instructions": {
                "type": "string",
                "description": "Optional instructions for the choice question when using 'options'.",
            },
            "questions": {
                "type": "object",
                "description": (
                    "Advanced multi-question schema mapping question_name -> "
                    "{type: 'choice'|'score'|'boolean', instructions: '...', criteria: {...}}."
                ),
            },
            "model": {
                "type": "string",
                "description": f"Optional model override (default: '{DEFAULTS['decision']}'). Validated for adequacy.",
            },
        },
        "required": ["state"],
    },
}


def handle_decide(args: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    state = (args.get("state") or "").strip()
    if not state:
        return json.dumps({"success": False, "error": "'state' is required for openrouter_decide."})

    api_key, base_url = resolve_credentials()
    if not api_key:
        return json.dumps({"success": False, "error": "OPENROUTER_API_KEY is not set."})
    try:
        alpha_base_url = _alpha_base_url(base_url)
    except ValueError as exc:
        return json.dumps({
            "success": False,
            "error": str(exc),
            "error_type": "unsupported_base_url",
        })

    model_id = get_configured_model("decision", explicit=args.get("model"))
    valid, err_msg, _ = validate_model_for_capability(model_id, "decision")
    if not valid:
        return json.dumps({
            "success": False,
            "error": err_msg,
            "error_type": "inadequate_model",
            "model": model_id,
        }, indent=2)

    questions = args.get("questions")
    if not isinstance(questions, dict) or not questions:
        options = args.get("options")
        if not isinstance(options, dict) or not options:
            return json.dumps({
                "success": False,
                "error": "Provide either 'options' (dict of choice -> description) or 'questions' (full Decisions schema).",
            })
        questions = {
            "decision": {
                "type": "choice",
                "instructions": args.get("instructions") or "Select the best option for the given state.",
                "criteria": {str(k): str(v) for k, v in options.items()},
            }
        }

    import httpx

    payload = {
        "model": model_id,
        "state": state,
        "questions": questions,
    }
    try:
        resp = httpx.post(
            f"{alpha_base_url}/decisions",
            headers=build_headers(api_key),
            json=payload,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            return json.dumps({
                "success": False,
                "error": f"OpenRouter Decisions API error (HTTP {resp.status_code}): {resp.text[:400]}",
                "model": model_id,
            }, indent=2)
        data = resp.json()
        return json.dumps({
            "success": True,
            "model": data.get("model") or model_id,
            "answers": data.get("answers"),
            "usage": data.get("usage"),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Decisions request failed: {exc}", "model": model_id})


# ---------------------------------------------------------------------------
# 2. openrouter_extract_fields (cloudflare/clairic via POST /api/alpha/extractions)
# ---------------------------------------------------------------------------
EXTRACT_FIELDS_SCHEMA = {
    "name": "openrouter_extract_fields",
    "description": (
        "Extract typed structured fields (text, number, boolean, category, list) with character spans "
        "and calibrated confidence scores from raw text or documents using OpenRouter's System-1 "
        f"Extractions API (default model: '{DEFAULTS['extraction']}')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The source text or document content to extract fields from.",
            },
            "fields": {
                "type": "object",
                "description": (
                    "Map of field_name -> field spec, e.g. "
                    "{'invoice_number': {'type': 'text'}, 'total': {'type': 'number', 'decimals': 2, 'unit': 'USD'}, "
                    "'contacts': {'type': 'list', 'items': 'text'}}."
                ),
            },
            "model": {
                "type": "string",
                "description": f"Optional model override (default: '{DEFAULTS['extraction']}'). Validated for adequacy.",
            },
        },
        "required": ["state", "fields"],
    },
}


def handle_extract_fields(args: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    state = (args.get("state") or "").strip()
    fields = args.get("fields")
    if not state or not isinstance(fields, dict) or not fields:
        return json.dumps({"success": False, "error": "Both 'state' and non-empty 'fields' dict are required."})

    api_key, base_url = resolve_credentials()
    if not api_key:
        return json.dumps({"success": False, "error": "OPENROUTER_API_KEY is not set."})
    try:
        alpha_base_url = _alpha_base_url(base_url)
    except ValueError as exc:
        return json.dumps({
            "success": False,
            "error": str(exc),
            "error_type": "unsupported_base_url",
        })

    model_id = get_configured_model("extraction", explicit=args.get("model"))
    valid, err_msg, _ = validate_model_for_capability(model_id, "extraction")
    if not valid:
        return json.dumps({
            "success": False,
            "error": err_msg,
            "error_type": "inadequate_model",
            "model": model_id,
        }, indent=2)

    import httpx

    payload = {"model": model_id, "state": state, "fields": fields}
    try:
        resp = httpx.post(
            f"{alpha_base_url}/extractions",
            headers=build_headers(api_key),
            json=payload,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            return json.dumps({
                "success": False,
                "error": f"OpenRouter Extractions API error (HTTP {resp.status_code}): {resp.text[:400]}",
                "model": model_id,
            }, indent=2)
        data = resp.json()
        return json.dumps({
            "success": True,
            "model": data.get("model") or model_id,
            "answers": data.get("answers"),
            "usage": data.get("usage"),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Extractions request failed: {exc}", "model": model_id})


# ---------------------------------------------------------------------------
# 3. openrouter_rerank (cohere/rerank-v3.5 via POST /api/v1/rerank)
# ---------------------------------------------------------------------------
RERANK_SCHEMA = {
    "name": "openrouter_rerank",
    "description": (
        "Rerank a list of candidate documents, search results, or passages by semantic relevance "
        f"to a query using OpenRouter's Rerank API (default model: '{DEFAULTS['rerank']}')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question to rank documents against.",
            },
            "documents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document strings to rank.",
            },
            "top_n": {
                "type": "integer",
                "description": "Optional number of top-ranked documents to return.",
            },
            "model": {
                "type": "string",
                "description": f"Optional model override (default: '{DEFAULTS['rerank']}'). Validated for adequacy.",
            },
        },
        "required": ["query", "documents"],
    },
}


def handle_rerank(args: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    query = (args.get("query") or "").strip()
    documents = args.get("documents")
    if not query or not isinstance(documents, list) or not documents:
        return json.dumps({"success": False, "error": "'query' and non-empty 'documents' list are required."})

    api_key, base_url = resolve_credentials()
    if not api_key:
        return json.dumps({"success": False, "error": "OPENROUTER_API_KEY is not set."})

    model_id = get_configured_model("rerank", explicit=args.get("model"))
    valid, err_msg, _ = validate_model_for_capability(model_id, "rerank")
    if not valid:
        return json.dumps({
            "success": False,
            "error": err_msg,
            "error_type": "inadequate_model",
            "model": model_id,
        }, indent=2)

    import httpx

    payload: dict[str, Any] = {
        "model": model_id,
        "query": query,
        "documents": [str(d) for d in documents],
    }
    if args.get("top_n"):
        payload["top_n"] = int(args["top_n"])

    try:
        resp = httpx.post(
            f"{base_url}/rerank",
            headers=build_headers(api_key),
            json=payload,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            return json.dumps({
                "success": False,
                "error": f"OpenRouter Rerank API error (HTTP {resp.status_code}): {resp.text[:400]}",
                "model": model_id,
            }, indent=2)
        data = resp.json()
        return json.dumps({
            "success": True,
            "model": data.get("model") or model_id,
            "results": data.get("results"),
            "usage": data.get("usage"),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Rerank request failed: {exc}", "model": model_id})


# ---------------------------------------------------------------------------
# 4. openrouter_embed (google/gemini-embedding-2 via POST /api/v1/embeddings)
# ---------------------------------------------------------------------------
EMBED_SCHEMA = {
    "name": "openrouter_embed",
    "description": (
        "Generate dense vector embeddings for text strings using OpenRouter's Embeddings API "
        f"(default model: '{DEFAULTS['embeddings']}'). Returns embedding dimensions, vector preview, "
        "and cosine similarity matrix when multiple texts are supplied."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of text strings to embed.",
            },
            "include_vectors": {
                "type": "boolean",
                "description": "Whether to return the full float vectors (default false: returns dim + similarity matrix + preview).",
                "default": False,
            },
            "model": {
                "type": "string",
                "description": f"Optional model override (default: '{DEFAULTS['embeddings']}'). Validated for adequacy.",
            },
        },
        "required": ["input"],
    },
}


def _cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return round(dot / (na * nb), 6) if na and nb else 0.0


def handle_embed(args: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    inputs = args.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs:
        return json.dumps({"success": False, "error": "'input' (list of strings) is required."})

    api_key, base_url = resolve_credentials()
    if not api_key:
        return json.dumps({"success": False, "error": "OPENROUTER_API_KEY is not set."})

    model_id = get_configured_model("embeddings", explicit=args.get("model"))
    valid, err_msg, _ = validate_model_for_capability(model_id, "embeddings")
    if not valid:
        return json.dumps({
            "success": False,
            "error": err_msg,
            "error_type": "inadequate_model",
            "model": model_id,
        }, indent=2)

    import httpx

    try:
        resp = httpx.post(
            f"{base_url}/embeddings",
            headers=build_headers(api_key),
            json={"model": model_id, "input": [str(x) for x in inputs]},
            timeout=30.0,
        )
        if resp.status_code >= 400:
            return json.dumps({
                "success": False,
                "error": f"OpenRouter Embeddings API error (HTTP {resp.status_code}): {resp.text[:400]}",
                "model": model_id,
            }, indent=2)
        body = resp.json()
        items = body.get("data") or []
        vectors = [item.get("embedding") or [] for item in items]
        dim = len(vectors[0]) if vectors else 0
        out: dict[str, Any] = {
            "success": True,
            "model": body.get("model") or model_id,
            "count": len(vectors),
            "dimensions": dim,
            "usage": body.get("usage"),
        }
        if len(vectors) > 1 and dim > 0:
            out["similarity_matrix"] = [
                [_cosine_sim(v1, v2) for v2 in vectors]
                for v1 in vectors
            ]
        if args.get("include_vectors"):
            out["embeddings"] = vectors
        else:
            out["vector_preview"] = [v[:5] for v in vectors]
        return json.dumps(out, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Embeddings request failed: {exc}", "model": model_id})


# ---------------------------------------------------------------------------
# 5. openrouter_everything_config (Inspect, Validate, Switch Models, Activate All)
# ---------------------------------------------------------------------------
CONFIG_TOOL_SCHEMA = {
    "name": "openrouter_everything_config",
    "description": (
        "Inspect, validate, switch models per action, or activate OpenRouter Everything across all Hermes "
        "capabilities (text, web_search, web_fetch, image_gen, video_gen, tts, stt, decision, extraction, "
        "rerank, embeddings). Automatically validates model adequacy and rejects mismatched models (e.g. "
        "trying to use an image model for video generation) with clear explanations and recommendations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "set_model", "validate", "list_models", "activate_all", "reset_defaults"],
                "description": (
                    "Action to perform: 'status' (show current models & active backends), "
                    "'set_model' (switch model for a capability after adequacy validation), "
                    "'validate' (check if a model is adequate for a capability), "
                    "'list_models' (list compatible OpenRouter models for a capability), "
                    "'activate_all' (set openrouter-everything as the active provider for web, image_gen, video_gen, tts, stt), "
                    "'reset_defaults' (reset all capability models to curated defaults)."
                ),
            },
            "capability": {
                "type": "string",
                "enum": list(CAPABILITY_SPECS.keys()),
                "description": "Target capability (e.g. 'web_search', 'image_gen', 'video_gen', 'tts', 'stt', 'decision', 'extraction', 'rerank', 'embeddings', 'text').",
            },
            "model": {
                "type": "string",
                "description": "OpenRouter model slug to validate or set (e.g. 'alibaba/wan-3.0-prime').",
            },
        },
        "required": ["action"],
    },
}


def activate_all_providers_in_config() -> dict[str, Any]:
    """Point Hermes' web, image_gen, video_gen, tts, and stt providers to openrouter-everything
    and ensure all curated default models are written to config.yaml."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    or_sec = cfg.setdefault("openrouter_everything", {})
    if not isinstance(or_sec, dict):
        or_sec = {}
        cfg["openrouter_everything"] = or_sec
    for cap, default_mod in DEFAULTS.items():
        or_sec.setdefault(cap, default_mod)

    # 1. Web Search & Extract
    web_cfg = cfg.setdefault("web", {})
    if isinstance(web_cfg, dict):
        web_cfg["backend"] = PROVIDER_ID
        web_cfg.pop("search_backend", None)
        web_cfg.pop("extract_backend", None)
        web_cfg.pop("use_gateway", None)

    # 2. Image Generation
    img_cfg = cfg.setdefault("image_gen", {})
    if isinstance(img_cfg, dict):
        img_cfg["provider"] = PROVIDER_ID
        img_cfg["model"] = or_sec.get("image_gen", DEFAULTS["image_gen"])
        img_cfg.pop("use_gateway", None)

    # 3. Video Generation
    vid_cfg = cfg.setdefault("video_gen", {})
    if isinstance(vid_cfg, dict):
        vid_cfg["provider"] = PROVIDER_ID
        vid_cfg["model"] = or_sec.get("video_gen", DEFAULTS["video_gen"])
        vid_cfg.pop("use_gateway", None)

    # 4. Text-to-Speech (TTS)
    tts_cfg = cfg.setdefault("tts", {})
    if isinstance(tts_cfg, dict):
        tts_cfg["provider"] = PROVIDER_ID
        tts_cfg["model"] = or_sec.get("tts", DEFAULTS["tts"])
        or_tts = tts_cfg.setdefault(PROVIDER_ID, {})
        if isinstance(or_tts, dict):
            or_tts["model"] = or_sec.get("tts", DEFAULTS["tts"])

    # 5. Speech-to-Text (STT)
    stt_cfg = cfg.setdefault("stt", {})
    if isinstance(stt_cfg, dict):
        stt_cfg["enabled"] = True
        stt_cfg["provider"] = PROVIDER_ID
        or_stt = stt_cfg.setdefault(PROVIDER_ID, {})
        if isinstance(or_stt, dict):
            or_stt["model"] = or_sec.get("stt", DEFAULTS["stt"])

    save_config(cfg)
    return {
        "web.backend": PROVIDER_ID,
        "image_gen.provider": PROVIDER_ID,
        "video_gen.provider": PROVIDER_ID,
        "tts.provider": PROVIDER_ID,
        "stt.provider": PROVIDER_ID,
        "models": {k: get_configured_model(k) for k in CAPABILITY_SPECS},
    }


def handle_config_tool(args: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    action = (args.get("action") or "status").strip().lower()
    capability = (args.get("capability") or "").strip().lower()
    model_id = (args.get("model") or "").strip()

    if action == "status":
        from hermes_cli.config import load_config
        cfg = load_config()
        status_rows = {}
        for cap, spec in CAPABILITY_SPECS.items():
            cur = get_configured_model(cap)
            ok, err, _ = validate_model_for_capability(cur, cap)
            status_rows[cap] = {
                "label": spec["label"],
                "active_model": cur,
                "default_model": DEFAULTS.get(cap),
                "adequate": ok,
                **({"validation_warning": err} if not ok else {}),
            }
        return json.dumps({
            "success": True,
            "openrouter_api_key_configured": has_credentials(),
            "active_hermes_providers": {
                "web.backend": (cfg.get("web") or {}).get("backend"),
                "image_gen.provider": (cfg.get("image_gen") or {}).get("provider"),
                "video_gen.provider": (cfg.get("video_gen") or {}).get("provider"),
                "tts.provider": (cfg.get("tts") or {}).get("provider"),
                "stt.provider": (cfg.get("stt") or {}).get("provider"),
            },
            "capabilities": status_rows,
        }, indent=2)

    if action == "validate":
        if not capability or capability not in CAPABILITY_SPECS:
            return json.dumps({
                "success": False,
                "error": f"Specify a valid 'capability' from: {list(CAPABILITY_SPECS.keys())}",
            })
        target_model = model_id or get_configured_model(capability)
        ok, err, entry = validate_model_for_capability(target_model, capability)
        return json.dumps({
            "success": True,
            "capability": capability,
            "capability_label": CAPABILITY_SPECS[capability]["label"],
            "model": target_model,
            "adequate": ok,
            "message": f"Model '{target_model}' is valid and adequate for {CAPABILITY_SPECS[capability]['label']}." if ok else err,
            "model_supported_capabilities": describe_model_capabilities(entry) if entry else [],
        }, indent=2)

    if action == "set_model":
        if not capability or capability not in CAPABILITY_SPECS:
            return json.dumps({
                "success": False,
                "error": f"Specify a valid 'capability' from: {list(CAPABILITY_SPECS.keys())}",
            })
        if not model_id:
            return json.dumps({"success": False, "error": "Specify 'model' to set."})

        ok, err, entry = validate_model_for_capability(model_id, capability)
        if not ok:
            return json.dumps({
                "success": False,
                "error": err,
                "error_type": "inadequate_model",
                "capability": capability,
                "rejected_model": model_id,
                "model_supported_capabilities": describe_model_capabilities(entry) if entry else [],
                "recommended_models": CAPABILITY_SPECS[capability].get("recommended", []),
            }, indent=2)

        save_plugin_config_key(capability, model_id)
        return json.dumps({
            "success": True,
            "capability": capability,
            "capability_label": CAPABILITY_SPECS[capability]["label"],
            "previous_default": DEFAULTS.get(capability),
            "new_model": model_id,
            "message": f"Updated {CAPABILITY_SPECS[capability]['label']} model to '{model_id}'.",
        }, indent=2)

    if action == "list_models":
        if not capability or capability not in CAPABILITY_SPECS:
            return json.dumps({
                "success": False,
                "error": f"Specify a valid 'capability' from: {list(CAPABILITY_SPECS.keys())}",
            })
        models = list_models_for_capability(capability)
        return json.dumps({
            "success": True,
            "capability": capability,
            "capability_label": CAPABILITY_SPECS[capability]["label"],
            "default_model": DEFAULTS.get(capability),
            "configured_model": get_configured_model(capability),
            "total_compatible_models": len(models),
            "models": [{k: v for k, v in m.items() if k != "_rank"} for m in models[:50]],
        }, indent=2)

    if action == "activate_all":
        activated = activate_all_providers_in_config()
        return json.dumps({
            "success": True,
            "message": "Activated 'openrouter-everything' as the provider for Web, Image Gen, Video Gen, TTS, and STT.",
            "activated": activated,
        }, indent=2)

    if action == "reset_defaults":
        for cap, mod in DEFAULTS.items():
            save_plugin_config_key(cap, mod)
        return json.dumps({
            "success": True,
            "message": "Reset all OpenRouter Everything capability models to curated defaults.",
            "defaults": DEFAULTS,
        }, indent=2)

    return json.dumps({"success": False, "error": f"Unknown action: {action}"})
