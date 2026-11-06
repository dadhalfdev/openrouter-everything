"""All 5 native Hermes capability providers (Web Search/Extract, Image Gen, Video Gen, TTS, STT)
backed by OpenRouter Everything under a single OPENROUTER_API_KEY, with model-adequacy validation."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from hermes_constants import (
    get_default_hermes_root, get_hermes_home, get_real_home, get_subprocess_home,
)

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response as img_error_response,
    resolve_aspect_ratio, save_b64_image, save_url_image, success_response as img_success_response,
)
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import (
    VideoGenProvider, error_response as vid_error_response, save_url_video,
    success_response as vid_success_response,
)
from agent.web_search_provider import WebSearchProvider

from .catalog import (
    DEFAULTS, DISPLAY_NAME, PROVIDER_ID, build_headers, fetch_full_catalog,
    fetch_image_catalog, fetch_video_catalog, get_configured_model,
    has_credentials, load_plugin_config, resolve_credentials,
    validate_model_for_capability,
)

logger = logging.getLogger(__name__)

_SETUP_SCHEMA_ENV = [
    {
        "key": "OPENROUTER_API_KEY",
        "prompt": "OpenRouter API key (shared across all OpenRouter Everything capabilities)",
        "url": "https://openrouter.ai/settings/keys",
    }
]


# ---------------------------------------------------------------------------
# 1. Web Search & Extract Provider
# ---------------------------------------------------------------------------
_VALID_SEARCH_ENGINES = frozenset({"auto", "native", "exa", "parallel", "firecrawl", "perplexity"})
_VALID_FETCH_ENGINES = frozenset({"auto", "native", "openrouter", "exa", "parallel", "firecrawl"})


class OpenRouterEverythingWebProvider(WebSearchProvider):
    """Web Search via ``openrouter:web_search`` server tool + Web Extract via ``openrouter:web_fetch`` server tool."""

    name = PROVIDER_ID
    display_name = f"{DISPLAY_NAME} (openrouter:web_search + openrouter:web_fetch)"

    def is_available(self) -> bool:
        return has_credentials()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return list(_SETUP_SCHEMA_ENV)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "1-key everything",
            "tag": (
                f"Default search: openrouter:web_search ({DEFAULTS['web_search']}); "
                f"extract: openrouter:web_fetch ({DEFAULTS['web_fetch']}); uses OPENROUTER_API_KEY"
            ),
            "env_vars": list(_SETUP_SCHEMA_ENV),
        }

    def search(self, query: str, limit: int = 5, *, model: Optional[str] = None, engine: Optional[str] = None) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "Search query is required"}

        api_key, base_url = resolve_credentials()
        if not api_key:
            return {
                "success": False,
                "error": "OPENROUTER_API_KEY is not set. Run `hermes openrouter-everything setup` or set OPENROUTER_API_KEY.",
            }

        # Allow shorthand model="openrouter:web_search" to resolve to the configured text/web_search chat model
        raw_model = (model or "").strip()
        if raw_model in ("openrouter:web_search", "web_search"):
            raw_model = ""
        model_id = get_configured_model("web_search", explicit=raw_model or None)
        if model_id in ("openrouter:web_search", "web_search"):
            model_id = DEFAULTS["text"]

        valid, err_msg, _ = validate_model_for_capability(model_id, "web_search")
        if not valid:
            return {"success": False, "error": err_msg, "error_type": "inadequate_model", "model": model_id}

        plugin_cfg = load_plugin_config()
        search_engine = (engine or plugin_cfg.get("web_search_engine") or DEFAULTS["web_search_engine"]).strip().lower()
        if search_engine not in _VALID_SEARCH_ENGINES:
            search_engine = "auto"

        import requests

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Use the web_search tool to search the web and provide a clear, factual synthesis "
                        "with citations. Focus on accuracy, dates, and primary sources."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 1024,
        }

        # Attach openrouter:web_search server tool by default (unless using a direct perplexity/sonar* model with no explicit engine)
        if not model_id.startswith("perplexity/") or engine is not None:
            tool_params: dict[str, Any] = {
                "engine": search_engine,
                "max_results": max(1, min(int(limit), 25)),
            }
            search_mode = plugin_cfg.get("web_search_mode")
            if isinstance(search_mode, str) and search_mode.strip():
                tool_params["mode"] = search_mode.strip()
            payload["tools"] = [{"type": "openrouter:web_search", "parameters": tool_params}]

        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=build_headers(api_key),
                json=payload,
                timeout=45.0,
            )
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "error": f"OpenRouter web search failed (HTTP {resp.status_code}): {resp.text[:300]}",
                }
            data = resp.json()
        except Exception as exc:
            return {"success": False, "error": f"OpenRouter web search request error: {exc}"}

        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        answer_text = (msg.get("content") or "").strip()
        annotations = msg.get("annotations") or []
        citations = data.get("citations") or []

        web_items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        # 1. Extract structured url_citation annotations (returned by openrouter:web_search and Sonar)
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            cit = ann.get("url_citation") if ann.get("type") == "url_citation" else ann
            if not isinstance(cit, dict):
                continue
            url = str(cit.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(cit.get("title") or "").strip() or (urlparse(url).netloc or url)
            snippet = str(cit.get("content") or cit.get("snippet") or "").strip()
            web_items.append({
                "title": title,
                "url": url,
                "description": snippet or answer_text[:400],
                "position": len(web_items) + 1,
            })

        # 2. Fallback to top-level citations array if annotations were empty
        for url in citations:
            if not isinstance(url, str) or not url.strip() or url.strip() in seen_urls:
                continue
            u = url.strip()
            seen_urls.add(u)
            web_items.append({
                "title": urlparse(u).netloc or u,
                "url": u,
                "description": answer_text[:400],
                "position": len(web_items) + 1,
            })

        # 3. Prepend or attach synthesized answer so the agent gets both the direct answer and source URLs
        if answer_text:
            if web_items:
                web_items[0]["description"] = f"{answer_text}\n\nSource: {web_items[0]['title']}"
            else:
                web_items.append({
                    "title": f"OpenRouter Web Search ({model_id})",
                    "url": "https://openrouter.ai",
                    "description": answer_text,
                    "position": 1,
                })

        return {
            "success": True,
            "data": {"web": web_items[: max(1, int(limit))]},
            "model": model_id,
            "tool": "openrouter:web_search",
            "engine": search_engine,
            "provider": self.name,
        }

    def extract(self, urls: List[str], format: Optional[str] = None, *, engine: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract clean markdown from URLs using OpenRouter's ``openrouter:web_fetch`` server tool."""
        del format
        api_key, base_url = resolve_credentials()
        if not api_key:
            return [
                {"url": u, "title": "", "content": "", "error": "OPENROUTER_API_KEY is not set"}
                for u in urls
            ]

        model_id = get_configured_model("web_fetch")
        if model_id in ("openrouter:web_fetch", "openrouter:webfetch", "web_fetch"):
            model_id = DEFAULTS["text"]

        valid, err_msg, _ = validate_model_for_capability(model_id, "web_fetch")
        if not valid:
            return [
                {"url": u, "title": "", "content": "", "error": err_msg}
                for u in urls
            ]

        plugin_cfg = load_plugin_config()
        fetch_engine = (engine or plugin_cfg.get("web_fetch_engine") or DEFAULTS["web_fetch_engine"]).strip().lower()
        if fetch_engine not in _VALID_FETCH_ENGINES:
            fetch_engine = "openrouter"

        import requests

        results: list[dict[str, Any]] = []
        for url in urls:
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Use the web_fetch tool to fetch the requested URL and return its full "
                            "clean Markdown content verbatim without summarizing or omitting sections."
                        ),
                    },
                    {"role": "user", "content": f"Fetch and return the full markdown content of: {url}"},
                ],
                "tools": [{"type": "openrouter:web_fetch", "parameters": {"engine": fetch_engine}}],
                "max_tokens": 8192,
            }
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers=build_headers(api_key),
                    json=payload,
                    timeout=60.0,
                )
                if resp.status_code >= 400:
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"OpenRouter web_fetch HTTP {resp.status_code}: {resp.text[:200]}",
                    })
                    continue
                body = resp.json()
                choices = body.get("choices") or []
                content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
                if not content:
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": "OpenRouter web_fetch returned empty content",
                    })
                else:
                    results.append({
                        "url": url,
                        "title": urlparse(url).netloc or url,
                        "content": content,
                        "error": None,
                    })
            except Exception as exc:
                results.append({"url": url, "title": "", "content": "", "error": str(exc)})
        return results


# ---------------------------------------------------------------------------
# 2. Image Generation Provider
# ---------------------------------------------------------------------------
_ASPECT_MAP = {"square": "1:1", "landscape": "16:9", "portrait": "9:16"}
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
_BLOCKED_ENV_BASENAMES = frozenset({
    ".env", ".env.local", ".env.development", ".env.production",
    ".env.test", ".env.staging", ".envrc",
})
_HOME_CREDENTIAL_DIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure",
    ".config/gh", ".config/gcloud",
)
_HOME_CREDENTIAL_FILES = (".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials")
_HERMES_CREDENTIAL_FILES = (
    "auth.json", "auth.lock", ".anthropic_oauth.json", ".env",
    "webhook_subscriptions.json", "auth/google_oauth.json",
    "cache/bws_cache.json", "cache/bws_cache.enc.json",
)
_HERMES_CREDENTIAL_DIRS = (
    "mcp-tokens", "browser-profile", "vault", "pairing", "skills/.hub", *_HOME_CREDENTIAL_DIRS,
)

# Chat-completions models default to the conservative three-image prompt limit.
# Dedicated Image API models have model-specific limits; its generic ceiling is 16.
_CHAT_IMAGE_REFERENCE_LIMIT = 3
_IMAGE_API_REFERENCE_LIMIT = 16
_CHAT_MODEL_REFERENCE_LIMITS = {
    "google/gemini-nano-banana-2.1": 14,
}
_IMAGE_API_MODEL_REFERENCE_LIMITS = {
    "google/gemini-3.1-flash-lite-image": 14,
    "google/gemini-3.1-flash-image": 14,
    "openai/gpt-image-2": 16,
    "openai/gpt-image-1-mini": 16,
    "microsoft/mai-image-2.5": 1,
    "microsoft/mai-image-2.5-pro": 1,
    "x-ai/grok-imagine-image-quality": 3,
    "krea/krea-2-medium": 1,
    "krea/krea-2-medium-turbo": 1,
    "qwen/qwen-image-3-pro": 4,
}


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _protected_local_path(path: Path) -> bool:
    """Apply Hermes' credential-read policy to an already-resolved local path.

    This is defense in depth, not a race-proof filesystem boundary. Ordinary
    symlink aliases are caught by resolution; opened files also get a hard-link
    count check below. Protected contents remain blocked even with valid media
    magic bytes.
    """
    if path.name.lower() in _BLOCKED_ENV_BASENAMES:
        return True

    active_home = get_hermes_home().expanduser().resolve()
    root = get_default_hermes_root().expanduser().resolve()
    hermes_homes = {active_home, root}
    # Also protect another profile's credential store, not just the active one.
    profiles = root / "profiles"
    if _is_under(path, profiles):
        relative = path.relative_to(profiles)
        if relative.parts:
            hermes_homes.add(profiles / relative.parts[0])

    homes: set[Path] = set()
    for candidate in (Path.home(), Path(get_real_home()), get_subprocess_home()):
        if candidate:
            homes.add(Path(candidate).expanduser().resolve())
    # A named profile can also own an isolated subprocess HOME.
    homes.update(hermes_home / "home" for hermes_home in hermes_homes)
    for home in homes:
        if any(path == (home / name).resolve() for name in _HOME_CREDENTIAL_FILES):
            return True
        if any(_is_under(path, (home / directory).resolve()) for directory in _HOME_CREDENTIAL_DIRS):
            return True

    for hermes_home in hermes_homes:
        if any(path == (hermes_home / name).resolve() for name in _HERMES_CREDENTIAL_FILES):
            return True
        if any(_is_under(path, (hermes_home / directory).resolve()) for directory in _HERMES_CREDENTIAL_DIRS):
            return True
    return False


def _resolve_local_path(value: str, label: str) -> Path:
    """Resolve and policy-check a local media path without silently dropping it."""
    if not value:
        raise ValueError(f"{label} is blank")
    # Check Windows device/NT namespaces before resolve(): merely touching a
    # remote UNC variant can initiate SMB authentication on Windows.
    nt_form = value.replace("/", "\\")
    if nt_form.startswith(("\\??\\", "\\\\.\\")) or (
        nt_form.startswith("\\\\?\\")
        and nt_form[4:].upper().startswith(("UNC\\", "GLOBALROOT\\"))
    ):
        raise ValueError(f"{label} uses a blocked Windows device/NT namespace path")
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} path could not be resolved") from exc
    if _protected_local_path(path):
        raise ValueError(f"{label} is a protected credential or internal Hermes path")
    return path


def _opened_file_stat(handle: Any, label: str) -> os.stat_result:
    """Validate the opened object, not a potentially replaced directory entry."""
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} does not exist or is not a regular file")
    if metadata.st_nlink > 1:
        raise ValueError(f"{label} has multiple hard links and is not accepted")
    return metadata


def _declared_image_reference_limit(entry: dict[str, Any]) -> Optional[int]:
    for key in ("max_reference_images", "max_refs"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _image_reference_limit(model_id: str, entry: dict[str, Any]) -> int:
    declared = _declared_image_reference_limit(entry)
    if declared is not None:
        return declared
    raw_endpoints = entry.get("endpoints")
    endpoints = set(raw_endpoints) if isinstance(raw_endpoints, (list, tuple, set, frozenset)) else set()
    use_chat = "/api/v1/chat/completions" in endpoints or not endpoints
    if use_chat:
        return _CHAT_MODEL_REFERENCE_LIMITS.get(model_id, _CHAT_IMAGE_REFERENCE_LIMIT)
    if "/api/v1/images" in endpoints:
        return _IMAGE_API_MODEL_REFERENCE_LIMITS.get(model_id, _IMAGE_API_REFERENCE_LIMIT)
    return _CHAT_IMAGE_REFERENCE_LIMIT


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    """Return a provider-safe raster MIME from magic bytes, never the filename."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _encode_image_bytes(raw: bytes) -> str:
    if not raw:
        raise ValueError("image reference is empty")
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError("image reference exceeds the 25 MiB limit")
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ValueError("image reference is not a supported PNG, JPEG, GIF, or WebP image")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _load_image_reference(ref: str) -> str:
    """Validate and convert one remote URL, data URL, or local image path.

    Remote HTTP(S) URLs pass through unchanged for OpenRouter to fetch. Local and
    inline inputs are size-capped and authenticated by content before any bytes
    are put on the wire. Invalid inputs raise instead of being silently dropped.
    """
    value = str(ref or "").strip()
    if not value:
        raise ValueError("image reference is blank")
    lowered = value.lower()
    if lowered.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if not parsed.netloc:
            raise ValueError("image URL has no host")
        return value
    if lowered.startswith("data:"):
        header, separator, encoded = value.partition(",")
        header_parts = header.lower().split(";")
        if not separator or not header_parts[0].startswith("data:image/") or "base64" not in header_parts[1:]:
            raise ValueError("only base64-encoded data:image/ URLs are supported")
        # Refuse an oversized payload before allocating its decoded copy.
        if len(encoded) > ((_MAX_INPUT_IMAGE_BYTES + 2) // 3) * 4 + 4:
            raise ValueError("image data URL exceeds the 25 MiB limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image data URL contains invalid base64") from exc
        return _encode_image_bytes(raw)

    parsed = urlparse(value)
    if parsed.scheme and (len(parsed.scheme) > 1 or "://" in value):
        raise ValueError(f"unsupported image reference scheme '{parsed.scheme}'; use HTTP(S), data:image/, or a local path")

    path = _resolve_local_path(value, "image reference")
    try:
        if not path.is_file():
            raise ValueError(f"image reference does not exist or is not a file: {value}")
        with path.open("rb") as handle:
            metadata = _opened_file_stat(handle, "image reference")
            if metadata.st_size > _MAX_INPUT_IMAGE_BYTES:
                raise ValueError("image reference exceeds the 25 MiB limit")
            raw = handle.read(_MAX_INPUT_IMAGE_BYTES + 1)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"could not read image reference: {value}") from exc
    return _encode_image_bytes(raw)


def _ref_to_data_or_url(ref: str) -> Optional[str]:
    try:
        return _load_image_reference(ref)
    except (OSError, ValueError):
        return None


class OpenRouterEverythingImageProvider(ImageGenProvider):
    """Image generation via OpenRouter (defaults to ``google/gemini-nano-banana-2.1``)."""

    name = PROVIDER_ID
    display_name = f"{DISPLAY_NAME} (Nano Banana 2.1 & 60+ models)"

    def is_available(self) -> bool:
        return has_credentials()

    def default_model(self) -> Optional[str]:
        return DEFAULTS["image_gen"]

    def capabilities(self) -> Dict[str, Any]:
        model_id = get_configured_model("image_gen")
        entry = fetch_full_catalog().get(model_id) or {}
        raw_architecture = entry.get("architecture")
        architecture = raw_architecture if isinstance(raw_architecture, dict) else {}
        input_modalities = architecture.get("input_modalities")
        # Missing metadata is unknown, not evidence of edit support. Reference
        # limits are selected for the actual model and API surface, not globalized.
        if not isinstance(input_modalities, list) or "image" not in input_modalities:
            return {"modalities": ["text"], "max_reference_images": 0}
        return {
            "modalities": ["text", "image"],
            "max_reference_images": _image_reference_limit(model_id, entry),
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "1-key everything",
            "tag": f"Default: {DEFAULTS['image_gen']} — text-to-image & editing (up to 14 refs); uses OPENROUTER_API_KEY",
            "env_vars": list(_SETUP_SCHEMA_ENV),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        catalog = fetch_image_catalog()
        rows: list[dict[str, Any]] = []
        default_id = DEFAULTS["image_gen"]
        for entry in catalog:
            mid = entry.get("id")
            if not mid:
                continue
            raw_architecture = entry.get("architecture")
            arch = raw_architecture if isinstance(raw_architecture, dict) else {}
            raw_in_mods = arch.get("input_modalities")
            in_mods = raw_in_mods if isinstance(raw_in_mods, list) else []
            if not isinstance(raw_in_mods, list) or not raw_in_mods:
                strengths = "Image generation (input modalities unavailable)"
            else:
                strengths = "Text + Image editing" if "image" in in_mods else "Text-to-image"
            if mid == default_id:
                strengths = f"★ Default · {strengths}"
            rows.append({
                "id": mid,
                "display": entry.get("name") or mid,
                "strengths": strengths,
                "input_modalities": in_mods,
            })
        rows.sort(key=lambda r: (0 if r["id"] == default_id else 1, r["id"]))
        if not rows:
            rows.append({
                "id": default_id,
                "display": "Google: Nano Banana 2.1",
                "strengths": "★ Default · Fast image generation & editing (up to 14 reference images)",
            })
        return rows

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        semantic_aspect = resolve_aspect_ratio(aspect_ratio)
        model_id = get_configured_model("image_gen", explicit=model)

        def fail(msg: str, err_type: str = "api_error") -> Dict[str, Any]:
            return img_error_response(
                error=msg, error_type=err_type, provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=semantic_aspect,
            )

        if not prompt:
            return fail("prompt is required", "invalid_request")

        api_key, base_url = resolve_credentials()
        if not api_key:
            return fail("OPENROUTER_API_KEY is not set.", "missing_api_key")

        # Validate model adequacy BEFORE spending tokens/requests
        valid, err_msg, entry = validate_model_for_capability(model_id, "image_gen")
        if not valid:
            return fail(err_msg or f"Model '{model_id}' does not support image generation.", "inadequate_model")

        refs: list[str] = []
        if image_url:
            refs.append(str(image_url))
        if reference_image_urls:
            refs.extend(reference_image_urls)
        if kwargs.get("reference_images"):
            refs.extend(kwargs["reference_images"])

        if refs and entry:
            raw_architecture = entry.get("architecture")
            architecture = raw_architecture if isinstance(raw_architecture, dict) else {}
            input_modalities = architecture.get("input_modalities")
            if not isinstance(input_modalities, list) or "image" not in input_modalities:
                return fail(
                    f"Model '{model_id}' does not accept image inputs "
                    f"(input_modalities={input_modalities}); choose an image-editing model.",
                    "inadequate_model",
                )
        max_refs = _image_reference_limit(model_id, entry or {})
        if len(refs) > max_refs:
            return fail(
                f"Model '{model_id}' accepts at most {max_refs} image references; "
                f"received {len(refs)}. Remove extra references; none were dropped.",
                "invalid_request",
            )
        usable_refs: list[str] = []
        for index, ref in enumerate(refs):
            try:
                usable_refs.append(_load_image_reference(ref))
            except (OSError, ValueError) as exc:
                return fail(f"Invalid image reference {index + 1}: {exc}", "invalid_request")
        exact_ratio = _ASPECT_MAP.get(semantic_aspect, "1:1")
        headers = build_headers(api_key)

        import requests

        # Check if model supports dedicated /images/generations endpoint or /chat/completions
        endpoints = set((entry or {}).get("endpoints") or [])
        use_chat = "/api/v1/chat/completions" in endpoints or not endpoints

        if not use_chat and "/api/v1/images" in endpoints:
            # Dedicated Image API
            payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "aspect_ratio": exact_ratio,
            }
            if usable_refs:
                payload["input_references"] = [{"type": "image_url", "image_url": {"url": u}} for u in usable_refs]
            try:
                resp = requests.post(f"{base_url}/images/generations", headers=headers, json=payload, timeout=180.0)
                if resp.status_code >= 400:
                    return fail(f"OpenRouter Image API error (HTTP {resp.status_code}): {resp.text[:300]}")
                body = resp.json()
                items = body.get("data") or []
                if not items:
                    return fail(f"OpenRouter Image API returned no image for '{model_id}'.", "empty_response")
                first = items[0]
                if first.get("b64_json"):
                    saved = save_b64_image(first["b64_json"], prefix="openrouter_everything")
                elif first.get("url"):
                    saved = save_url_image(first["url"], prefix="openrouter_everything")
                else:
                    return fail("OpenRouter Image API entry had neither b64_json nor url.", "empty_response")
                return img_success_response(
                    image=str(saved), model=model_id, prompt=prompt,
                    aspect_ratio=semantic_aspect, provider=self.name,
                    modality="image" if usable_refs else "text",
                )
            except Exception as exc:
                return fail(f"OpenRouter Image API request failed: {exc}")

        # Chat Completions multimodal image generation (supported by google/gemini-nano-banana-2.1, gpt-5.4-image-2, etc.)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for u in usable_refs:
            content.append({"type": "image_url", "image_url": {"url": u}})

        chat_payload: dict[str, Any] = {
            "model": model_id,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
            "image_config": {"aspect_ratio": exact_ratio},
        }
        try:
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=chat_payload, timeout=180.0)
            if resp.status_code >= 400:
                return fail(f"OpenRouter image generation error (HTTP {resp.status_code}): {resp.text[:300]}")
            body = resp.json()
            choices = body.get("choices") or []
            images = ((choices[0].get("message") or {}).get("images") or []) if choices else []
            if not images:
                return fail(f"OpenRouter model '{model_id}' returned no image.", "empty_response")
            img_url = ((images[0].get("image_url") or {}).get("url") or "").strip()
            if not img_url:
                return fail("OpenRouter response image entry was empty.", "empty_response")
            if img_url.startswith("data:"):
                b64 = img_url.split(",", 1)[1] if "," in img_url else ""
                saved = save_b64_image(b64, prefix="openrouter_everything")
            else:
                saved = save_url_image(img_url, prefix="openrouter_everything")
            return img_success_response(
                image=str(saved), model=model_id, prompt=prompt,
                aspect_ratio=semantic_aspect, provider=self.name,
                modality="image" if usable_refs else "text",
            )
        except Exception as exc:
            return fail(f"OpenRouter image generation failed: {exc}")


# ---------------------------------------------------------------------------
# 3. Video Generation Provider
# ---------------------------------------------------------------------------
_TERMINAL_VIDEO_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled", "expired"})
_RES_HEIGHT = {"480p": 480, "540p": 540, "720p": 720, "768p": 768, "1080p": 1080, "1K": 1024, "2K": 1440, "4K": 2160}


def _clamp_nearest(val: Any, supported: list[Any], height_map: Optional[dict[str, int]] = None) -> Any:
    if not supported or val in supported:
        return val
    if height_map is not None:
        target = height_map.get(str(val))
        if target is None:
            return supported[0]
        scored = [(abs(height_map[str(s)] - target), i, s) for i, s in enumerate(supported) if str(s) in height_map]
        return min(scored)[2] if scored else supported[0]
    try:
        return min(supported, key=lambda s: (abs(int(s) - int(val)), s))
    except (TypeError, ValueError):
        return supported[0]


class OpenRouterEverythingVideoProvider(VideoGenProvider):
    """Video generation via OpenRouter ``/api/v1/videos`` (defaults to ``alibaba/wan-3.0-prime``)."""

    name = PROVIDER_ID
    display_name = f"{DISPLAY_NAME} (Wan 3.0 Prime & 30+ models)"
    _poll_interval_s = 8.0
    _poll_deadline_s = 900.0

    def is_available(self) -> bool:
        return has_credentials()

    def default_model(self) -> Optional[str]:
        return DEFAULTS["video_gen"]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "1-key everything",
            "tag": f"Default: {DEFAULTS['video_gen']} (2-30s, 480p/720p/1080p, audio, i2v); uses OPENROUTER_API_KEY",
            "env_vars": list(_SETUP_SCHEMA_ENV),
        }

    def _entry(self, model_id: str) -> dict[str, Any]:
        for e in fetch_video_catalog():
            if e.get("id") == model_id:
                return e
        if model_id == "alibaba/wan-3.0-prime":
            return {
                "id": "alibaba/wan-3.0-prime",
                "name": "Alibaba: Wan 3.0 Prime",
                "supported_resolutions": ["480p", "720p", "1080p"],
                "supported_aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
                "supported_durations": list(range(2, 31)),
                "supported_frame_images": ["first_frame", "last_frame"],
                "generate_audio": True,
                "seed": True,
            }
        return {}

    def capabilities(self) -> Dict[str, Any]:
        entry = self._entry(get_configured_model("video_gen"))
        durations = [int(d) for d in (entry.get("supported_durations") or range(2, 31))]
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": list(entry.get("supported_aspect_ratios") or ["16:9", "4:3", "1:1", "3:4", "9:16"]),
            "resolutions": list(entry.get("supported_resolutions") or ["480p", "720p", "1080p"]),
            "min_duration": min(durations) if durations else 2,
            "max_duration": max(durations) if durations else 30,
            "supports_audio": bool(entry.get("generate_audio", True)),
            "supports_seed": bool(entry.get("seed", True)),
            "supports_negative_prompt": False,
            "supports_upscale": False,
            "max_reference_images": 3,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        default_id = DEFAULTS["video_gen"]
        rows: list[dict[str, Any]] = []
        for entry in fetch_video_catalog():
            mid = entry.get("id")
            durations = entry.get("supported_durations") or []
            if not mid or not durations:
                continue
            res = "/".join(entry.get("supported_resolutions") or ["720p"])
            strengths = f"{min(durations)}-{max(durations)}s, {res}"
            if entry.get("generate_audio"):
                strengths += ", audio"
            if mid == default_id:
                strengths = f"★ Default · {strengths}"
            rows.append({
                "id": mid,
                "display": entry.get("name") or mid,
                "strengths": strengths,
                "modalities": ["text", "image"] if entry.get("supported_frame_images") else ["text"],
                "min_duration": min(durations),
                "max_duration": max(durations),
            })
        rows.sort(key=lambda r: (0 if r["id"] == default_id else 1, r["id"]))
        return rows

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del negative_prompt, kwargs
        prompt = (prompt or "").strip()
        model_id = get_configured_model("video_gen", explicit=model)

        def fail(msg: str, err_type: str = "api_error") -> Dict[str, Any]:
            return vid_error_response(
                error=msg, error_type=err_type, provider=self.name,
                model=model_id, prompt=prompt, aspect_ratio=aspect_ratio,
            )

        if not prompt:
            return fail("prompt is required", "invalid_request")

        api_key, base_url = resolve_credentials()
        if not api_key:
            return fail("OPENROUTER_API_KEY is not set.", "missing_credentials")

        # Validate model adequacy BEFORE submitting a video job!
        valid, err_msg, _ = validate_model_for_capability(model_id, "video_gen")
        if not valid:
            return fail(err_msg or f"Model '{model_id}' cannot be used for video generation.", "inadequate_model")

        entry = self._entry(model_id)
        payload: dict[str, Any] = {"model": model_id, "prompt": prompt}
        if aspect_ratio:
            sup_ar = list(entry.get("supported_aspect_ratios") or [])
            payload["aspect_ratio"] = aspect_ratio if not sup_ar or aspect_ratio in sup_ar else sup_ar[0]
        if resolution:
            sup_res = list(entry.get("supported_resolutions") or [])
            payload["resolution"] = _clamp_nearest(resolution, sup_res, _RES_HEIGHT)
        if duration:
            sup_dur = list(entry.get("supported_durations") or [])
            payload["duration"] = int(_clamp_nearest(int(duration), sup_dur))
        if image_url:
            try:
                part_url = _load_image_reference(image_url)
            except (OSError, ValueError) as exc:
                return fail(f"Invalid image_url: {exc}", "invalid_request")
            payload["frame_images"] = [{"type": "image_url", "image_url": {"url": part_url}, "frame_type": "first_frame"}]
        if reference_image_urls:
            if len(reference_image_urls) > 3:
                return fail(
                    f"Model '{model_id}' accepts at most 3 reference images; "
                    f"received {len(reference_image_urls)}. Remove extra references; none were dropped.",
                    "invalid_request",
                )
            ref_parts: list[str] = []
            for index, ref in enumerate(reference_image_urls):
                try:
                    ref_parts.append(_load_image_reference(ref))
                except (OSError, ValueError) as exc:
                    return fail(f"Invalid reference_image_urls[{index}]: {exc}", "invalid_request")
            payload["input_references"] = [{"type": "image_url", "image_url": {"url": u}} for u in ref_parts]
        if audio is not None and entry.get("generate_audio", True):
            payload["generate_audio"] = bool(audio)
        if seed is not None and entry.get("seed", True):
            payload["seed"] = int(seed)

        import requests

        headers = build_headers(api_key)
        session = requests.Session()
        try:
            sub = session.post(f"{base_url}/videos", headers=headers, json=payload, timeout=60.0)
            if sub.status_code >= 400:
                return fail(f"OpenRouter video submit rejected (HTTP {sub.status_code}): {sub.text[:300]}")
            job_id = str(sub.json().get("id") or "").strip()
            if not job_id:
                return fail("OpenRouter video response did not contain a job id")

            deadline = time.monotonic() + self._poll_deadline_s
            last_status = "pending"
            job_data: dict[str, Any] = {}
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return fail(f"Video job {job_id} timed out after {int(self._poll_deadline_s)}s (status={last_status})", "timeout")
                r = session.get(f"{base_url}/videos/{job_id}", headers=headers, timeout=min(30.0, rem))
                r.raise_for_status()
                job_data = r.json()
                last_status = str(job_data.get("status") or "").lower()
                if last_status in _TERMINAL_VIDEO_STATUSES:
                    break
                time.sleep(min(self._poll_interval_s, max(0.1, deadline - time.monotonic())))

            if last_status != "completed":
                return fail(f"Video job {job_id} ended with status '{last_status}': {job_data.get('error')}", "job_failed")

            video_path = str(save_url_video(
                f"{base_url}/videos/{job_id}/content",
                prefix="openrouter_everything",
                headers=headers,
                require_video_content_type=True,
                trusted_origin=True,
            ))
            usage = job_data.get("usage") if isinstance(job_data.get("usage"), dict) else {}
            extra: dict[str, Any] = {"job_id": job_id}
            if usage.get("cost") is not None:
                extra["cost"] = usage["cost"]
            return vid_success_response(
                video=video_path,
                model=model_id,
                prompt=prompt,
                modality="image" if image_url else "text",
                aspect_ratio=str(payload.get("aspect_ratio") or aspect_ratio),
                duration=int(payload.get("duration") or duration or 5),
                provider=self.name,
                extra=extra,
            )
        except Exception as exc:
            return fail(f"OpenRouter video generation failed: {exc}")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# 4. Text-to-Speech (TTS) Provider
# ---------------------------------------------------------------------------
class OpenRouterEverythingTTSProvider(TTSProvider):
    """Text-to-speech via OpenRouter ``POST /api/v1/audio/speech`` (defaults to ``fish-audio/drama-3``)."""

    name = PROVIDER_ID
    display_name = f"{DISPLAY_NAME} (Fish Audio Drama-3 & 20+ TTS models)"
    voice_compatible = True

    def is_available(self) -> bool:
        return has_credentials()

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "1-key everything",
            "tag": f"Default: {DEFAULTS['tts']} (expressive inline [emotion] tags, multi-speaker); uses OPENROUTER_API_KEY",
            "env_vars": list(_SETUP_SCHEMA_ENV),
        }

    def list_voices(self) -> List[Dict[str, Any]]:
        return [
            {"id": "", "name": "Model Default (Expressive / Auto)", "language": "multilingual"},
            {"id": "alloy", "name": "Alloy (OpenAI TTS)", "language": "multilingual"},
            {"id": "nova", "name": "Nova (OpenAI TTS)", "language": "multilingual"},
        ]

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
    ) -> str:
        api_key, base_url = resolve_credentials()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Run `hermes openrouter-everything setup` or set OPENROUTER_API_KEY.")

        model_id = get_configured_model("tts", explicit=model)
        valid, err_msg, entry = validate_model_for_capability(model_id, "tts")
        if not valid:
            raise ValueError(err_msg or f"Model '{model_id}' does not support Text-to-Speech.")

        # Determine response_format from output_path or format arg
        ext = Path(output_path).suffix.lstrip(".").lower()
        resp_fmt = ext if ext in ("mp3", "wav", "ogg", "opus", "flac", "pcm", "aac") else (format or "mp3")
        if resp_fmt == "ogg":
            resp_fmt = "opus"

        payload: dict[str, Any] = {
            "model": model_id,
            "input": text,
            "response_format": resp_fmt if resp_fmt in ("mp3", "wav", "pcm", "opus", "flac") else "mp3",
        }

        # Only pass `voice` when the model supports voices or user explicitly configured one for it
        plugin_cfg = load_plugin_config()
        resolved_voice = (voice or plugin_cfg.get("tts_voice") or "").strip()
        # Guard against Hermes' default OpenAI voice ("alloy") leaking into fish-audio/drama-3
        if resolved_voice == "alloy" and not model_id.startswith("openai/"):
            resolved_voice = ""
        if resolved_voice:
            payload["voice"] = resolved_voice
        elif model_id.startswith("openai/"):
            payload["voice"] = "alloy"

        if speed is not None and speed != 1.0 and "speed" in ((entry or {}).get("supported_parameters") or []):
            payload["speed"] = max(0.25, min(4.0, float(speed)))

        import requests

        resp = requests.post(
            f"{base_url}/audio/speech",
            headers=build_headers(api_key),
            json=payload,
            timeout=90.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter TTS error on '{model_id}' (HTTP {resp.status_code}): {resp.text[:300]}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return str(out)


# ---------------------------------------------------------------------------
# 5. Speech-to-Text (STT / Transcription) Provider
# ---------------------------------------------------------------------------
class OpenRouterEverythingSTTProvider(TranscriptionProvider):
    """Speech-to-text via OpenRouter ``POST /api/v1/audio/transcriptions``
    (defaults to ``openai/whisper-large-v3-turbo``)."""

    name = PROVIDER_ID
    display_name = f"{DISPLAY_NAME} (Whisper Large v3 Turbo, Voxtral, Parakeet & 17 STT models)"

    def is_available(self) -> bool:
        return has_credentials()

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "model": {
                "type": "str",
                "default": DEFAULTS["stt"],
                "description": "OpenRouter STT model slug (e.g. openai/whisper-large-v3-turbo, mistralai/voxtral-mini-transcribe-3-0)",
            },
            "language": {
                "type": "str",
                "default": "",
                "description": "Optional ISO-639-1 language code hint",
            },
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_key, base_url = resolve_credentials()
        if not api_key:
            return {
                "success": False,
                "transcript": "",
                "error": "OPENROUTER_API_KEY is not set.",
                "provider": self.name,
            }

        model_id = get_configured_model("stt", explicit=model)
        valid, err_msg, entry = validate_model_for_capability(model_id, "stt")
        if not valid:
            return {
                "success": False,
                "transcript": "",
                "error": err_msg or f"Model '{model_id}' does not support Speech-to-Text transcription.",
                "error_type": "inadequate_model",
                "provider": self.name,
            }

        try:
            p = _resolve_local_path(str(file_path or "").strip(), "audio file")
        except (OSError, ValueError) as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"Invalid audio file: {exc}",
                "error_type": "invalid_request",
                "provider": self.name,
            }
        if not p.is_file():
            return {
                "success": False,
                "transcript": "",
                "error": f"Audio file not found: {file_path}",
                "error_type": "invalid_request",
                "provider": self.name,
            }

        mime = mimetypes.guess_type(p.name)[0] or "audio/mpeg"
        sup_params = set((entry or {}).get("supported_parameters") or [])
        form_data: dict[str, Any] = {"model": model_id}
        if language and ("language" in sup_params or not sup_params):
            form_data["language"] = language
        if prompt and ("prompt" in sup_params or not sup_params):
            form_data["prompt"] = prompt

        import httpx

        try:
            with p.open("rb") as f:
                try:
                    _opened_file_stat(f, "audio file")
                except (OSError, ValueError) as exc:
                    return {
                        "success": False,
                        "transcript": "",
                        "error": f"Invalid audio file: {exc}",
                        "error_type": "invalid_request",
                        "provider": self.name,
                    }
                resp = httpx.post(
                    f"{base_url}/audio/transcriptions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-Title": "Hermes Agent (OpenRouter Everything)",
                    },
                    data=form_data,
                    files={"file": (p.name, f, mime)},
                    timeout=120.0,
                )
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "transcript": "",
                    "error": f"OpenRouter STT failed on '{model_id}' (HTTP {resp.status_code}): {resp.text[:300]}",
                    "provider": self.name,
                }
            body = resp.json()
            transcript = str(body.get("text") or body.get("transcript") or "").strip()
            return {
                "success": True,
                "transcript": transcript,
                "model": body.get("model") or model_id,
                "usage": body.get("usage"),
                "provider": self.name,
            }
        except Exception as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"OpenRouter STT request failed: {exc}",
                "provider": self.name,
            }
