"""CLI (`hermes openrouter-everything`) and TUI slash (`/openrouter-everything`) commands."""

from __future__ import annotations

import argparse
from typing import Any

from .catalog import (
    CAPABILITY_SPECS, DEFAULTS, PROVIDER_ID, get_configured_model,
    has_credentials, list_models_for_capability, save_plugin_config_key,
    validate_model_for_capability,
)
from .tools import activate_all_providers_in_config


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("status", help="Show current models, adequacy status, and active Hermes backends")

    p_setup = sub.add_parser("setup", help="One-step setup: configure OPENROUTER_API_KEY and activate all capabilities")
    p_setup.add_argument("--api-key", help="Optional OpenRouter API key to save to .env")
    p_setup.add_argument("--activate-all", action="store_true", default=True, help="Set openrouter-everything as active provider for web, image_gen, video_gen, tts, stt")

    p_set = sub.add_parser("set-model", help="Switch the model for a capability (with adequacy validation)")
    p_set.add_argument("capability", choices=list(CAPABILITY_SPECS.keys()), help="Capability to configure")
    p_set.add_argument("model", help="OpenRouter model slug (e.g. alibaba/wan-3.0-prime)")

    p_val = sub.add_parser("validate", help="Validate whether a model is adequate for a capability")
    p_val.add_argument("capability", choices=list(CAPABILITY_SPECS.keys()), help="Target capability")
    p_val.add_argument("model", nargs="?", help="Model slug to test (defaults to currently configured model)")

    p_models = sub.add_parser("models", help="List compatible OpenRouter models for a capability")
    p_models.add_argument("capability", choices=list(CAPABILITY_SPECS.keys()), help="Target capability")
    p_models.add_argument("--limit", type=int, default=25, help="Max models to print")

    p_eng = sub.add_parser("set-engine", help="Set the server-tool engine for web_search or web_fetch")
    p_eng.add_argument("tool", choices=["web_search", "web_fetch"], help="Which server tool to configure")
    p_eng.add_argument("engine", help="Engine: auto|native|exa|parallel|firecrawl|perplexity (or openrouter for web_fetch)")

    sub.add_parser("reset", help="Reset all capability models back to the curated defaults")


def _format_status_text() -> str:
    from hermes_cli.config import load_config

    cfg = load_config()
    plugin_cfg = cfg.get("openrouter_everything") if isinstance(cfg.get("openrouter_everything"), dict) else {}
    search_eng = plugin_cfg.get("web_search_engine") or DEFAULTS["web_search_engine"]
    fetch_eng = plugin_cfg.get("web_fetch_engine") or DEFAULTS["web_fetch_engine"]
    key_ok = has_credentials()
    lines = [
        "OpenRouter Everything Plugin — Status",
        "=" * 74,
        f"OPENROUTER_API_KEY : {'configured' if key_ok else 'MISSING (set OPENROUTER_API_KEY or run `hermes openrouter-everything setup`)'}",
        f"Web Search Tool    : openrouter:web_search (engine={search_eng})",
        f"Web Fetch Tool     : openrouter:web_fetch  (engine={fetch_eng})",
        "",
        "Active Hermes Providers:",
        f"  web.backend        : {(cfg.get('web') or {}).get('backend', '(auto)')}",
        f"  image_gen.provider : {(cfg.get('image_gen') or {}).get('provider', '(auto)')}",
        f"  video_gen.provider : {(cfg.get('video_gen') or {}).get('provider', '(auto)')}",
        f"  tts.provider       : {(cfg.get('tts') or {}).get('provider', '(auto)')}",
        f"  stt.provider       : {(cfg.get('stt') or {}).get('provider', '(auto)')}",
        "",
        f"{'Capability':<14} {'Active Model':<38} {'Status'}",
        "-" * 74,
    ]
    for cap, spec in CAPABILITY_SPECS.items():
        cur = get_configured_model(cap)
        ok, err, _ = validate_model_for_capability(cur, cap)
        extra = ""
        if cap == "web_search" and not cur.startswith("perplexity/"):
            extra = f" [via openrouter:web_search ({search_eng})]"
        elif cap == "web_fetch":
            extra = f" [via openrouter:web_fetch ({fetch_eng})]"
        status = f"OK{extra}" if ok else f"INADEQUATE ({err})"
        lines.append(f"{cap:<14} {cur:<38} {status}")
    return "\n".join(lines)


def handle_cli(args: argparse.Namespace) -> None:
    cmd = getattr(args, "subcommand", None) or "status"

    if cmd == "status":
        print(_format_status_text())
        return

    if cmd == "setup":
        from hermes_cli.config import save_env_value

        api_key = getattr(args, "api_key", None)
        if api_key:
            save_env_value("OPENROUTER_API_KEY", api_key.strip())
            print("Saved OPENROUTER_API_KEY to ~/.hermes/.env")
        elif not has_credentials():
            import sys
            if sys.stdin and sys.stdin.isatty():
                try:
                    entered = input("Enter your OpenRouter API key (https://openrouter.ai/settings/keys): ").strip()
                    if entered:
                        save_env_value("OPENROUTER_API_KEY", entered)
                        print("Saved OPENROUTER_API_KEY to ~/.hermes/.env")
                except EOFError:
                    pass

        activated = activate_all_providers_in_config()
        print(f"Activated '{PROVIDER_ID}' across Web, Image Gen, Video Gen, TTS, and STT.")
        for k, v in activated["models"].items():
            print(f"  {k:<14} -> {v}")
        return

    if cmd == "set-model":
        cap = args.capability
        mod = args.model.strip()
        ok, err, _ = validate_model_for_capability(mod, cap)
        if not ok:
            print(f"ERROR: {err}")
            raise SystemExit(1)
        save_plugin_config_key(cap, mod)
        print(f"Updated {cap} ({CAPABILITY_SPECS[cap]['label']}) model to: {mod}")
        return

    if cmd == "validate":
        cap = args.capability
        mod = (args.model or get_configured_model(cap)).strip()
        ok, err, _ = validate_model_for_capability(mod, cap)
        if ok:
            print(f"VALID: '{mod}' is adequate for {cap} ({CAPABILITY_SPECS[cap]['label']}).")
        else:
            print(f"INADEQUATE: {err}")
            raise SystemExit(1)
        return

    if cmd == "models":
        cap = args.capability
        rows = list_models_for_capability(cap)
        limit = getattr(args, "limit", 25) or 25
        print(f"Compatible OpenRouter models for {cap} ({CAPABILITY_SPECS[cap]['label']}) — total {len(rows)}:")
        print(f"  Default: {DEFAULTS.get(cap)} | Configured: {get_configured_model(cap)}\n")
        for r in rows[:limit]:
            star = "★ " if r.get("recommended") else "  "
            print(f"  {star}{r['id']:<42} {r['display']} ({r['modality']})")
        return

    if cmd == "set-engine":
        tool_name = args.tool
        eng = args.engine.strip().lower()
        valid_s = {"auto", "native", "exa", "parallel", "firecrawl", "perplexity"}
        valid_f = {"auto", "native", "openrouter", "exa", "parallel", "firecrawl"}
        if tool_name == "web_search" and eng not in valid_s:
            print(f"ERROR: Invalid web_search engine '{eng}'. Choose from: {sorted(valid_s)}")
            raise SystemExit(1)
        if tool_name == "web_fetch" and eng not in valid_f:
            print(f"ERROR: Invalid web_fetch engine '{eng}'. Choose from: {sorted(valid_f)}")
            raise SystemExit(1)
        save_plugin_config_key(f"{tool_name}_engine", eng)
        print(f"Updated openrouter:{tool_name} engine to: {eng}")
        return

    if cmd == "reset":
        for cap, mod in DEFAULTS.items():
            save_plugin_config_key(cap, mod)
        print("Reset all OpenRouter Everything models to curated defaults.")


def handle_slash(args_str: str) -> str:
    parts = (args_str or "").strip().split()
    if not parts or parts[0] == "status":
        return _format_status_text()

    sub = parts[0].lower()
    if sub == "activate-all":
        activate_all_providers_in_config()
        return f"Activated '{PROVIDER_ID}' across web, image_gen, video_gen, tts, and stt.\n\n" + _format_status_text()

    if sub == "set" and len(parts) >= 3:
        cap, mod = parts[1].lower(), parts[2].strip()
        if cap not in CAPABILITY_SPECS:
            return f"Unknown capability '{cap}'. Valid: {', '.join(CAPABILITY_SPECS.keys())}"
        ok, err, _ = validate_model_for_capability(mod, cap)
        if not ok:
            return f"Model validation failed:\n{err}"
        save_plugin_config_key(cap, mod)
        return f"Updated {cap} ({CAPABILITY_SPECS[cap]['label']}) model to '{mod}'."

    if sub == "validate" and len(parts) >= 3:
        cap, mod = parts[1].lower(), parts[2].strip()
        if cap not in CAPABILITY_SPECS:
            return f"Unknown capability '{cap}'. Valid: {', '.join(CAPABILITY_SPECS.keys())}"
        ok, err, _ = validate_model_for_capability(mod, cap)
        return f"VALID: '{mod}' is adequate for {cap}." if ok else f"INADEQUATE:\n{err}"

    if sub == "models" and len(parts) >= 2:
        cap = parts[1].lower()
        if cap not in CAPABILITY_SPECS:
            return f"Unknown capability '{cap}'. Valid: {', '.join(CAPABILITY_SPECS.keys())}"
        rows = list_models_for_capability(cap)[:20]
        lines = [f"Compatible models for {cap} ({len(rows)} shown):"]
        for r in rows:
            star = "★ " if r.get("recommended") else "  "
            lines.append(f"  {star}{r['id']} — {r['display']}")
        return "\n".join(lines)

    return (
        "Usage:\n"
        "  /openrouter-everything status\n"
        "  /openrouter-everything activate-all\n"
        "  /openrouter-everything set <capability> <model>\n"
        "  /openrouter-everything validate <capability> <model>\n"
        "  /openrouter-everything models <capability>"
    )
