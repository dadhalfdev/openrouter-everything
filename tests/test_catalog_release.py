"""Offline release-regression tests for manifest, catalog configuration, and disclosure."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from hermes_yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "plugin.yaml"
CATALOG_PATH = ROOT / "catalog.py"
README_PATH = ROOT / "README.md"


def _load_catalog():
    module_name = "openrouter_everything_catalog_release_test"
    spec = importlib.util.spec_from_file_location(module_name, CATALOG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load_catalog()


def _manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


class ManifestReleaseTests(unittest.TestCase):
    def test_manifest_uses_supported_fields_and_declares_all_optional_env(self):
        manifest = _manifest()
        supported_fields = {
            "name",
            "version",
            "description",
            "author",
            "requires_hermes",
            "requires_env",
            "optional_env",
            "provides_tools",
        }
        self.assertLessEqual(set(manifest), supported_fields)
        self.assertNotIn("provides_cli_commands", manifest)
        self.assertNotIn("provides_slash_commands", manifest)
        self.assertEqual(manifest["requires_env"], ["OPENROUTER_API_KEY"])

        entries = manifest["optional_env"]
        self.assertTrue(all(isinstance(entry, dict) for entry in entries))
        self.assertTrue(all(entry.get("password") is False for entry in entries))
        self.assertTrue(all(entry.get("description") for entry in entries))
        actual = {entry["name"] for entry in entries}
        expected = {"OPENROUTER_BASE_URL"} | {
            f"OPENROUTER_{capability.upper()}_MODEL"
            for capability in catalog.CAPABILITY_SPECS
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(entries), len(actual), "optional_env entries must be unique")

    def test_manifest_tools_match_registrations_and_commands_still_register(self):
        manifest = _manifest()
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        registered_tools = []
        registered_methods = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            registered_methods.add(node.func.attr)
            if node.func.attr != "register_tool":
                continue
            name_node = next((kw.value for kw in node.keywords if kw.arg == "name"), None)
            self.assertIsInstance(name_node, ast.Constant)
            name = getattr(name_node, "value", None)
            self.assertIsInstance(name, str)
            registered_tools.append(name)

        self.assertEqual(set(manifest["provides_tools"]), set(registered_tools))
        self.assertEqual(len(manifest["provides_tools"]), len(registered_tools))
        self.assertIn("register_cli_command", registered_methods)
        self.assertIn("register_command", registered_methods)

    def test_minimum_hermes_covers_trusted_origin_video_downloads(self):
        # Release evidence: v2026.9.14 (Hermes 0.21.3) lacks trusted_origin;
        # 0.22.0 provides the credential-write policy check used by setup.
        self.assertEqual(_manifest()["requires_hermes"], ">=0.22.0")


class CatalogConfigurationTests(unittest.TestCase):
    def test_every_capability_model_env_override_is_used(self):
        requested = []

        def fake_secret(name, default=None):
            requested.append(name)
            if name.startswith("OPENROUTER_") and name.endswith("_MODEL"):
                return f"test/{name.lower()}"
            return default

        with patch("agent.secret_scope.get_secret", side_effect=fake_secret):
            for capability in catalog.CAPABILITY_SPECS:
                with self.subTest(capability=capability):
                    resolved = catalog.get_configured_model(capability)
                    self.assertEqual(
                        resolved,
                        f"test/openrouter_{capability}_model",
                    )

        self.assertEqual(
            set(requested),
            {f"OPENROUTER_{capability.upper()}_MODEL" for capability in catalog.CAPABILITY_SPECS},
        )

    def test_base_url_override_is_trimmed_and_used_for_public_catalog(self):
        secrets = {"OPENROUTER_BASE_URL": "https://router.example.test/api/v1///"}
        original_cache = catalog._CATALOG_CACHE
        response = Mock()
        response.json.return_value = {
            "data": [{"id": "test/model", "architecture": {"output_modalities": ["text"]}}]
        }
        response.raise_for_status.return_value = None
        try:
            with (
                patch("agent.secret_scope.get_secret", side_effect=lambda name, default=None: secrets.get(name, default)),
                patch("requests.get", return_value=response) as get,
            ):
                self.assertEqual(
                    catalog.resolve_credentials(),
                    ("", "https://router.example.test/api/v1"),
                )
                fetched = catalog.fetch_full_catalog(force=True)
            self.assertIn("test/model", fetched)
            self.assertEqual(
                get.call_args.args[0],
                "https://router.example.test/api/v1/models?output_modalities=all",
            )
        finally:
            setattr(catalog, "_CATALOG_CACHE", original_cache)

    def test_tts_model_write_leaves_other_provider_config_untouched(self):
        original = {
            "openrouter_everything": {},
            "tts": {
                "provider": "edge",
                "model": "edge-default",
                "openrouter-everything": {"model": "unchanged"},
            },
        }
        with (
            patch("hermes_cli.config.load_config", return_value=deepcopy(original)),
            patch("hermes_cli.config.save_config") as save,
        ):
            catalog.save_plugin_config_key("tts", "test/speech")

        saved = save.call_args.args[0]
        self.assertEqual(saved["openrouter_everything"]["tts"], "test/speech")
        self.assertEqual(saved["tts"], original["tts"])

    def test_tts_model_write_does_not_create_inactive_section(self):
        with (
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.config.save_config") as save,
        ):
            catalog.save_plugin_config_key("tts", "test/speech")

        saved = save.call_args.args[0]
        self.assertEqual(saved["openrouter_everything"]["tts"], "test/speech")
        self.assertNotIn("tts", saved)

    def test_tts_model_write_syncs_only_active_plugin_provider(self):
        original = {
            "tts": {
                "provider": catalog.PROVIDER_ID,
                "model": "old/speech",
                catalog.PROVIDER_ID: {"model": "old/speech"},
            }
        }
        with (
            patch("hermes_cli.config.load_config", return_value=deepcopy(original)),
            patch("hermes_cli.config.save_config") as save,
        ):
            catalog.save_plugin_config_key("tts", "test/speech")

        saved = save.call_args.args[0]
        self.assertEqual(saved["tts"]["model"], "test/speech")
        self.assertEqual(saved["tts"][catalog.PROVIDER_ID]["model"], "test/speech")

    def test_public_image_default_remains_valid(self):
        model_id = "google/gemini-nano-banana-2.1"
        self.assertEqual(catalog.DEFAULTS["image_gen"], model_id)
        entry = {
            "id": model_id,
            "name": "Google: Nano Banana 2.1",
            "endpoints": ["/api/v1/chat/completions", "/api/v1/images/generations"],
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text", "image"],
                "modality": "text+image->text+image",
            },
        }
        valid, error, resolved = catalog.validate_model_for_capability(
            model_id,
            "image_gen",
            catalog={model_id: entry},
        )
        self.assertTrue(valid, error)
        self.assertIs(resolved, entry)


class ReadmeReleaseTests(unittest.TestCase):
    def test_readme_discloses_network_mutations_costs_and_direct_install(self):
        readme = README_PATH.read_text(encoding="utf-8")
        required_text = {
            "https://github.com/dadhalfdev/openrouter-everything",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_<CAPABILITY>_MODEL",
            "GET /models?output_modalities=all",
            "GET /images/models",
            "GET /videos/models",
            "POST /chat/completions",
            "POST /images/generations",
            "GET /videos/{job_id}",
            "GET /videos/{job_id}/content",
            "POST /audio/speech",
            "POST /audio/transcriptions",
            "POST /rerank",
            "POST /embeddings",
            "https://openrouter.ai/api/alpha/decisions",
            "https://openrouter.ai/api/alpha/extractions",
            "openrouter_everything_config",
            "activate_all",
            "reset_defaults",
            "metered charges",
            "0.22.0 or newer",
        }
        missing = sorted(text for text in required_text if text not in readme)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
