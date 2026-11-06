"""Regression tests for alpha endpoint routing and safe CLI credential setup."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parent))

tools = importlib.import_module("openrouter-everything.tools")
cli = importlib.import_module("openrouter-everything.cli")

FIXTURE_KEY = "sk-test-fixture-not-a-real-credential"


def successful_response(payload: dict | None = None) -> Mock:
    response = Mock(status_code=200, text="")
    response.json.return_value = payload or {"answers": {}}
    return response


class AlphaRoutingTests(unittest.TestCase):
    def test_alpha_base_url_derives_default_and_relay_urls(self) -> None:
        cases = {
            "https://openrouter.ai/api/v1": "https://openrouter.ai/api/alpha",
            "https://openrouter.ai/api/v1/": "https://openrouter.ai/api/alpha",
            "https://relay.example/api/v1": "https://relay.example/api/alpha",
            "http://127.0.0.1:8080/api/v1": "http://127.0.0.1:8080/api/alpha",
        }
        for base_url, expected in cases.items():
            with self.subTest(base_url=base_url):
                self.assertEqual(tools._alpha_base_url(base_url), expected)

    def test_decisions_uses_configured_relay_without_openrouter_fallback(self) -> None:
        with patch.object(
            tools, "resolve_credentials",
            return_value=(FIXTURE_KEY, "https://relay.example/api/v1"),
        ), patch.object(
            tools, "get_configured_model", return_value="fixture/decision"
        ), patch.object(
            tools, "validate_model_for_capability", return_value=(True, "", {})
        ), patch("httpx.post", return_value=successful_response()) as post:
            raw_result = tools.handle_decide({
                "state": "fixture state",
                "options": {"one": "first option"},
            })

        self.assertTrue(json.loads(raw_result)["success"])
        self.assertEqual(post.call_args.args[0], "https://relay.example/api/alpha/decisions")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], f"Bearer {FIXTURE_KEY}"
        )
        self.assertNotIn(FIXTURE_KEY, raw_result)

    def test_extractions_uses_configured_relay_without_openrouter_fallback(self) -> None:
        with patch.object(
            tools, "resolve_credentials",
            return_value=(FIXTURE_KEY, "https://relay.example/api/v1"),
        ), patch.object(
            tools, "get_configured_model", return_value="fixture/extraction"
        ), patch.object(
            tools, "validate_model_for_capability", return_value=(True, "", {})
        ), patch("httpx.post", return_value=successful_response()) as post:
            raw_result = tools.handle_extract_fields({
                "state": "fixture state",
                "fields": {"total": {"type": "number"}},
            })

        self.assertTrue(json.loads(raw_result)["success"])
        self.assertEqual(post.call_args.args[0], "https://relay.example/api/alpha/extractions")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], f"Bearer {FIXTURE_KEY}"
        )
        self.assertNotIn(FIXTURE_KEY, raw_result)

    def test_alpha_handlers_fail_closed_for_unresolvable_base_urls(self) -> None:
        invalid_base_urls = [
            "https://relay.example/v1",
            "https://relay.example/custom/api/v1",
            "https://relay.example/api/v1/chat/completions",
            "https://relay.example/api/v1?target=other",
            "https://relay.example/api/v1#fragment",
            "https://user:fixture-password@relay.example/api/v1",
            "file:///api/v1",
            "https://relay.example:not-a-port/api/v1",
        ]
        calls = [
            (tools.handle_decide, {
                "state": "fixture state", "options": {"one": "first option"},
            }),
            (tools.handle_extract_fields, {
                "state": "fixture state", "fields": {"total": {"type": "number"}},
            }),
        ]
        for base_url in invalid_base_urls:
            for handler, arguments in calls:
                with self.subTest(base_url=base_url, handler=handler.__name__), patch.object(
                    tools, "resolve_credentials", return_value=(FIXTURE_KEY, base_url)
                ), patch("httpx.post") as post:
                    raw_result = handler(arguments)
                    result = json.loads(raw_result)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["error_type"], "unsupported_base_url")
                    self.assertNotIn(base_url, raw_result)
                    self.assertNotIn(FIXTURE_KEY, raw_result)
                    post.assert_not_called()


class CliCredentialTests(unittest.TestCase):
    @staticmethod
    def setup_args(api_key: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(subcommand="setup", api_key=api_key)

    def test_interactive_setup_uses_hidden_prompt_and_never_prints_key(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "has_credentials", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("getpass.getpass", return_value=f"  {FIXTURE_KEY}  ") as prompt, patch(
            "builtins.input", side_effect=AssertionError("echoing input() must not be used")
        ), patch(
            "hermes_cli.config.require_env_writable"
        ) as require_writable, patch(
            "hermes_cli.config.save_env_value"
        ) as save, patch.object(
            cli, "activate_all_providers_in_config", return_value={"models": {}}
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.handle_cli(self.setup_args())

        prompt.assert_called_once()
        require_writable.assert_called_once_with("OPENROUTER_API_KEY", "set")
        save.assert_called_once_with("OPENROUTER_API_KEY", FIXTURE_KEY)
        self.assertIn("Saved OPENROUTER_API_KEY", stdout.getvalue())
        self.assertNotIn(FIXTURE_KEY, stdout.getvalue() + stderr.getvalue())

    def test_non_tty_without_credentials_fails_before_prompt_save_or_activation(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "has_credentials", return_value=False), patch(
            "sys.stdin.isatty", return_value=False
        ), patch("getpass.getpass") as prompt, patch(
            "hermes_cli.config.save_env_value"
        ) as save, patch.object(cli, "activate_all_providers_in_config") as activate, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.handle_cli(self.setup_args())

        self.assertEqual(raised.exception.code, 1)
        prompt.assert_not_called()
        save.assert_not_called()
        activate.assert_not_called()
        self.assertIn("non-interactive", stderr.getvalue())
        self.assertNotIn("Saved OPENROUTER_API_KEY", stdout.getvalue())

    def test_non_tty_with_existing_credentials_still_activates(self) -> None:
        with patch.object(cli, "has_credentials", return_value=True), patch(
            "sys.stdin.isatty", return_value=False
        ), patch("getpass.getpass") as prompt, patch(
            "hermes_cli.config.save_env_value"
        ) as save, patch.object(
            cli, "activate_all_providers_in_config", return_value={"models": {}}
        ) as activate, contextlib.redirect_stdout(io.StringIO()):
            cli.handle_cli(self.setup_args())

        prompt.assert_not_called()
        save.assert_not_called()
        activate.assert_called_once_with()

    def test_explicit_api_key_setup_remains_compatible_with_warning(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("hermes_cli.config.require_env_writable") as require_writable, patch(
            "hermes_cli.config.save_env_value"
        ) as save, patch.object(
            cli, "activate_all_providers_in_config", return_value={"models": {}}
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.handle_cli(self.setup_args(FIXTURE_KEY))

        require_writable.assert_called_once_with("OPENROUTER_API_KEY", "set")
        save.assert_called_once_with("OPENROUTER_API_KEY", FIXTURE_KEY)
        self.assertIn("process listings", stderr.getvalue())
        self.assertIn("Saved OPENROUTER_API_KEY", stdout.getvalue())
        self.assertNotIn(FIXTURE_KEY, stdout.getvalue() + stderr.getvalue())

    def test_refused_env_write_is_not_reported_as_saved_or_activated(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch(
            "hermes_cli.config.require_env_writable",
            side_effect=ValueError("fixture managed-policy refusal"),
        ), patch("hermes_cli.config.save_env_value") as save, patch.object(
            cli, "activate_all_providers_in_config"
        ) as activate, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.handle_cli(self.setup_args(FIXTURE_KEY))

        self.assertEqual(raised.exception.code, 1)
        save.assert_not_called()
        activate.assert_not_called()
        self.assertIn("managed-policy refusal", stderr.getvalue())
        self.assertNotIn("Saved OPENROUTER_API_KEY", stdout.getvalue())
        self.assertNotIn(FIXTURE_KEY, stdout.getvalue() + stderr.getvalue())

    def test_writer_error_cannot_echo_key_material(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("hermes_cli.config.require_env_writable"), patch(
            "hermes_cli.config.save_env_value",
            side_effect=ValueError(f"bad credential: {FIXTURE_KEY}"),
        ), patch.object(cli, "activate_all_providers_in_config") as activate, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.handle_cli(self.setup_args(FIXTURE_KEY))

        self.assertEqual(raised.exception.code, 1)
        activate.assert_not_called()
        self.assertIn("could not write", stderr.getvalue())
        self.assertNotIn("Saved OPENROUTER_API_KEY", stdout.getvalue())
        self.assertNotIn(FIXTURE_KEY, stdout.getvalue() + stderr.getvalue())

    def test_cancelled_hidden_prompt_does_not_save_or_activate(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "has_credentials", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("getpass.getpass", side_effect=EOFError), patch(
            "hermes_cli.config.save_env_value"
        ) as save, patch.object(cli, "activate_all_providers_in_config") as activate, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.handle_cli(self.setup_args())

        self.assertEqual(raised.exception.code, 1)
        save.assert_not_called()
        activate.assert_not_called()
        self.assertIn("cancelled", stderr.getvalue())

    def test_empty_explicit_key_is_rejected_without_save_or_activation(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("hermes_cli.config.save_env_value") as save, patch.object(
            cli, "activate_all_providers_in_config"
        ) as activate, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.handle_cli(self.setup_args("   "))

        self.assertEqual(raised.exception.code, 1)
        save.assert_not_called()
        activate.assert_not_called()
        self.assertIn("cannot be empty", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
