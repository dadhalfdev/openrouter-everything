"""Regression tests for local image references and image-edit model metadata.

All provider traffic is mocked. No credentials or network access are used.
"""

from __future__ import annotations

import base64
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import requests


_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = "_openrouter_everything_reference_safety"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_PLUGIN_ROOT)]
    sys.modules[_PACKAGE] = package
providers = importlib.import_module(f"{_PACKAGE}.providers")

# A real 1x1 PNG, rather than text with a PNG extension or a PNG-looking prefix.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
REMOTE_URL = "https://cdn.example.test/assets/source.png?token=abc%2F123"
BASE_URL = "https://openrouter.example.test/api/v1"


def _entry(input_modalities=None):
    architecture = {"output_modalities": ["image", "text"]}
    if input_modalities is not None:
        architecture["input_modalities"] = input_modalities
    return {
        "id": "test/image-model",
        "name": "Test Image Model",
        "architecture": architecture,
        "endpoints": ["/api/v1/chat/completions"],
    }


def _stub_common(monkeypatch, entry=None):
    """Isolate provider flows from config, credentials, and both catalogs."""
    entry = _entry(["text", "image"]) if entry is None else entry
    monkeypatch.setattr(providers, "resolve_credentials", lambda: ("test-key", BASE_URL))
    monkeypatch.setattr(providers, "build_headers", lambda _key: {"Authorization": "Bearer test-key"})
    monkeypatch.setattr(
        providers,
        "get_configured_model",
        lambda _capability, explicit=None: explicit or "test/image-model",
    )
    monkeypatch.setattr(
        providers,
        "validate_model_for_capability",
        lambda _model, _capability: (True, None, entry),
    )
    monkeypatch.setattr(providers, "fetch_video_catalog", lambda: [])


def _chat_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"images": [{"image_url": {"url": PNG_DATA_URL}}]}}]
    }
    return response


def _generate(kind, **kwargs):
    if kind == "image":
        return providers.OpenRouterEverythingImageProvider().generate("edit this", **kwargs)
    return providers.OpenRouterEverythingVideoProvider().generate("animate this", **kwargs)


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize(
    "relative", ["auth.json", ".env", ".ssh/id_rsa.png", "home/.ssh/id_rsa.png"]
)
def test_protected_credential_paths_never_reach_provider(
    monkeypatch, tmp_path, kind, relative
):
    """Valid image magic in a credential store must not weaken the path guard."""
    hermes_home = tmp_path / ".hermes"
    credential = hermes_home / relative
    credential.parent.mkdir(parents=True, exist_ok=True)
    credential.write_bytes(PNG_BYTES)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, image_url=str(credential))

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    assert "protected credential" in result["error"]
    post.assert_not_called()
    session_type.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_symlink_to_protected_credential_is_rejected(monkeypatch, tmp_path, kind):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    credential = hermes_home / "auth.json"
    credential.write_bytes(PNG_BYTES)
    link = tmp_path / "innocent.png"
    link.symlink_to(credential)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, image_url=str(link))

    assert result["success"] is False
    assert "protected credential" in result["error"]
    post.assert_not_called()
    session_type.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize(
    "reference",
    [
        "missing-local-file",
        "text-with-png-extension",
        "non-image-data-url",
        "invalid-base64-data-url",
        "unsupported-url-scheme",
    ],
)
def test_invalid_references_fail_before_any_generation_request(
    monkeypatch, tmp_path, kind, reference
):
    missing = tmp_path / "missing.png"
    text_file = tmp_path / "fake.png"
    text_file.write_text("not an image", encoding="utf-8")
    refs = {
        "missing-local-file": str(missing),
        "text-with-png-extension": str(text_file),
        "non-image-data-url": "data:image/png;base64,"
        + base64.b64encode(b"not an image").decode("ascii"),
        "invalid-base64-data-url": "data:image/png;base64,%%%not-base64%%%",
        "unsupported-url-scheme": "file:///etc/passwd",
    }
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, image_url=refs[reference])

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    post.assert_not_called()
    session_type.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_one_bad_reference_cannot_be_silently_dropped(monkeypatch, tmp_path, kind):
    """A mixed valid/invalid edit must fail, not degrade to a partial edit."""
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(
        kind,
        reference_image_urls=[REMOTE_URL, str(tmp_path / "missing.png")],
    )

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    post.assert_not_called()
    session_type.assert_not_called()


def test_oversize_local_reference_is_rejected_before_reading(monkeypatch, tmp_path):
    image = tmp_path / "oversize.png"
    with image.open("wb") as handle:
        handle.truncate(providers._MAX_INPUT_IMAGE_BYTES + 1)
    _stub_common(monkeypatch)
    post = MagicMock()
    monkeypatch.setattr(requests, "post", post)

    result = _generate("image", image_url=str(image))

    assert result["success"] is False
    assert "25 MiB" in result["error"]
    post.assert_not_called()


def test_image_flow_inlines_real_local_image_and_preserves_remote_url(monkeypatch, tmp_path):
    local = tmp_path / "source.without-a-png-suffix"
    local.write_bytes(PNG_BYTES)
    _stub_common(monkeypatch)
    post = MagicMock(return_value=_chat_response())
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(providers, "save_b64_image", lambda *_args, **_kwargs: Path("/tmp/out.png"))

    result = _generate("image", image_url=str(local), reference_image_urls=[REMOTE_URL])

    assert result["success"] is True
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert urls == [PNG_DATA_URL, REMOTE_URL]
    assert result["modality"] == "image"


def test_video_flow_inlines_real_local_image_and_preserves_remote_url(monkeypatch, tmp_path):
    local = tmp_path / "source.png"
    local.write_bytes(PNG_BYTES)
    _stub_common(monkeypatch)
    session = MagicMock()
    rejected = MagicMock(status_code=400, text="stop after inspecting payload")
    session.post.return_value = rejected
    session_type = MagicMock(return_value=session)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate("video", image_url=str(local), reference_image_urls=[REMOTE_URL])

    assert result["success"] is False  # deliberate mocked stop; no polling or download
    payload = session.post.call_args.kwargs["json"]
    assert payload["frame_images"][0]["image_url"]["url"] == PNG_DATA_URL
    assert payload["input_references"][0]["image_url"]["url"] == REMOTE_URL
    session.get.assert_not_called()


def test_text_only_model_rejects_edit_before_request(monkeypatch):
    _stub_common(monkeypatch, _entry(["text"]))
    post = MagicMock()
    monkeypatch.setattr(requests, "post", post)

    result = _generate("image", image_url=REMOTE_URL)

    assert result["success"] is False
    assert result["error_type"] == "inadequate_model"
    assert "does not accept image inputs" in result["error"]
    post.assert_not_called()


def test_text_input_model_can_still_generate_without_references(monkeypatch):
    _stub_common(monkeypatch, _entry(["text"]))
    post = MagicMock(return_value=_chat_response())
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(providers, "save_b64_image", lambda *_args, **_kwargs: Path("/tmp/out.png"))

    result = _generate("image")

    assert result["success"] is True
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "edit this"}]


@pytest.mark.parametrize(
    "input_modalities,expected",
    [
        (["text", "image"], {"modalities": ["text", "image"], "max_reference_images": 3}),
        (["text"], {"modalities": ["text"], "max_reference_images": 0}),
        (None, {"modalities": ["text"], "max_reference_images": 0}),
    ],
)
def test_capabilities_use_selected_models_actual_input_metadata(
    monkeypatch, input_modalities, expected
):
    entry = _entry(input_modalities)
    monkeypatch.setattr(providers, "get_configured_model", lambda _capability: entry["id"])
    monkeypatch.setattr(providers, "fetch_full_catalog", lambda: {entry["id"]: entry})

    assert providers.OpenRouterEverythingImageProvider().capabilities() == expected


def test_capabilities_do_not_infer_editing_for_unknown_model(monkeypatch):
    monkeypatch.setattr(providers, "get_configured_model", lambda _capability: "unknown/model")
    monkeypatch.setattr(providers, "fetch_full_catalog", dict)

    assert providers.OpenRouterEverythingImageProvider().capabilities() == {
        "modalities": ["text"],
        "max_reference_images": 0,
    }


@pytest.mark.parametrize("kind", ["image", "video"])
def test_references_over_model_limit_fail_instead_of_being_sliced(monkeypatch, kind):
    _stub_common(monkeypatch)
    load = MagicMock()
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(providers, "_load_image_reference", load)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, reference_image_urls=[REMOTE_URL] * 4)

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    assert "at most 3" in result["error"]
    assert "received 4" in result["error"]
    load.assert_not_called()
    post.assert_not_called()
    session_type.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_blank_reference_cannot_be_silently_dropped(monkeypatch, kind):
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, reference_image_urls=[REMOTE_URL, ""])

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    assert "blank" in result["error"]
    post.assert_not_called()
    session_type.assert_not_called()


def test_oversize_data_url_is_rejected_before_base64_decode(monkeypatch):
    decode = MagicMock()
    monkeypatch.setattr(providers.base64, "b64decode", decode)
    encoded = "A" * (((providers._MAX_INPUT_IMAGE_BYTES + 2) // 3) * 4 + 5)

    with pytest.raises(ValueError, match="25 MiB"):
        providers._load_image_reference("data:image/png;base64," + encoded)

    decode.assert_not_called()


def test_data_url_is_canonicalized_from_content_not_declared_mime():
    jpeg = b"\xff\xd8\xff\xe0" + b"safe-jpeg-body"
    value = "data:image/png;base64," + base64.b64encode(jpeg).decode("ascii")

    converted = providers._load_image_reference(value)

    assert converted.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(converted.split(",", 1)[1]) == jpeg


@pytest.mark.parametrize("kind", ["image", "video"])
def test_hardlink_alias_to_protected_credential_is_rejected(monkeypatch, tmp_path, kind):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    credential = hermes_home / "auth.json"
    credential.write_bytes(PNG_BYTES)
    alias = tmp_path / "innocent.png"
    alias.hardlink_to(credential)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _stub_common(monkeypatch)
    post = MagicMock()
    session_type = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "Session", session_type)

    result = _generate(kind, image_url=str(alias))

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    assert "multiple hard links" in result["error"]
    post.assert_not_called()
    session_type.assert_not_called()


@pytest.mark.parametrize(
    "model_id,endpoints,declared,expected",
    [
        (providers.DEFAULTS["image_gen"], ["/api/v1/chat/completions"], None, 14),
        ("openai/gpt-image-2", ["/api/v1/images"], None, 16),
        ("microsoft/mai-image-2.5", ["/api/v1/images"], None, 1),
        ("new/chat-model", ["/api/v1/chat/completions"], None, 3),
        ("new/image-api-model", ["/api/v1/images"], None, 16),
        ("new/image-api-model", ["/api/v1/images"], 7, 7),
        # Boolean and string metadata are not integer limits.
        ("new/image-api-model", ["/api/v1/images"], True, 16),
        ("new/image-api-model", ["/api/v1/images"], "14", 16),
    ],
)
def test_capabilities_use_model_and_api_specific_reference_limits(
    monkeypatch, model_id, endpoints, declared, expected
):
    entry = _entry(["text", "image"])
    entry["id"] = model_id
    entry["endpoints"] = endpoints
    if declared is not None:
        entry["max_reference_images"] = declared
    monkeypatch.setattr(providers, "get_configured_model", lambda _capability: model_id)
    monkeypatch.setattr(providers, "fetch_full_catalog", lambda: {model_id: entry})

    capabilities = providers.OpenRouterEverythingImageProvider().capabilities()

    assert capabilities == {
        "modalities": ["text", "image"],
        "max_reference_images": expected,
    }


def test_declared_model_limit_is_enforced_without_slicing_valid_references(monkeypatch):
    entry = _entry(["text", "image"])
    entry["max_reference_images"] = 4
    _stub_common(monkeypatch, entry)
    post = MagicMock(return_value=_chat_response())
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(providers, "save_b64_image", lambda *_args, **_kwargs: Path("/tmp/out.png"))

    result = _generate("image", reference_image_urls=[REMOTE_URL] * 4)

    assert result["success"] is True
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 4


def test_missing_image_input_metadata_rejects_edit_before_request(monkeypatch):
    _stub_common(monkeypatch, _entry(None))
    post = MagicMock()
    monkeypatch.setattr(requests, "post", post)

    result = _generate("image", image_url=REMOTE_URL)

    assert result["success"] is False
    assert result["error_type"] == "inadequate_model"
    post.assert_not_called()


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_transcription_rejects_protected_files_and_aliases(
    monkeypatch, tmp_path, alias_kind
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    credential = hermes_home / "auth.json"
    credential.write_bytes(b"not-a-real-credential")
    audio = credential
    if alias_kind != "direct":
        audio = tmp_path / f"{alias_kind}.mp3"
        if alias_kind == "symlink":
            audio.symlink_to(credential)
        else:
            audio.hardlink_to(credential)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _stub_common(monkeypatch)
    post = MagicMock()
    monkeypatch.setattr(httpx, "post", post)

    result = providers.OpenRouterEverythingSTTProvider().transcribe(str(audio))

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    if alias_kind == "hardlink":
        assert "multiple hard links" in result["error"]
    else:
        assert "protected credential" in result["error"]
    post.assert_not_called()


def test_transcription_allows_an_ordinary_local_audio_file(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"offline-test-audio")
    _stub_common(monkeypatch)
    response = MagicMock(status_code=200)
    response.json.return_value = {"text": "offline transcript"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)

    result = providers.OpenRouterEverythingSTTProvider().transcribe(str(audio))

    assert result["success"] is True
    assert result["transcript"] == "offline transcript"
    assert post.call_args.kwargs["files"]["file"][0] == "sample.mp3"


def test_model_picker_treats_malformed_input_metadata_as_unknown(monkeypatch):
    monkeypatch.setattr(providers, "fetch_image_catalog", lambda: [
        {"id": "bad/architecture", "architecture": ["text", "image"]},
        {"id": "bad/modalities", "architecture": {"input_modalities": "text,image"}},
    ])

    rows = providers.OpenRouterEverythingImageProvider().list_models()

    assert [row["input_modalities"] for row in rows] == [[], []]
    assert all("input modalities unavailable" in row["strengths"] for row in rows)
