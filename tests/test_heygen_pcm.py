"""TTS regression tests: PCM-only speech models (HeyGen) are requested as PCM and wrapped as WAV.

All provider traffic is mocked. No credentials or network access are used.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parent))

providers = importlib.import_module("openrouter-everything.providers")

HEYGEN_MODEL = "heygen/heygen-voice-1-turbo:free"
FISH_MODEL = "fish-audio/drama-3"
# Two seconds of 16-bit mono silence-ish samples at 44.1 kHz.
PCM_44K = b"\x01\x00" * 44100 * 2


def _speech_response(content: bytes, content_type: str = "audio/pcm;rate=44100;channels=1",
                     status: int = 200) -> Mock:
    response = Mock(status_code=status, text="", content=content)
    response.headers = {"Content-Type": content_type}
    return response


class PcmTtsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        patches = [
            patch.object(providers, "resolve_credentials", return_value=("sk-test", "https://openrouter.ai/api/v1")),
            patch.object(providers, "validate_model_for_capability", return_value=(True, None, {})),
            patch.object(providers, "load_plugin_config", return_value={}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _synthesize(self, model: str, output_name: str, response: Mock) -> tuple[str, dict]:
        provider = providers.OpenRouterEverythingTTSProvider()
        with patch("requests.post", return_value=response) as post:
            path = provider.synthesize("Teste de voz.", str(self.dir / output_name), model=model)
        payload = post.call_args.kwargs["json"]
        return path, payload

    def test_heygen_is_requested_as_pcm_and_saved_as_wav(self) -> None:
        path, payload = self._synthesize(HEYGEN_MODEL, "out.mp3", _speech_response(PCM_44K))

        self.assertEqual(payload["response_format"], "pcm")
        self.assertTrue(path.endswith(".wav"), path)
        with wave.open(path, "rb") as w:
            self.assertEqual(w.getframerate(), 44100)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.readframes(w.getnframes()), PCM_44K)

    def test_heygen_wav_uses_the_rate_from_the_header_not_24k(self) -> None:
        response = _speech_response(PCM_44K, "audio/pcm;rate=24000;channels=1")
        path, _ = self._synthesize(HEYGEN_MODEL, "rate.wav", response)
        with wave.open(path, "rb") as w:
            self.assertEqual(w.getframerate(), 24000)

    def test_heygen_defaults_when_header_has_no_rate_or_channels(self) -> None:
        response = _speech_response(PCM_44K, "audio/pcm")
        path, _ = self._synthesize(HEYGEN_MODEL, "default.wav", response)
        with wave.open(path, "rb") as w:
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getnchannels(), 1)

    def test_heygen_trims_a_trailing_partial_frame(self) -> None:
        response = _speech_response(PCM_44K + b"\x07", "audio/pcm;rate=44100;channels=1")
        path, _ = self._synthesize(HEYGEN_MODEL, "odd.wav", response)
        with wave.open(path, "rb") as w:
            self.assertEqual(w.readframes(w.getnframes()), PCM_44K)

    def test_heygen_empty_pcm_is_an_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no PCM audio"):
            self._synthesize(HEYGEN_MODEL, "empty.wav", _speech_response(b""))

    def test_heygen_http_error_still_surfaces(self) -> None:
        response = _speech_response(b'{"error":"HeyGen Voice only supports response_format pcm."}',
                                    "application/json", status=400)
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            self._synthesize(HEYGEN_MODEL, "err.mp3", response)

    def test_other_models_keep_mp3_behaviour(self) -> None:
        mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00fake-mp3-bytes"
        path, payload = self._synthesize(FISH_MODEL, "speech.mp3",
                                         _speech_response(mp3, "audio/mpeg"))
        self.assertEqual(payload["response_format"], "mp3")
        self.assertTrue(path.endswith(".mp3"), path)
        self.assertEqual(Path(path).read_bytes(), mp3)


if __name__ == "__main__":
    unittest.main()
