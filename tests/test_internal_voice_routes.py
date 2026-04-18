import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.internal_voice.provider import SynthesisResult


class InternalVoiceRoutesTest(unittest.TestCase):
    def setUp(self):
        os.environ["VOICE_AUTH_MODE"] = "open"
        os.environ["MAX_CHARS_PER_CHUNK"] = "20"
        os.environ["DEFAULT_AUDIO_FORMAT"] = "mp3"
        self.client = TestClient(app)

    @patch("app.internal_voice.provider.OpenAIVoiceProvider.synthesize")
    def test_open_access_no_auth_required(self, mock_synthesize):
        mock_synthesize.return_value = SynthesisResult(ok=True, audio_bytes=b"abc")
        payload = {
            "child_id": "c1",
            "age_band": "7-9",
            "prompt_id": "p1",
            "voice_text": "hello child",
            "voice_pacing": "slow",
            "voice_chunk_id": "chunk-1",
        }

        response = self.client.post("/internal/voice/checkin-prompt", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("app.internal_voice.provider.OpenAIVoiceProvider.synthesize")
    def test_valid_report_section_request(self, mock_synthesize):
        mock_synthesize.return_value = SynthesisResult(ok=True, audio_bytes=b"xyz")
        payload = {
            "child_id": "c2",
            "section_key": "focus",
            "voice_text": "Strong attention and consistent effort.",
            "voice_chunk_id": "2",
        }

        response = self.client.post("/internal/voice/report-section", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "openai")
        self.assertEqual(body["chunk_index"], 2)
        self.assertIn("asset_ref", body)

    def test_invalid_request_rejection(self):
        payload = {
            "child_id": "c2",
            "section_key": "focus",
            "voice_text": "   ",
            "voice_chunk_id": "2",
        }

        response = self.client.post("/internal/voice/report-section", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "invalid_request")

    @patch("app.internal_voice.provider.OpenAIVoiceProvider.synthesize")
    def test_provider_failure_fallback(self, mock_synthesize):
        mock_synthesize.return_value = SynthesisResult(ok=False, error="timeout")
        payload = {
            "child_id": "c1",
            "age_band": "7-9",
            "prompt_id": "p1",
            "voice_text": "hello child",
            "voice_pacing": "slow",
            "voice_chunk_id": "chunk-1",
        }

        response = self.client.post("/internal/voice/checkin-prompt", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "provider_unavailable")

    @patch("app.internal_voice.provider.OpenAIVoiceProvider.synthesize")
    def test_chunk_handling(self, mock_synthesize):
        mock_synthesize.return_value = SynthesisResult(ok=True, audio_bytes=b"chunk")
        long_text = "word " * 20
        payload = {
            "child_id": "c1",
            "age_band": "7-9",
            "prompt_id": "p1",
            "voice_text": long_text,
            "voice_pacing": "slow",
            "voice_chunk_id": "chunk-5",
        }

        response = self.client.post("/internal/voice/checkin-prompt", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertGreater(mock_synthesize.call_count, 1)
        self.assertEqual(response.json()["chunk_index"], 5)

    @patch("app.internal_voice.provider.OpenAIVoiceProvider.synthesize")
    def test_response_shape_consistency(self, mock_synthesize):
        mock_synthesize.return_value = SynthesisResult(ok=True, audio_bytes=b"ok")
        payload = {
            "child_id": "c1",
            "age_band": "7-9",
            "prompt_id": "p1",
            "voice_text": "hello",
            "voice_pacing": "slow",
            "voice_chunk_id": "1",
        }

        response = self.client.post("/internal/voice/checkin-prompt", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], ["ok", "fallback", "provider_unavailable", "invalid_request"])
        self.assertIn("provider", body)
        self.assertIn("playable_text", body)
        self.assertTrue("audio_url" in body or "asset_ref" in body)
        self.assertIn("chunk_index", body)
        self.assertIn("expires_at", body)


if __name__ == "__main__":
    unittest.main()
