import importlib
import os
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient


class FakeSpeechResponse:
    def read(self):
        return b"audio"


class ServerAuthTest(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _client(self, **env):
        os.environ.pop("AIVOICE_API_KEY", None)
        os.environ.pop("INTERNAL_VOICE_TOKEN", None)
        os.environ.pop("VOICE_AUTH_MODE", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.update(env)

        import server

        server = importlib.reload(server)
        server.client = Mock()
        server.client.audio.speech.create.return_value = FakeSpeechResponse()
        return TestClient(server.app)

    def test_speak_accepts_internal_token_contract(self):
        client = self._client(
            OPENAI_API_KEY="openai-key",
            VOICE_AUTH_MODE="internal",
            INTERNAL_VOICE_TOKEN="service-token",
        )

        response = client.post(
            "/speak",
            headers={"x-internal-token": "service-token"},
            json={"text": "hello", "voice": "alloy", "format": "mp3", "speed": 0.9, "pitch": 0},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"audio")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")

    def test_speak_rejects_missing_internal_token_with_safe_detail(self):
        client = self._client(
            OPENAI_API_KEY="openai-key",
            VOICE_AUTH_MODE="internal",
            INTERNAL_VOICE_TOKEN="service-token",
        )

        response = client.post("/speak", json={"text": "hello", "format": "mp3"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "missing_internal_token")

    def test_speak_rejects_invalid_internal_token_with_safe_detail(self):
        client = self._client(
            OPENAI_API_KEY="openai-key",
            VOICE_AUTH_MODE="internal",
            INTERNAL_VOICE_TOKEN="service-token",
        )

        response = client.post(
            "/speak",
            headers={"x-internal-token": "wrong-token"},
            json={"text": "hello", "format": "mp3"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "invalid_internal_token")

    def test_speak_keeps_legacy_aivoice_key_contract(self):
        client = self._client(OPENAI_API_KEY="openai-key", AIVOICE_API_KEY="legacy-token")

        response = client.post(
            "/speak",
            headers={"X-AIVOICE-KEY": "legacy-token"},
            json={"text": "hello", "format": "mp3"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"audio")

    def test_cors_debug_reports_expected_header_without_secret(self):
        client = self._client(
            OPENAI_API_KEY="openai-key",
            VOICE_AUTH_MODE="strict",
            INTERNAL_VOICE_TOKEN="service-token",
        )

        response = client.get("/cors-debug")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["server_expect_header_key"], "x-internal-token")
        self.assertTrue(body["auth_required"])
        self.assertNotIn("service-token", response.text)

    def test_openai_401_uses_safe_detail(self):
        client = self._client(OPENAI_API_KEY="openai-key")

        import server

        error = Exception("401 Unauthorized")
        error.status_code = 401
        server.client.audio.speech.create.side_effect = error

        response = client.post("/speak", json={"text": "hello", "format": "mp3"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "openai_auth_failed")


if __name__ == "__main__":
    unittest.main()
