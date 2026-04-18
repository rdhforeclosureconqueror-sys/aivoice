import base64
import logging
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    ok: bool
    audio_bytes: bytes | None = None
    error: str | None = None


class OpenAIVoiceProvider:
    provider_name = "openai"

    def __init__(self, api_key: str, model: str, timeout_ms: int):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_ms / 1000

    def synthesize(self, text: str, voice: str, audio_format: str) -> SynthesisResult:
        if not self._api_key:
            return SynthesisResult(ok=False, error="OPENAI_API_KEY missing")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "voice": voice,
            "input": text,
            "format": audio_format,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload)
                response.raise_for_status()
                return SynthesisResult(ok=True, audio_bytes=response.content)
        except Exception as exc:
            logger.warning("openai_tts_failed", extra={"error": str(exc)})
            return SynthesisResult(ok=False, error=str(exc))


def build_asset_ref(audio_bytes: bytes, audio_format: str) -> str:
    mime = "audio/mpeg" if audio_format == "mp3" else "audio/wav"
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
