import logging
from datetime import datetime, timedelta, timezone

from .chunking import split_text_into_chunks
from .config import get_voice_gateway_settings
from .models import VoiceGatewayResponse
from .provider import OpenAIVoiceProvider, build_asset_ref


logger = logging.getLogger(__name__)


class VoiceGatewayService:
    def __init__(self):
        settings = get_voice_gateway_settings()
        self.settings = settings
        self.provider = OpenAIVoiceProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_tts_model,
            timeout_ms=settings.voice_timeout_ms,
        )

    @staticmethod
    def parse_chunk_index(voice_chunk_id: str) -> int:
        digits = "".join(c for c in (voice_chunk_id or "") if c.isdigit())
        return int(digits) if digits else 0

    def generate_voice_response(self, voice_text: str, voice_chunk_id: str) -> VoiceGatewayResponse:
        chunk_index = self.parse_chunk_index(voice_chunk_id)
        trimmed_text = (voice_text or "").strip()

        if not trimmed_text:
            return VoiceGatewayResponse(
                status="invalid_request",
                provider=self.settings.default_voice_provider,
                playable_text="",
                chunk_index=chunk_index,
            )

        chunks = split_text_into_chunks(trimmed_text, self.settings.max_chars_per_chunk)
        if not chunks:
            return VoiceGatewayResponse(
                status="invalid_request",
                provider=self.settings.default_voice_provider,
                playable_text="",
                chunk_index=chunk_index,
            )

        logger.info("voice_chunk_count", extra={"count": len(chunks)})

        audio_parts: list[bytes] = []
        for part in chunks:
            result = self.provider.synthesize(
                text=part,
                voice=self.settings.default_voice,
                audio_format=self.settings.default_audio_format,
            )
            if not result.ok or not result.audio_bytes:
                logger.warning("voice_provider_unavailable", extra={"error": result.error})
                return VoiceGatewayResponse(
                    status="provider_unavailable",
                    provider=self.settings.default_voice_provider,
                    playable_text=trimmed_text,
                    chunk_index=chunk_index,
                )
            audio_parts.append(result.audio_bytes)

        audio_bytes = b"".join(audio_parts)
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()

        logger.info("voice_provider_success", extra={"provider": self.settings.default_voice_provider})

        return VoiceGatewayResponse(
            status="ok",
            provider=self.settings.default_voice_provider,
            playable_text=trimmed_text,
            asset_ref=build_asset_ref(audio_bytes, self.settings.default_audio_format),
            chunk_index=chunk_index,
            expires_at=expires_at,
        )
