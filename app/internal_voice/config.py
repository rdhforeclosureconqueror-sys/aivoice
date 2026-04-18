import os
from dataclasses import dataclass


@dataclass
class VoiceGatewaySettings:
    openai_api_key: str
    default_voice_provider: str
    default_voice: str
    default_audio_format: str
    voice_timeout_ms: int
    max_chars_per_chunk: int
    voice_auth_mode: str
    internal_voice_token: str
    openai_tts_model: str


def get_voice_gateway_settings() -> VoiceGatewaySettings:
    return VoiceGatewaySettings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        default_voice_provider=os.getenv("DEFAULT_VOICE_PROVIDER", "openai").strip() or "openai",
        default_voice=os.getenv("DEFAULT_VOICE", "alloy").strip() or "alloy",
        default_audio_format=os.getenv("DEFAULT_AUDIO_FORMAT", "mp3").strip().lower() or "mp3",
        voice_timeout_ms=int(os.getenv("VOICE_TIMEOUT_MS", "15000")),
        max_chars_per_chunk=int(os.getenv("MAX_CHARS_PER_CHUNK", "800")),
        voice_auth_mode=os.getenv("VOICE_AUTH_MODE", "open").strip().lower() or "open",
        internal_voice_token=os.getenv("INTERNAL_VOICE_TOKEN", "").strip(),
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
    )
