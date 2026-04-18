from typing import Literal

from pydantic import BaseModel, Field


ResponseStatus = Literal["ok", "fallback", "provider_unavailable", "invalid_request"]


class VoiceGatewayResponse(BaseModel):
    status: ResponseStatus
    provider: str
    playable_text: str
    audio_url: str | None = None
    asset_ref: str | None = None
    chunk_index: int
    expires_at: str | None = None


class CheckinPromptRequest(BaseModel):
    child_id: str = Field(min_length=1)
    age_band: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    voice_text: str = Field(min_length=1)
    voice_pacing: str | None = None
    voice_chunk_id: str = Field(min_length=1)


class ReportSectionRequest(BaseModel):
    child_id: str = Field(min_length=1)
    section_key: str = Field(min_length=1)
    voice_text: str = Field(min_length=1)
    voice_chunk_id: str = Field(min_length=1)
