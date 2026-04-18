import logging

from fastapi import APIRouter, Depends

from .auth import enforce_voice_access
from .config import get_voice_gateway_settings
from .models import CheckinPromptRequest, ReportSectionRequest, VoiceGatewayResponse
from .service import VoiceGatewayService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal/voice", tags=["internal-voice"])


@router.post("/checkin-prompt", response_model=VoiceGatewayResponse, dependencies=[Depends(enforce_voice_access)])
def create_checkin_prompt(payload: CheckinPromptRequest):
    logger.info("internal_voice_endpoint_hit", extra={"endpoint": "checkin-prompt", "child_id": payload.child_id})
    service = VoiceGatewayService()
    response = service.generate_voice_response(
        voice_text=payload.voice_text,
        voice_chunk_id=payload.voice_chunk_id,
    )
    if response.status == "provider_unavailable":
        logger.warning("internal_voice_fallback", extra={"endpoint": "checkin-prompt"})
    return response


@router.post("/report-section", response_model=VoiceGatewayResponse, dependencies=[Depends(enforce_voice_access)])
def create_report_section(payload: ReportSectionRequest):
    logger.info("internal_voice_endpoint_hit", extra={"endpoint": "report-section", "child_id": payload.child_id})
    service = VoiceGatewayService()
    response = service.generate_voice_response(
        voice_text=payload.voice_text,
        voice_chunk_id=payload.voice_chunk_id,
    )
    if response.status == "provider_unavailable":
        logger.warning("internal_voice_fallback", extra={"endpoint": "report-section"})
    return response


@router.get("/health", dependencies=[Depends(enforce_voice_access)])
def internal_voice_health():
    logger.info("internal_voice_endpoint_hit", extra={"endpoint": "health"})
    return {
        "status": "ok",
        "service": "internal-voice-gateway",
        "provider": get_voice_gateway_settings().default_voice_provider,
    }


@router.get("/config", dependencies=[Depends(enforce_voice_access)])
def internal_voice_config():
    logger.info("internal_voice_endpoint_hit", extra={"endpoint": "config"})
    settings = get_voice_gateway_settings()
    return {
        "status": "ok",
        "voice_auth_mode": settings.voice_auth_mode,
        "default_voice_provider": settings.default_voice_provider,
        "default_voice": settings.default_voice,
        "default_audio_format": settings.default_audio_format,
        "voice_timeout_ms": settings.voice_timeout_ms,
        "max_chars_per_chunk": settings.max_chars_per_chunk,
    }
