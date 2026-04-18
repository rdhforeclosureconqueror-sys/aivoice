from fastapi import Header, HTTPException

from .config import get_voice_gateway_settings


def enforce_voice_access(x_internal_token: str | None = Header(default=None)) -> None:
    settings = get_voice_gateway_settings()
    mode = settings.voice_auth_mode

    if mode == "open":
        return

    expected = settings.internal_voice_token
    if mode in {"internal", "strict"}:
        if not expected:
            raise HTTPException(status_code=503, detail="internal voice auth is not configured")
        if x_internal_token != expected:
            raise HTTPException(status_code=401, detail="invalid internal token")
        return

    raise HTTPException(status_code=500, detail=f"Unsupported VOICE_AUTH_MODE: {mode}")
