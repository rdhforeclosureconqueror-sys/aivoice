import logging

from fastapi import Header, HTTPException

from .config import get_voice_gateway_settings


logger = logging.getLogger(__name__)


def _log_auth_failure(reason: str, **extra) -> None:
    logger.warning(reason, extra={"auth_reason": reason, **extra})


def enforce_voice_access(x_internal_token: str | None = Header(default=None)) -> None:
    settings = get_voice_gateway_settings()
    mode = settings.voice_auth_mode

    if mode == "open":
        return

    expected = settings.internal_voice_token
    if mode in {"internal", "strict"}:
        if not expected:
            _log_auth_failure("missing_internal_token", mode=mode)
            raise HTTPException(status_code=503, detail="missing_internal_token")
        if not x_internal_token:
            _log_auth_failure("missing_internal_token", mode=mode, expected_header="x-internal-token")
            raise HTTPException(status_code=401, detail="missing_internal_token")
        if x_internal_token != expected:
            _log_auth_failure("invalid_internal_token", mode=mode, expected_header="x-internal-token")
            raise HTTPException(status_code=401, detail="invalid_internal_token")
        return

    _log_auth_failure("local_route_auth_failed", mode=mode)
    raise HTTPException(status_code=500, detail="local_route_auth_failed")
