import io
import logging
import os
from typing import Optional, Literal, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from app.internal_voice import router as internal_voice_router

# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "aiVoice (OpenAI TTS + Whisper STT)"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
VOICE_AUTH_MODE = os.getenv("VOICE_AUTH_MODE", "open").strip().lower() or "open"
INTERNAL_VOICE_TOKEN = os.getenv("INTERNAL_VOICE_TOKEN", "").strip()
OPENVOICE_UPSTREAM_URL = os.getenv("OPENVOICE_UPSTREAM_URL", "").strip()

# IMPORTANT: Origins must be EXACT matches to the browser's Origin header:
# like "https://mufasafitsite.onrender.com" (no trailing slash, no path)
DEFAULT_ALLOWED_ORIGINS = [
    "https://mufasa-knowledge-bank.onrender.com",
    "https://prince-of-pan-africa.onrender.com",
    "https://mufasafitsite.onrender.com",
    "https://simbawaujamaa.com",
    "https://www.simbawaujamaa.com",
    # dev helpers (safe to keep):
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
AIVOICE_API_KEY = os.getenv("AIVOICE_API_KEY", "").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
logger = logging.getLogger(__name__)

# ----------------------------
# HELPERS
# ----------------------------
def normalize_origin(o: str) -> str:
    return (o or "").strip().rstrip("/")  # strip spaces + trailing slash


def build_origins() -> List[str]:
    env_list = []
    if ALLOWED_ORIGINS_ENV:
        env_list = [normalize_origin(x) for x in ALLOWED_ORIGINS_ENV.split(",") if normalize_origin(x)]

    defaults = [normalize_origin(x) for x in DEFAULT_ALLOWED_ORIGINS if normalize_origin(x)]

    # If they set ALLOWED_ORIGINS="*" then allow all
    if "*" in env_list:
        return ["*"]

    # Merge unique
    merged = []
    for o in defaults + env_list:
        if o not in merged:
            merged.append(o)

    return merged


def _safe_log(event: str, **extra):
    logger.warning(event, extra={"auth_reason": event, **extra})


def _require_client():
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")


def _configured_token() -> tuple[str, str]:
    """Return the active local service-token value and expected header name.

    INTERNAL_VOICE_TOKEN is the preferred contract for service-to-service calls.
    AIVOICE_API_KEY / X-AIVOICE-KEY remains supported for existing callers.
    """
    if INTERNAL_VOICE_TOKEN:
        return INTERNAL_VOICE_TOKEN, "x-internal-token"
    if AIVOICE_API_KEY:
        return AIVOICE_API_KEY, "x-aivoice-key"
    return "", ""


def _request_token(request: Request, expected_header: str) -> str | None:
    if expected_header == "x-internal-token":
        return request.headers.get("x-internal-token")
    if expected_header == "x-aivoice-key":
        return request.headers.get("x-aivoice-key")
    return None


def _require_service_key(request: Request):
    # DO NOT block preflight
    if request.method == "OPTIONS":
        return

    expected_token, expected_header = _configured_token()

    if VOICE_AUTH_MODE in {"open", ""} and not expected_token:
        return

    if VOICE_AUTH_MODE not in {"open", "internal", "strict"}:
        _safe_log("local_route_auth_failed", path=request.url.path, mode=VOICE_AUTH_MODE)
        raise HTTPException(status_code=500, detail="local_route_auth_failed")

    if not expected_token:
        _safe_log("missing_internal_token", path=request.url.path, mode=VOICE_AUTH_MODE)
        raise HTTPException(status_code=503, detail="missing_internal_token")

    provided_token = _request_token(request, expected_header)
    if not provided_token:
        _safe_log("missing_internal_token", path=request.url.path, expected_header=expected_header)
        raise HTTPException(status_code=401, detail="missing_internal_token")

    if provided_token != expected_token:
        _safe_log("invalid_internal_token", path=request.url.path, expected_header=expected_header)
        raise HTTPException(status_code=401, detail="invalid_internal_token")


def _mime(fmt: str):
    fmt = (fmt or "mp3").lower().strip()
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"


def _is_openai_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 401

# ----------------------------
# APP
# ----------------------------
app = FastAPI(title=APP_TITLE)

origins = build_origins()

# NOTE:
# - If origins == ["*"] then allow_credentials MUST be False (browser rule).
# - We explicitly allow both current and legacy auth headers + Content-Type.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-internal-token", "X-AIVOICE-KEY"],
    expose_headers=["Content-Type", "Content-Length"],
    max_age=86400,
)

# ----------------------------
# MODELS
# ----------------------------
class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    format: Optional[Literal["mp3", "wav"]] = "mp3"
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 0.0

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    _, expected_header = _configured_token()
    return {
        "ok": True,
        "service": "aivoice",
        "model": OPENAI_TTS_MODEL,
        "allowed_origins": origins,  # helpful
        "voice_auth_mode": VOICE_AUTH_MODE,
        "auth_required": bool(_configured_token()[0]) or VOICE_AUTH_MODE in {"internal", "strict"},
        "expected_auth_header": expected_header or None,
        "openvoice_upstream_configured": bool(OPENVOICE_UPSTREAM_URL),
        "endpoints": ["/health", "/cors-debug", "/speak", "/tts", "/stt", "/whisper"],
    }

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_TTS_MODEL}

# ✅ THIS tells you the exact Origin the browser is sending + what server allows
@app.get("/cors-debug")
def cors_debug(request: Request):
    _, expected_header = _configured_token()
    return {
        "origin_header": request.headers.get("origin"),
        "access_control_request_method": request.headers.get("access-control-request-method"),
        "access_control_request_headers": request.headers.get("access-control-request-headers"),
        "server_allowed_origins": origins,
        "server_expect_header_key": expected_header or None,
        "voice_auth_mode": VOICE_AUTH_MODE,
        "auth_required": bool(_configured_token()[0]) or VOICE_AUTH_MODE in {"internal", "strict"},
        "openvoice_upstream_configured": bool(OPENVOICE_UPSTREAM_URL),
    }

@app.post("/speak")
def speak(req: SpeakRequest, request: Request):
    _require_service_key(request)
    _require_client()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    voice = (req.voice or OPENAI_TTS_VOICE).strip()
    fmt = (req.format or "mp3").strip().lower()

    try:
        audio = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text
        )

        audio_bytes = audio.read()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=_mime(fmt),
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        if _is_openai_auth_error(e):
            _safe_log("openai_auth_failed", path=request.url.path)
            raise HTTPException(status_code=502, detail="openai_auth_failed")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

# Alias for /tts (keep identical behavior)
@app.post("/tts")
def tts(req: SpeakRequest, request: Request):
    return speak(req, request)

@app.post("/stt")
async def stt(file: UploadFile = File(...), request: Request = None):
    if request:
        _require_service_key(request)
    _require_client()

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    f = io.BytesIO(audio_bytes)
    f.name = file.filename or "audio.wav"

    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
        return JSONResponse({"text": result.text})
    except Exception as e:
        if _is_openai_auth_error(e):
            _safe_log("openai_auth_failed", path=request.url.path)
            raise HTTPException(status_code=502, detail="openai_auth_failed")
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

@app.post("/whisper")
async def whisper(file: UploadFile = File(...), request: Request = None):
    return await stt(file, request)



app.include_router(internal_voice_router)
