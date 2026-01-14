import io
import os
from typing import Optional, Literal, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "aiVoice (OpenAI TTS + Whisper STT)"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

# IMPORTANT: Origins must be EXACT matches to the browser's Origin header:
# like "https://mufasafitsite.onrender.com" (no trailing slash, no path)
DEFAULT_ALLOWED_ORIGINS = [
    "https://mufasa-knowledge-bank.onrender.com",
    "https://prince-of-pan-africa.onrender.com",
    "https://mufasafitsite.onrender.com",
    # dev helpers (safe to keep):
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
AIVOICE_API_KEY = os.getenv("AIVOICE_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

def _require_client():
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

def _require_service_key(request: Request):
    # DO NOT block preflight
    if request.method == "OPTIONS":
        return
    if AIVOICE_API_KEY and request.headers.get("X-AIVOICE-KEY") != AIVOICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-AIVOICE-KEY")

def _mime(fmt: str):
    fmt = (fmt or "mp3").lower().strip()
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"

# ----------------------------
# APP
# ----------------------------
app = FastAPI(title=APP_TITLE)

origins = build_origins()

# NOTE:
# - If origins == ["*"] then allow_credentials MUST be False (browser rule).
# - We explicitly allow X-AIVOICE-KEY + Content-Type for your /speak calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-AIVOICE-KEY"],
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

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "aivoice",
        "model": OPENAI_TTS_MODEL,
        "allowed_origins": origins,  # helpful
        "endpoints": ["/health", "/cors-debug", "/speak", "/tts", "/stt", "/whisper"],
    }

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_TTS_MODEL}

# ✅ THIS tells you the exact Origin the browser is sending + what server allows
@app.get("/cors-debug")
def cors_debug(request: Request):
    return {
        "origin_header": request.headers.get("origin"),
        "access_control_request_method": request.headers.get("access-control-request-method"),
        "access_control_request_headers": request.headers.get("access-control-request-headers"),
        "server_allowed_origins": origins,
        "server_expect_header_key": "X-AIVOICE-KEY (optional)",
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
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

@app.post("/whisper")
async def whisper(file: UploadFile = File(...), request: Request = None):
    return await stt(file, request)

