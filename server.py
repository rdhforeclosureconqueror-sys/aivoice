import io
import os
from typing import Literal, Optional

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

# Comma-separated allowed origins. Use "*" to allow all.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://mufasa-knowledge-bank.onrender.com,https://prince-of-pan-africa.onrender.com"
)

# Optional service-to-service key (recommended)
AIVOICE_API_KEY = os.getenv("AIVOICE_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title=APP_TITLE)

# ----------------------------
# CORS
# ----------------------------
origins = ["*"] if "*" in ALLOWED_ORIGINS else [
    o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# MODELS
# ----------------------------
class SpeakRequest(BaseModel):
    text: str
    format: Optional[Literal["mp3", "wav"]] = "mp3"
    voice: Optional[str] = None  # override env voice if provided

# ----------------------------
# HELPERS
# ----------------------------
def _mime_for(fmt: str) -> str:
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"

def _require_client():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")

def _require_service_key(request: Request):
    """
    If AIVOICE_API_KEY is set, require header:
      X-AIVOICE-KEY: <AIVOICE_API_KEY>
    If not set, no auth is enforced (public).
    """
    if not AIVOICE_API_KEY:
        return
    if request.headers.get("X-AIVOICE-KEY", "") != AIVOICE_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-AIVOICE-KEY")

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "aivoice",
        "endpoints": ["/health", "/speak", "/tts", "/stt", "/whisper"],
        "tts_model": OPENAI_TTS_MODEL,
    }

@app.get("/health")
def health():
    return {"ok": True, "service": "aivoice", "tts_model": OPENAI_TTS_MODEL}

@app.post("/speak")
def speak(req: SpeakRequest, request: Request):
    _require_service_key(request)
    _require_client()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    fmt = (req.format or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="format must be mp3 or wav")

    voice = req.voice or OPENAI_TTS_VOICE

    try:
        audio = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text,
            format=fmt,
        )
        audio_bytes = audio.read()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=_mime_for(fmt),
            headers={
                "Cache-Control": "no-store",
                "X-Voice": voice,
                "X-Format": fmt,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

# Alias endpoint: Knowledge Bank may call /tts
@app.post("/tts")
def tts(req: SpeakRequest, request: Request):
    return speak(req, request)

@app.post("/stt")
async def stt(file: UploadFile = File(...), request: Request = None):
    if request:
        _require_service_key(request)
    _require_client()

    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    f = io.BytesIO(audio_bytes)
    f.name = file.filename or "audio.wav"

    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
        return JSONResponse({"text": result.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

# Alias endpoint: some clients may call /whisper
@app.post("/whisper")
async def whisper(file: UploadFile = File(...), request: Request = None):
    return await stt(file=file, request=request)
