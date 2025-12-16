import io
import os
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI Python SDK (new style)
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
APP_TITLE = "aiVoice (OpenAI TTS + Whisper STT)"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

# Comma-separated list of allowed origins.
# Examples:
#   https://mufasa-real-assistant.onrender.com,https://your-black-history-site.onrender.com
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not OPENAI_API_KEY:
    # Don't crash on import; just fail requests with a clear error.
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title=APP_TITLE)

# ----------------------------
# CORS (critical for browser calls)
# ----------------------------
origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Models
# ----------------------------
class SpeakRequest(BaseModel):
    text: str
    format: Optional[Literal["mp3", "wav"]] = "mp3"
    voice: Optional[str] = None  # override env voice if provided

# ----------------------------
# Helpers
# ----------------------------
def _mime_for(fmt: str) -> str:
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"

def _require_client():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "aivoice", "tts_model": OPENAI_TTS_MODEL}

@app.post("/speak")
def speak(req: SpeakRequest):
    _require_client()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    fmt = req.format or "mp3"
    voice = req.voice or OPENAI_TTS_VOICE

    try:
        # OpenAI TTS -> returns binary audio
        audio = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text,
            format=fmt,   # mp3 or wav
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

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Speech -> Text
    Send multipart/form-data with a file field named 'file'
    """
    _require_client()

    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="empty file")

        # Whisper expects a file-like object with a name
        f = io.BytesIO(audio_bytes)
        f.name = file.filename or "audio.wav"

        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
        return JSONResponse({"text": result.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")
