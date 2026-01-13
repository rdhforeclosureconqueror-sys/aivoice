import io
import os
from typing import Optional, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
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

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://mufasa-knowledge-bank.onrender.com,https://prince-of-pan-africa.onrender.com,https://mufasafitsite.onrender.com"
)

AIVOICE_API_KEY = os.getenv("AIVOICE_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# APP
# ----------------------------
app = FastAPI(title=APP_TITLE)

# ✅ Improved CORS handling
origins = ["*"] if "*" in ALLOWED_ORIGINS else [
    o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ----------------------------
# MODELS
# ----------------------------
class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    format: Optional[Literal["mp3", "wav"]] = "mp3"

# ----------------------------
# HELPERS
# ----------------------------
def _require_client():
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

# ✅ Allow preflight (OPTIONS) without auth
def _require_service_key(request: Request):
    if request.method == "OPTIONS":
        return
    if AIVOICE_API_KEY and request.headers.get("X-AIVOICE-KEY") != AIVOICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-AIVOICE-KEY")

def _mime(fmt: str):
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "aivoice",
        "endpoints": ["/health", "/speak", "/tts", "/stt", "/whisper"],
    }

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_TTS_MODEL}

# ✅ Properly handle browser preflight CORS requests
@app.options("/speak")
async def speak_options(request: Request):
    response = Response(status_code=200)
    response.headers["Access-Control-Allow-Origin"] = ",".join(origins)
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-AIVOICE-KEY"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

@app.post("/speak")
def speak(req: SpeakRequest, request: Request):
    _require_service_key(request)
    _require_client()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    voice = req.voice or OPENAI_TTS_VOICE
    fmt = req.format or "mp3"

    try:
        audio = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text
        )

        audio_bytes = audio.read()

        response = StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=_mime(fmt),
            headers={"Cache-Control": "no-store"},
        )
        response.headers["Access-Control-Allow-Origin"] = ",".join(origins)
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-AIVOICE-KEY"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

# Alias for /tts
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
        return {"text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

# Alias
@app.post("/whisper")
async def whisper(file: UploadFile = File(...), request: Request = None):
    return await stt(file, request)
