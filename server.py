# server.py
# Render entrypoint for AIVOICE (OpenAI TTS + optional proxy mode).
# This intentionally does NOT import OpenVoice so the service boots reliably.

from fastapi import FastAPI
from app.main import app as aivoice_app  # <-- your working FastAPI app is here (app/main.py)

app = FastAPI(title="aiVoice (OpenAI TTS/STT Gateway)", version="1.0.0")

# Mount your existing app (it already has /health and /speak)
app.mount("/", aivoice_app)

# Optional: a simple root endpoint (doesn't break anything if main.py already has "/")
@app.get("/")
def root():
    return {"ok": True, "service": "aivoice", "features": ["tts(/speak)"], "openvoice": False}
