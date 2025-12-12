import io
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import your OpenVoice wrapper
import openvoice_app

app = FastAPI(title="aiVoice (OpenVoice API)")

class SpeakRequest(BaseModel):
    text: str
    speaker: str | None = None  # demo-speaker-0, etc
    speed: float | None = 1.0

# ---- lazy-loaded singleton ----
_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # Try common patterns. We’ll finalize once we see your openvoice_app.py structure.
    if hasattr(openvoice_app, "get_engine"):
        _ENGINE = openvoice_app.get_engine()
    elif hasattr(openvoice_app, "OpenVoiceApp"):
        _ENGINE = openvoice_app.OpenVoiceApp()
    elif hasattr(openvoice_app, "engine"):
        _ENGINE = openvoice_app.engine
    else:
        raise RuntimeError("Could not find engine entrypoint in openvoice_app.py")

    return _ENGINE

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/speak")
def speak(req: SpeakRequest):
    try:
        engine = get_engine()

        # Expectation: engine.speak(text, speaker=..., speed=...) -> bytes OR file path
        out = None
        if hasattr(engine, "speak"):
            out = engine.speak(req.text, speaker=req.speaker, speed=req.speed)
        elif hasattr(openvoice_app, "speak"):
            out = openvoice_app.speak(req.text, speaker=req.speaker, speed=req.speed)
        else:
            raise RuntimeError("No speak() method found (engine.speak or openvoice_app.speak).")

        # If you return a filepath, convert to bytes:
        if isinstance(out, str) and os.path.exists(out):
            with open(out, "rb") as f:
                audio_bytes = f.read()
        elif isinstance(out, (bytes, bytearray)):
            audio_bytes = bytes(out)
        else:
            raise RuntimeError("speak() must return bytes or a valid file path.")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
