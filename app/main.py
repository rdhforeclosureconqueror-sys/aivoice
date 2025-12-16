import os
import io
import shutil
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# If you want proxy mode:
import httpx

from .tts_engine import synthesize_audio_bytes, get_audio_mime

app = FastAPI(title="OpenVoice FastAPI Wrapper", version="1.0.0")


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None        # optional: "demo-speaker-0", etc.
    format: str | None = "mp3"      # "mp3" or "wav"
    speed: float | None = 1.0       # optional
    pitch: float | None = 0.0       # optional


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "OpenVoice API is live. Try /docs, /ffmpeg, or POST /speak"
    }


@app.head("/")
def root_head():
    # Prevent 405 spam from Render health checks
    return


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ffmpeg")
def ffmpeg_check():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"ffmpeg_found": False, "path": None}

    version_line = subprocess.check_output(
        ["ffmpeg", "-version"], text=True
    ).splitlines()[0]

    return {"ffmpeg_found": True, "path": ffmpeg_path, "version_line": version_line}


@app.post("/speak")
async def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    fmt = (req.format or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="format must be mp3 or wav")

    # ---- MODE SWITCH ----
    # If OPENVOICE_UPSTREAM_URL is set, proxy to it (standardize interface for your other services)
    upstream = os.getenv("OPENVOICE_UPSTREAM_URL", "").strip()

    try:
        if upstream:
            # Proxy mode: forward request to upstream OpenVoice endpoint
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    upstream.rstrip("/") + "/speak",
                    json=req.model_dump(),
                )
                r.raise_for_status()

                # Upstream returns audio bytes directly
                content_type = r.headers.get("content-type") or get_audio_mime(fmt)
                return Response(content=r.content, media_type=content_type)

        # Local mode: run synth inside this service
        audio_bytes = synthesize_audio_bytes(
            text=text,
            voice=req.voice,
            fmt=fmt,
            speed=req.speed or 1.0,
            pitch=req.pitch or 0.0,
        )
        mime = get_audio_mime(fmt)

        # StreamingResponse is safer for larger audio
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=mime)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")
