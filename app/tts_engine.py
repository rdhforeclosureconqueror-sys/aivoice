import os
import tempfile
import subprocess
from pathlib import Path

# NOTE:
# This file implements REAL audio synthesis using OpenAI TTS.
# It can optionally convert output format using ffmpeg if present.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

def get_audio_mime(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt == "mp3":
        return "audio/mpeg"
    if fmt == "wav":
        return "audio/wav"
    return "application/octet-stream"


def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        return True
    except Exception:
        return False


def _convert_with_ffmpeg(in_path: str, out_path: str) -> None:
    """
    Convert audio using ffmpeg. Raises if ffmpeg fails.
    """
    cmd = ["ffmpeg", "-y", "-i", in_path, out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _openai_tts_bytes(text: str, voice: str | None, out_fmt: str) -> bytes:
    """
    Generate speech bytes via OpenAI TTS.
    out_fmt should be "mp3" or "wav".
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    # Use the modern OpenAI client style if installed, otherwise fallback.
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Choose a safe default voice if none provided
        chosen_voice = (voice or "alloy")

        # OpenAI supports common formats; mp3 is safest for web playback.
        # If wav isn't supported in your account/model, we generate mp3 then convert (if ffmpeg exists).
        preferred_format = "mp3" if out_fmt not in ("mp3", "wav") else out_fmt

        # Try direct generation
        resp = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=chosen_voice,
            input=text,
            format=preferred_format,  # mp3 typically works best
        )
        audio_bytes = resp.read()

        # If they asked for wav but we produced mp3, convert if possible
        if out_fmt == "wav" and preferred_format == "mp3":
            if not _ffmpeg_exists():
                # No conversion available; return mp3 bytes anyway
                return audio_bytes
            with tempfile.TemporaryDirectory() as td:
                mp3_path = str(Path(td) / "in.mp3")
                wav_path = str(Path(td) / "out.wav")
                Path(mp3_path).write_bytes(audio_bytes)
                _convert_with_ffmpeg(mp3_path, wav_path)
                return Path(wav_path).read_bytes()

        return audio_bytes

    except ImportError:
        # Older "openai" style (less preferred)
        import openai
        openai.api_key = OPENAI_API_KEY

        chosen_voice = (voice or "alloy")
        # This older interface usually returns mp3-like bytes; handle wav conversion if requested.
        resp = openai.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=chosen_voice,
            input=text,
        )
        audio_bytes = resp.read()

        if out_fmt == "wav":
            if not _ffmpeg_exists():
                return audio_bytes
            with tempfile.TemporaryDirectory() as td:
                mp3_path = str(Path(td) / "in.mp3")
                wav_path = str(Path(td) / "out.wav")
                Path(mp3_path).write_bytes(audio_bytes)
                _convert_with_ffmpeg(mp3_path, wav_path)
                return Path(wav_path).read_bytes()

        return audio_bytes


def synthesize_audio_bytes(
    text: str,
    voice: str | None,
    fmt: str,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> bytes:
    """
    REAL synthesis implementation:
    - Generates speech with OpenAI TTS
    - Returns mp3 or wav bytes (best-effort)
    - speed/pitch params are accepted for future use (OpenVoice), but not applied here
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text is required")

    fmt = (fmt or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        fmt = "mp3"

    return _openai_tts_bytes(text=text, voice=voice, out_fmt=fmt)
