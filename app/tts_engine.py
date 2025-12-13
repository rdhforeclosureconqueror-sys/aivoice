import os

def get_audio_mime(fmt: str) -> str:
    return "audio/mpeg" if fmt == "mp3" else "audio/wav"


def synthesize_audio_bytes(text: str, voice: str | None, fmt: str, speed: float, pitch: float) -> bytes:
    """
    TODO: Replace this stub with OpenVoice inference code.

    This stub lets you deploy the API first (so your other systems can integrate),
    then swap in real synthesis once OpenVoice is working in the Render environment.
    """
    # --- STUB AUDIO (silence) ---
    # Returns a tiny valid WAV header w/ short silence so downstream systems can be tested.
    if fmt == "wav":
        return _tiny_silence_wav()

    # For mp3, easiest is: generate wav and convert with ffmpeg,
    # but Render needs ffmpeg installed. We'll keep stub simple:
    # return wav even if requested mp3, until real engine is wired.
    return _tiny_silence_wav()


def _tiny_silence_wav() -> bytes:
    # 0.1s mono silence, 16-bit PCM, 22050Hz
    import wave, io
    sample_rate = 22050
    duration_s = 0.1
    nframes = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()
