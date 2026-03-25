"""
voice.py — Speech I/O for AutoResearch AI.

STT : faster-whisper (base model, CPU, int8) — free, no API key, runs locally.
TTS : edge-tts (Microsoft Neural voices)     — free, no API key, streams via HTTPS.

Both modules are lazy-loaded so the main app starts fast even if the voice
dependencies haven't been installed. Any function will raise ImportError with
a clear install message if the dep is missing — the rest of the app is unaffected.

Exports:
    transcribe(audio_bytes)            : bytes → transcript str
    speak_sync(text, output_path, voice): str → mp3 file path
    VOICE_OPTIONS                      : display_name → edge-tts voice id dict
"""

import asyncio
import os
import tempfile

# ─────────────────────────────────────────────
# WHISPER — lazy singleton
# ─────────────────────────────────────────────
_whisper_model = None


def _get_whisper():
    """Load WhisperModel once and reuse across calls (avoids reloading weights)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run:  pip install faster-whisper"
            )
        # base model — ~150MB, good accuracy, fast on CPU with int8 quantisation
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_bytes: bytes) -> str:
    """
    Convert raw audio bytes (wav / webm / mp3) to a transcript string.

    Writes to a temp file because faster-whisper expects a file path, not bytes.
    Temp file is deleted immediately after transcription regardless of outcome.

    Args:
        audio_bytes: Raw audio bytes from st.audio_input().getvalue()

    Returns:
        Transcript string. Empty string if audio was silent or too short.
    """
    model = _get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# EDGE-TTS — async under the hood, sync surface
# ─────────────────────────────────────────────

async def _speak_async(text: str, output_path: str, voice: str) -> None:
    """Internal coroutine — wraps edge_tts.Communicate and saves to file."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is not installed. "
            "Run:  pip install edge-tts"
        )
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(output_path)


def speak_sync(
    text: str,
    output_path: str = "voice_response.mp3",
    voice: str = "en-US-GuyNeural",
) -> str:
    """
    Convert text to speech and save as an mp3 file.

    Uses Microsoft Neural voices via edge-tts — free, no API key required.
    Runs the async coroutine synchronously so callers don't need to manage
    an event loop.

    Args:
        text        : Text to speak. Should be plain sentences — no markdown.
        output_path : Where to save the mp3. Defaults to voice_response.mp3 in cwd.
        voice       : edge-tts voice ID. See VOICE_OPTIONS for choices.

    Returns:
        output_path (for passing directly to st.audio)
    """
    asyncio.run(_speak_async(text, output_path, voice))
    return output_path


# ─────────────────────────────────────────────
# CONVERSATIONAL PHRASES
# ─────────────────────────────────────────────

GREETING_TEXT = (
    "Hey there! Welcome to AutoResearch AI. "
    "I can research any brand, compare competitors, or decode a landing page for you. "
    "Just say a brand URL or name and I'll get started."
)


def speak_greeting(output_path: str = "voice_greeting.mp3", voice: str = "en-US-GuyNeural") -> str:
    """Generate and save the welcome greeting mp3. Returns the file path."""
    return speak_sync(GREETING_TEXT, output_path=output_path, voice=voice)


def speak_acknowledgment(brand: str, output_path: str = "voice_ack.mp3",
                         voice: str = "en-US-GuyNeural") -> str:
    """Generate a conversational acknowledgment before research starts."""
    text = f"Sure! I'll go and research {brand} for you right now. Give me a moment."
    return speak_sync(text, output_path=output_path, voice=voice)


# ─────────────────────────────────────────────
# AVAILABLE VOICES
# ─────────────────────────────────────────────
VOICE_OPTIONS: dict[str, str] = {
    "Guy (US Male)":     "en-US-GuyNeural",
    "Aria (US Female)":  "en-US-AriaNeural",
    "Ryan (UK Male)":    "en-GB-RyanNeural",
    "Sonia (UK Female)": "en-GB-SoniaNeural",
}


# ─────────────────────────────────────────────
# SARVAM AI — Official SDK (STT + TTS)
# pip install sarvamai>=0.1.27
# ─────────────────────────────────────────────

# Module-level singleton — created once per process.
# Streamlit-safe: module globals persist across reruns within the same session.
# Re-reads SARVAM_API_KEY from env each time so secrets injected by app.py are visible.
_sarvam_client_instance = None


def _sarvam_client():
    """Return the shared SarvamAI client, creating it once per process."""
    global _sarvam_client_instance
    if _sarvam_client_instance is not None:
        return _sarvam_client_instance
    try:
        from sarvamai import SarvamAI
    except ImportError:
        raise ImportError("sarvamai is not installed. Run: pip install sarvamai")
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment.")
    _sarvam_client_instance = SarvamAI(api_subscription_key=api_key)
    return _sarvam_client_instance


def _detect_audio_mime(audio_bytes: bytes) -> tuple[str, str]:
    """
    Detect audio format from magic bytes and return (filename, mime_type) to use
    for the Sarvam multipart upload.

    Sarvam supported formats: WAV, MP3, OGG, FLAC, AAC.
    WebM (Chrome/Streamlit Cloud default) is NOT in the list — mapped to OGG
    because both use the Opus codec and Sarvam accepts the stream correctly.
    MP4/AAC (Safari) mapped to AAC.

    Returns: (filename string, MIME type string)
    """
    if audio_bytes[:4] == b"RIFF":
        return "audio.wav", "audio/wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "audio.mp3", "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio.ogg", "audio/ogg"
    if audio_bytes[:4] == b"fLaC":
        return "audio.flac", "audio/flac"
    if audio_bytes[4:8] == b"ftyp" or audio_bytes[:4] == b"\x00\x00\x00\x18":
        return "audio.aac", "audio/aac"
    # WebM EBML magic bytes — remap to OGG (same Opus codec, Sarvam accepts it)
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "audio.ogg", "audio/ogg"
    # Unknown — default to WAV and hope for the best
    return "audio.wav", "audio/wav"


def transcribe_sarvam(audio_bytes: bytes) -> str:
    """
    Speech-to-text via Sarvam Saaras v3 REST API.

    Uses requests directly (not the SDK) so we control the filename and MIME type
    in the multipart upload. Critical for Streamlit Cloud where st.audio_input()
    returns webm/opus (Chrome) — we remap it to audio/ogg (same codec, Sarvam accepts it).

    Returns transcript string. Returns "" on any failure — never raises.
    """
    import requests as _requests
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        print("[sarvam_stt] SARVAM_API_KEY not set.")
        return ""

    filename, mime = _detect_audio_mime(audio_bytes)
    print(f"[sarvam_stt] Detected format: {mime} → sending as {filename}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        with open(tmp_path, "rb") as audio_file:
            resp = _requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": api_key},
                files={"file": (filename, audio_file, mime)},
                data={"model": "saaras:v3", "mode": "transcribe"},
                timeout=30,
            )
        if not resp.ok:
            print(f"[sarvam_stt] API error {resp.status_code}: {resp.text}")
            return ""
        transcript = resp.json().get("transcript", "")
        print(f"[sarvam_stt] Transcript: '{transcript}'")
        return transcript
    except Exception as e:
        print(f"[sarvam_stt] Error: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def speak_sarvam(text: str) -> bytes | None:
    """
    Text-to-speech via Sarvam Bulbul v2 SDK.

    Speaker: anushka (female, English — default for bulbul:v2).
    Returns raw WAV bytes ready to pass to st.audio().
    Returns None on any failure — caller must handle gracefully.
    Caps at 500 chars (Bulbul limit).
    """
    import base64
    text = text[:500].strip()
    if not text:
        return None
    try:
        client = _sarvam_client()
        response = client.text_to_speech.convert(
            target_language_code="en-IN",
            speaker="anushka",
            model="bulbul:v2",
            text=text,
        )
        # Docs: join all audio chunks then base64-decode
        audios = getattr(response, "audios", None)
        if audios:
            return base64.b64decode("".join(audios))
        print("[sarvam_tts] No audio in response.")
        return None
    except Exception as e:
        print(f"[sarvam_tts] Error: {e}")
        return None
