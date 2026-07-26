"""Text-to-speech via Microsoft Edge neural voices (edge-tts)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import edge_tts


async def synthesize_speech(
    text: str,
    *,
    voice: str,
    rate: str = "+0%",
) -> bytes:
    """Return MP3 bytes for the given Polish (or other) utterance."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await communicate.save(str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
