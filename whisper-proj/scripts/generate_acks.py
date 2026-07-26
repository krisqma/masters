#!/usr/bin/env python3
"""Generate short ack MP3s with edge-tts (same voice as production TTS)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("Install edge-tts: pip install edge-tts", file=sys.stderr)
    raise

VOICE = "pl-PL-ZofiaNeural"
RATE = "+8%"

PHRASES = [
    ("01", "Sekundka."),
    ("02", "Już sprawdzam w danych."),
    ("03", "Hmm, spojrzę."),
    ("04", "Okej, sprawdzam."),
    ("05", "Zaraz wrócę z odpowiedzią."),
    ("06", "Aha, moment."),
    ("07", "Patrzę na kontekst."),
    ("08", "Już się tym zajmuję."),
    ("09", "Daj mi chwilę."),
    ("10", "Sprawdzam."),
    ("11", "O, zaraz zobaczę."),
    ("12", "Jasne, sprawdzam dane."),
    ("13", "Hmm, zaraz powiem."),
    ("14", "Moment, zaglądam."),
    ("15", "Już patrzę."),
    ("16", "Rozumiem, sprawdzam."),
    ("17", "Chwileczkę."),
    ("18", "Zaraz to ogarnę."),
    ("19", "Okej, moment."),
    ("20", "Patrzę w dane domu."),
    ("21", "Już wracam z info."),
    ("22", "Aha, sprawdzę."),
    ("23", "Dobra, zaraz."),
    ("24", "Zaglądam do czujników."),
]


async def synthesize(phrase_id: str, text: str, out_dir: Path) -> None:
    out_path = out_dir / f"{phrase_id}.mp3"
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(str(out_path))
    print(f"OK {out_path.name}: {text}")


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "public" / "acks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for phrase_id, text in PHRASES:
        await synthesize(phrase_id, text, out_dir)

    print(f"Done: {len(PHRASES)} files in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
