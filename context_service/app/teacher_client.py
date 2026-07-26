"""Async client for the Teacher LLM (large model on Mac)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherInsight:
    summary: str
    source_timestamp: str


TEACHER_SYSTEM_PROMPT = (
    "Jesteś analitykiem IoT smart-home. "
    "Otrzymujesz punktowy snapshot odczytów z czujników mieszkania z jednego znacznika czasu. "
    "Nie zakładaj wartości dla brakujących pól i nie mieszaj stanów z innych momentów. "
    "Odpowiadaj WYŁĄCZNIE po polsku (bez innych języków). "
    "Twoim zadaniem jest zwięzłe podsumowanie stanu mieszkania w 3-5 zdaniach. "
    "Skup się na: temperaturach w pomieszczeniach (w tym łazience, jeśli jest w snapshocie), "
    "zużyciu energii, otwartych oknach/drzwiach, obecności osób, jakości powietrza. "
    "Używaj tylko nazw pomieszczeń z snapshota — nie zmyślaj salonu ani innych pokoi. "
    "Jeśli coś jest nietypowe (wysoka temperatura, duże zużycie energii, otwarte okno nocą) — zaznacz to. "
    "Nie powtarzaj wszystkich surowych liczb — interpretuj je."
)


async def ask_teacher(
    sensor_text: str,
    *,
    ollama_url: str,
    model: str,
    source_timestamp: str,
    timeout_seconds: float = 120.0,
) -> TeacherInsight:
    """Send sensor snapshot to Teacher and return condensed insight."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": sensor_text},
        ],
        "stream": False,
        "keep_alive": "30m",
    }

    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()

    body = response.json()
    content = body.get("message", {}).get("content", "")

    if not content:
        logger.warning("Teacher returned empty response, body keys: %s", list(body.keys()))
        content = "[Nauczyciel nie zwrócił analizy]"

    return TeacherInsight(summary=content.strip(), source_timestamp=source_timestamp)
