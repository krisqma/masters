"""Async client for the Teacher LLM (large model on Mac)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Exact one-line reply when Teacher finds nothing noteworthy.
NO_ANOMALY_SENTINEL = "BRAK_ANOMALII"


@dataclass(frozen=True)
class TeacherInsight:
    summary: str
    source_timestamp: str


TEACHER_SYSTEM_PROMPT = (
    "Jesteś analitykiem IoT smart-home. "
    "Otrzymujesz punktowy snapshot odczytów z czujników mieszkania z jednego znacznika czasu. "
    "Nie zakładaj wartości dla brakujących pól i nie mieszaj stanów z innych momentów. "
    "Odpowiadaj WYŁĄCZNIE po polsku (bez innych języków). "
    "Twoim JEDYNYM zadaniem jest wykrycie anomalii i ciekawych odchyleń — nie streszczaj całego domu. "
    "Szukaj m.in.: skrajnych temperatur lub wilgotności, otwartych okien/drzwi, dużego zużycia mocy/energii, "
    "obecności w nietypowej porze, słabej jakości powietrza (wysoki VOC), sprzecznych odczytów. "
    "Używaj tylko nazw pomieszczeń i liczb ze snapshota — nic nie zmyślaj. "
    "Gdy znajdziesz anomalie: wypisz je zwięźle (1–4 krótkie punkty lub zdania) z konkretnymi wartościami. "
    "Gdy NIC nietypowego nie ma — odpowiedz DOKŁADNIE jedną linią: "
    f"{NO_ANOMALY_SENTINEL} "
    "(bez innych słów, bez streszczenia, bez zdań w stylu „wszystko w normie” / „prawidłowa”). "
    "Zakaz ogólnych podsumowań stanu mieszkania, gdy nie ma anomalii."
)


def normalize_findings(raw: str) -> Optional[str]:
    """Return findings text, or None when Teacher reports no anomalies."""
    text = (raw or "").strip()
    if not text:
        return None
    # Accept sentinel alone or as the only meaningful line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].upper() == NO_ANOMALY_SENTINEL:
        return None
    if text.upper() == NO_ANOMALY_SENTINEL:
        return None
    # Strip accidental sentinel prefix/suffix mixed with content.
    filtered = [ln for ln in lines if ln.upper() != NO_ANOMALY_SENTINEL]
    if not filtered:
        return None
    return "\n".join(filtered)


async def ask_teacher(
    sensor_text: str,
    *,
    ollama_url: str,
    model: str,
    source_timestamp: str,
    timeout_seconds: float = 120.0,
) -> Optional[TeacherInsight]:
    """Ask Teacher for anomaly findings only. Returns None when none / empty."""
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
        return None

    findings = normalize_findings(content)
    if findings is None:
        logger.info("Teacher: no anomalies (%s).", NO_ANOMALY_SENTINEL)
        return None

    return TeacherInsight(summary=findings, source_timestamp=source_timestamp)
