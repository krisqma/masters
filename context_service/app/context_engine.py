"""Context Engine — published (session) facts + staging background refresh."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data_source import SensorSimulator
from .teacher_client import TeacherInsight, ask_teacher

logger = logging.getLogger(__name__)

# Immutable Student rules — MUST stay byte-identical across requests so Ollama
# can reuse the KV / prefix cache. Never interpolate timestamps or facts here.
STUDENT_STATIC_PREFIX = """\
Jesteś Wilguś — przyjazny ptasi gospodarz domku gościnnego (wilga = oriole; „Wilguś” to zdrobnienie).
Mówisz o sobie w pierwszej osobie, krótko i ciepło, po polsku.
Pomagasz gościowi zrozumieć komfort domu (temperatura, wilgotność, ruch, światło) na podstawie danych z czujników.
Nie sterujesz urządzeniami i nie wykonujesz akcji — tylko informujesz i rozmawiasz.
Gdy pytanie NIE dotyczy stanu domu — zwykle 1–2 zdania, bez liczb z czujników.
Gdy pytanie dotyczy stanu domu, pomieszczenia, komfortu, odczytów albo overview — cytuj liczby i jednostki WYŁĄCZNIE z bloku „Stan mieszkania” w kontekście.
Przy pytaniu o wszystkie pomieszczenia, overview domu albo tę samą metrykę w całym domku — wymień WSZYSTKIE dostępne wartości z faktów dla wszystkich pokoi z listy; nie zastępuj liczb słowami „w normie” / „prawidłowa”.
Sekcja „Findings” (jeśli jest) to opcjonalne ostrzeżenia o anomaliach — możesz je wspomnieć, ale nie zastępują faktów.
Nie doklejaj odczytów „przy okazji” do odpowiedzi o czymś innym.
Gdy pytają kim jesteś, skąd nazwa, gdzie jesteś albo o co chodzi w domku — odpowiadaj z tej persony, BEZ liczb z czujników.
Gdy wypowiedź jest niejasna, bezsensowna albo wygląda na błąd STT — nie zgaduj intencji sensorowej; krótko poproś o powtórzenie lub odpowiedz w charakterze bez odczytów.
O stanie domu nie zgaduj, nie uśredniaj i nie zmyślaj metryk ani pomieszczeń spoza kontekstu; przy braku danych w faktach powiedz to wprost.
Gdy pytają z jakich pomieszczeń składa się mieszkanie — wymień WSZYSTKIE z listy „Pomieszczenia mieszkania” w kontekście, nic nie pomijaj i nic nie dodawaj.
Jeśli dane są sprzeczne lub oznaczone jako stare, zaznacz niepewność.
Uwzględniaj wcześniejsze wiadomości w rozmowie (dopowiedzenia, korektury).
Przy literówkach STT mapuj nazwy na pomieszczenia z listy w kontekście (np. „Sanon” → najbliższa sensowna nazwa z listy), gdy sens jest oczywisty."""


def build_dynamic_context(
    *,
    facts_text: str,
    findings: Optional[str],
) -> str:
    """Facts always; findings only when Teacher reported anomalies."""
    parts = [
        "Użyj poniższych danych tylko, gdy pytanie dotyczy stanu domu. W przeciwnym razie ich nie cytuj.",
        facts_text.rstrip(),
    ]
    if findings and findings.strip():
        parts.append("---")
        parts.append("Findings (anomalie / uwagi systemu nadrzędnego):")
        parts.append(findings.strip())
    return "\n".join(parts)


@dataclass
class ContextSlot:
    facts_text: str
    timestamp: str
    rooms: list[str]
    findings: Optional[str] = None

    @property
    def dynamic_context(self) -> str:
        return build_dynamic_context(
            facts_text=self.facts_text,
            findings=self.findings,
        )

    @property
    def facts_chars(self) -> int:
        return len(self.facts_text)


class ContextEngine:
    """Published slot for Student sessions; staging refreshed in the background."""

    def __init__(
        self,
        *,
        teacher_url: str,
        teacher_model: str,
        sensor_csv_path: Path,
        refresh_interval: int = 300,
    ) -> None:
        self._teacher_url = teacher_url
        self._teacher_model = teacher_model
        self._refresh_interval = refresh_interval
        self._simulator = SensorSimulator(csv_path=sensor_csv_path)

        self._published: Optional[ContextSlot] = None
        self._staging: Optional[ContextSlot] = None
        self._staging_generation: int = 0
        self._published_generation: int = 0
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._published is not None

    @property
    def simulator_status(self) -> dict:
        sim = self._simulator
        return {
            "cursor": sim._cursor,
            "total_rows": len(sim._rows),
            "loaded": sim._loaded,
            "published_generation": self._published_generation,
            "staging_generation": self._staging_generation,
        }

    @property
    def source_timestamp(self) -> Optional[str]:
        return self._published.timestamp if self._published else None

    @property
    def latest_insight(self) -> Optional[TeacherInsight]:
        """API alias: findings only (None when no anomalies)."""
        if self._published is None or self._published.findings is None:
            return None
        return TeacherInsight(
            summary=self._published.findings,
            source_timestamp=self._published.timestamp,
        )

    @property
    def static_prefix(self) -> str:
        return STUDENT_STATIC_PREFIX

    @property
    def dynamic_context(self) -> Optional[str]:
        if self._published is None:
            return None
        return self._published.dynamic_context

    @property
    def system_prompt(self) -> Optional[str]:
        """Backward-compatible combined prompt (static + dynamic)."""
        dynamic = self.dynamic_context
        if dynamic is None:
            return None
        return f"{STUDENT_STATIC_PREFIX}\n\n{dynamic}"

    async def start(self) -> None:
        logger.info(
            "Context Engine starting (teacher=%s, model=%s, refresh=%ds)",
            self._teacher_url,
            self._teacher_model,
            self._refresh_interval,
        )
        # First snapshot becomes both published and staging.
        await self._load_snapshot_into(target="both")
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Context Engine stopped.")

    async def advance(self) -> dict:
        """Publish staging (or load a fresh snapshot) for a new conversation session."""
        async with self._lock:
            if (
                self._staging is not None
                and self._staging_generation > self._published_generation
            ):
                self._published = ContextSlot(
                    facts_text=self._staging.facts_text,
                    timestamp=self._staging.timestamp,
                    rooms=list(self._staging.rooms),
                    findings=self._staging.findings,
                )
                self._published_generation = self._staging_generation
                source = "staging"
            else:
                await self._load_snapshot_into(target="published", hold_lock=False)
                # Keep staging in sync when we had nothing newer.
                if self._staging is None or self._staging_generation < self._published_generation:
                    assert self._published is not None
                    self._staging = ContextSlot(
                        facts_text=self._published.facts_text,
                        timestamp=self._published.timestamp,
                        rooms=list(self._published.rooms),
                        findings=self._published.findings,
                    )
                    self._staging_generation = self._published_generation
                source = "fresh_snapshot"

            assert self._published is not None
            self._ready.set()
            logger.info(
                "Context advanced (source=%s, timestamp=%s, published_facts=%d chars, "
                "staging_facts=%d chars, published_gen=%d, staging_gen=%d)",
                source,
                self._published.timestamp,
                self._published.facts_chars,
                self._staging.facts_chars if self._staging else 0,
                self._published_generation,
                self._staging_generation,
            )
            return {
                "status": "ready",
                "source": source,
                "source_timestamp": self._published.timestamp,
                "published_facts_chars": self._published.facts_chars,
                "staging_facts_chars": self._staging.facts_chars if self._staging else 0,
                "published_generation": self._published_generation,
                "staging_generation": self._staging_generation,
                "insight": self._published.findings,
            }

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval)
            try:
                await self._load_snapshot_into(target="staging")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Staging refresh failed, will retry in %ds",
                    self._refresh_interval,
                )

    async def _load_snapshot_into(
        self,
        *,
        target: str,
        hold_lock: bool = True,
    ) -> None:
        """Load next CSV row + Teacher findings into staging and/or published."""

        async def _run() -> None:
            snapshot = self._simulator.next_snapshot()
            if not snapshot.readings:
                logger.warning(
                    "No sensor readings at cursor %d, skipping load.",
                    snapshot.cursor,
                )
                return

            logger.info(
                "Loading snapshot for %s — simulated time: %s (row %d/%d)",
                target,
                snapshot.timestamp,
                snapshot.cursor,
                snapshot.total_rows,
            )

            facts_text = snapshot.to_student_facts_text()
            rooms = list(snapshot.apartment_rooms or self._simulator.apartment_rooms)
            findings_text: Optional[str] = None
            try:
                insight = await ask_teacher(
                    snapshot.to_text(),
                    ollama_url=self._teacher_url,
                    model=self._teacher_model,
                    source_timestamp=snapshot.timestamp,
                )
                if insight is not None:
                    findings_text = insight.summary
            except Exception:
                logger.exception(
                    "Teacher anomaly analysis failed; keeping facts without findings."
                )

            slot = ContextSlot(
                facts_text=facts_text,
                timestamp=snapshot.timestamp,
                rooms=rooms,
                findings=findings_text,
            )

            if target in ("staging", "both"):
                self._staging = slot
                self._staging_generation += 1
            if target in ("published", "both"):
                self._published = ContextSlot(
                    facts_text=slot.facts_text,
                    timestamp=slot.timestamp,
                    rooms=list(slot.rooms),
                    findings=slot.findings,
                )
                if target == "both":
                    self._published_generation = self._staging_generation
                else:
                    self._published_generation += 1
                self._ready.set()

            logger.info(
                "Snapshot loaded (target=%s, timestamp=%s, facts=%d chars, findings=%s, "
                "published_gen=%d, staging_gen=%d)",
                target,
                snapshot.timestamp,
                len(facts_text),
                f"{len(findings_text)} chars" if findings_text else "none",
                self._published_generation,
                self._staging_generation,
            )

        if hold_lock:
            async with self._lock:
                await _run()
        else:
            # Caller already holds the lock.
            await _run()
