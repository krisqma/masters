"""Context Engine — periodically refreshes Teacher insights and builds prompts."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .data_source import SensorSimulator
from .teacher_client import TeacherInsight, ask_teacher

logger = logging.getLogger(__name__)

# Immutable Student rules — MUST stay byte-identical across requests so Ollama
# can reuse the KV / prefix cache. Never interpolate timestamps or insights here.
STUDENT_STATIC_PREFIX = """\
Jesteś Wilguś — przyjazny ptasi gospodarz domku gościnnego (wilga = oriole; „Wilguś” to zdrobnienie).
Mówisz o sobie w pierwszej osobie, krótko i ciepło, po polsku; zwykle 1–2 zdania.
Pomagasz gościowi zrozumieć komfort domu (temperatura, wilgotność, ruch, światło) na podstawie danych z czujników.
Nie sterujesz urządzeniami i nie wykonujesz akcji — tylko informujesz i rozmawiasz.
Dane z kontekstu domu cytuj WYŁĄCZNIE, gdy użytkownik pyta o stan domu, pomieszczenie, komfort albo wprost o odczyty.
Nie doklejaj temperatur ani innych odczytów „przy okazji” do odpowiedzi o czymś innym.
Gdy pytają kim jesteś, skąd nazwa, gdzie jesteś albo o co chodzi w domku — odpowiadaj z tej persony, BEZ liczb z czujników.
Gdy wypowiedź jest niejasna, bezsensowna albo wygląda na błąd STT — nie zgaduj intencji sensorowej; krótko poproś o powtórzenie lub odpowiedz w charakterze bez odczytów.
O stanie domu nie zgaduj, nie uśredniaj i nie zmyślaj metryk ani pomieszczeń spoza kontekstu; przy braku danych powiedz to wprost.
Gdy pytają z jakich pomieszczeń składa się mieszkanie — wymień WSZYSTKIE z listy „Pomieszczenia mieszkania” w kontekście, nic nie pomijaj i nic nie dodawaj.
Jeśli dane są sprzeczne lub oznaczone jako stare, zaznacz niepewność.
Uwzględniaj wcześniejsze wiadomości w rozmowie (dopowiedzenia, korektury).
Przy literówkach STT mapuj nazwy na pomieszczenia z listy w kontekście (np. „Sanon” → najbliższa sensowna nazwa z listy), gdy sens jest oczywisty."""

DYNAMIC_CONTEXT_TEMPLATE = """\
Kontekst domu (analiza snapshotu czujników z {timestamp} UTC, przygotowana przez system nadrzędny).
Użyj poniższych danych tylko, gdy pytanie dotyczy stanu domu. W przeciwnym razie ich nie cytuj.
Pomieszczenia mieszkania (pełna lista, źródło: czujniki): {rooms}.
---
{insight}
---"""


def build_dynamic_context(*, timestamp: str, insight: str, rooms: list[str]) -> str:
    rooms_text = ", ".join(rooms) if rooms else "brak danych"
    return DYNAMIC_CONTEXT_TEMPLATE.format(
        timestamp=timestamp,
        insight=insight,
        rooms=rooms_text,
    )


class ContextEngine:
    """Manages background Teacher refresh and exposes current context."""

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

        self._latest_insight: Optional[TeacherInsight] = None
        self._latest_rooms: list[str] = []
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._ready = asyncio.Event()

    @property
    def is_ready(self) -> bool:
        return self._latest_insight is not None

    @property
    def simulator_status(self) -> dict:
        sim = self._simulator
        return {
            "cursor": sim._cursor,
            "total_rows": len(sim._rows),
            "loaded": sim._loaded,
        }

    @property
    def latest_insight(self) -> Optional[TeacherInsight]:
        return self._latest_insight

    @property
    def static_prefix(self) -> str:
        return STUDENT_STATIC_PREFIX

    @property
    def dynamic_context(self) -> Optional[str]:
        if self._latest_insight is None:
            return None
        return build_dynamic_context(
            timestamp=self._latest_insight.source_timestamp,
            insight=self._latest_insight.summary,
            rooms=self._latest_rooms,
        )

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
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Context Engine stopped.")

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self._do_refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Context refresh failed, will retry in %ds", self._refresh_interval)
            await asyncio.sleep(self._refresh_interval)

    async def _do_refresh(self) -> None:
        snapshot = self._simulator.next_snapshot()

        if not snapshot.readings:
            logger.warning("No sensor readings at cursor %d, skipping Teacher call.", snapshot.cursor)
            return

        logger.info(
            "Refreshing context — simulated time: %s (row %d/%d)",
            snapshot.timestamp,
            snapshot.cursor,
            snapshot.total_rows,
        )

        sensor_text = snapshot.to_text()

        insight = await ask_teacher(
            sensor_text,
            ollama_url=self._teacher_url,
            model=self._teacher_model,
            source_timestamp=snapshot.timestamp,
        )

        self._latest_insight = insight
        self._latest_rooms = snapshot.apartment_rooms()
        self._ready.set()
        logger.info(
            "Context refreshed (timestamp=%s, rooms=%s, insight=%d chars)",
            insight.source_timestamp,
            ",".join(self._latest_rooms),
            len(insight.summary),
        )
