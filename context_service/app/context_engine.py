"""Context Engine — periodically refreshes Teacher insights and builds system prompts."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .data_source import SensorSimulator
from .teacher_client import TeacherInsight, ask_teacher

logger = logging.getLogger(__name__)

STUDENT_SYSTEM_PROMPT_TEMPLATE = """\
Jesteś Wilgą — inteligentnym asystentem domowym. Odpowiadasz krótko, konkretnie i po polsku.

Poniżej znajduje się aktualna analiza stanu mieszkania przygotowana przez system nadrzędny \
na podstawie danych z czujników ({timestamp}):

---
{insight}
---

Wykorzystaj tę wiedzę, aby odpowiadać na pytania użytkownika o dom, temperaturę, \
zużycie energii, otwarte okna, obecność osób itp. \
Jeśli pytanie nie dotyczy domu — odpowiedz normalnie, ale nie wymyślaj danych których nie masz."""


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
    def system_prompt(self) -> Optional[str]:
        if self._latest_insight is None:
            return None
        return STUDENT_SYSTEM_PROMPT_TEMPLATE.format(
            timestamp=self._latest_insight.source_timestamp,
            insight=self._latest_insight.summary,
        )

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
        self._ready.set()
        logger.info(
            "Context refreshed (timestamp=%s, insight=%d chars)",
            insight.source_timestamp,
            len(insight.summary),
        )
