"""Keep Student Ollama model + system-prefix KV cache warm on the RPi."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .config import FALLBACK_STUDENT_STATIC_PREFIX, Settings

logger = logging.getLogger(__name__)

PING_USER_MESSAGE = "ping"


def ollama_base_url(chat_url: str) -> str:
    """http://host:11434/api/chat → http://host:11434"""
    normalized = chat_url.strip().rstrip("/")
    if normalized.endswith("/api/chat"):
        return normalized[: -len("/api/chat")]
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return normalized


def _model_matches(loaded_name: str, wanted: str) -> bool:
    loaded = loaded_name.strip()
    target = wanted.strip()
    if not loaded or not target:
        return False
    if loaded == target:
        return True
    # Ollama sometimes reports tags with digest suffix or bare family name.
    return loaded.startswith(f"{target}") or target.startswith(loaded.split(":")[0])


class StudentKeepWarm:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self._last_ok_at: Optional[float] = None
        self._student_warm = False

    @property
    def student_warm(self) -> bool:
        return self._student_warm

    async def start(self) -> None:
        if not self._settings.student_keep_warm_enabled:
            logger.info("Student keep-warm disabled (STUDENT_KEEP_WARM_ENABLED=false).")
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="student-keep-warm")

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run_loop(self) -> None:
        interval = max(30, self._settings.student_keep_warm_interval_seconds)
        logger.info(
            "Student keep-warm loop started (interval=%ds, model=%s).",
            interval,
            self._settings.ollama_model,
        )
        # Immediate warmup on boot (prefix-aware).
        await self._keep_warm_once(reason="startup")
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                await self._keep_warm_once(reason="interval")

    async def _resolve_static_prefix(self, client: httpx.AsyncClient) -> str:
        try:
            resp = await client.get(
                f"{self._settings.context_service_url.rstrip('/')}/api/context",
                timeout=2.0,
            )
            if resp.status_code == 200:
                body = resp.json()
                prefix = body.get("static_prefix")
                if isinstance(prefix, str) and prefix.strip():
                    return prefix
        except Exception:
            logger.debug("Context Service unavailable for warmup prefix; using fallback.")
        return FALLBACK_STUDENT_STATIC_PREFIX

    async def _is_model_loaded(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            resp = await client.get(f"{base_url}/api/ps", timeout=5.0)
            if resp.status_code != 200:
                return False
            payload: dict[str, Any] = resp.json()
            models = payload.get("models") or []
            wanted = self._settings.ollama_model
            for item in models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("model") or "")
                if _model_matches(name, wanted):
                    return True
        except Exception as error:
            logger.warning("Student /api/ps check failed: %s", error)
        return False

    async def _ping_with_prefix(
        self,
        client: httpx.AsyncClient,
        *,
        static_prefix: str,
    ) -> float:
        """Return elapsed seconds for a non-streaming prefix-warming chat."""
        started = time.monotonic()
        resp = await client.post(
            self._settings.ollama_url,
            json={
                "model": self._settings.ollama_model,
                "messages": [
                    {"role": "system", "content": static_prefix},
                    {"role": "user", "content": PING_USER_MESSAGE},
                ],
                "stream": False,
                "keep_alive": self._settings.ollama_keep_alive,
                "options": {
                    # Keep the keep-warm generation tiny.
                    "num_predict": 1,
                },
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self._settings.ollama_read_timeout_seconds),
                write=30.0,
                pool=30.0,
            ),
        )
        resp.raise_for_status()
        return time.monotonic() - started

    async def _keep_warm_once(self, *, reason: str) -> None:
        base_url = ollama_base_url(self._settings.ollama_url)
        try:
            async with httpx.AsyncClient() as client:
                static_prefix = await self._resolve_static_prefix(client)
                loaded = await self._is_model_loaded(client, base_url)
                elapsed = await self._ping_with_prefix(client, static_prefix=static_prefix)
                self._student_warm = True
                self._last_ok_at = time.monotonic()
                state = "reloaded" if not loaded else "loaded"
                logger.info(
                    "Student keep-warm (%s): %s model=%s elapsed=%.1fs keep_alive=%s",
                    reason,
                    state,
                    self._settings.ollama_model,
                    elapsed,
                    self._settings.ollama_keep_alive,
                )
        except Exception as error:
            self._student_warm = False
            logger.warning(
                "Student keep-warm (%s) failed for %s: %s",
                reason,
                self._settings.ollama_model,
                error,
            )
