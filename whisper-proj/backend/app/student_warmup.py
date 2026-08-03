"""Keep Student Ollama model + prefix KV cache warm on the RPi."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import httpx

from .config import FALLBACK_STUDENT_STATIC_PREFIX, Settings

logger = logging.getLogger(__name__)

PING_USER_MESSAGE = "ping"

StudentStatus = Literal["warming", "warm", "error"]


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
    return loaded.startswith(f"{target}") or target.startswith(loaded.split(":")[0])


class StudentKeepWarm:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self._last_ok_at: Optional[float] = None
        self._student_warm = False
        self._student_status: StudentStatus = "warming"
        self._student_error: Optional[str] = None
        self._chat_busy = False
        self._session_warming = False
        self._ollama_lock = asyncio.Lock()

    @property
    def student_warm(self) -> bool:
        return self._student_warm

    @property
    def student_status(self) -> StudentStatus:
        return self._student_status

    @property
    def student_error(self) -> Optional[str]:
        return self._student_error

    @property
    def chat_busy(self) -> bool:
        return self._chat_busy

    def set_chat_busy(self, busy: bool) -> None:
        self._chat_busy = busy

    async def acquire_ollama(self) -> None:
        await self._ollama_lock.acquire()

    def release_ollama(self) -> None:
        if self._ollama_lock.locked():
            self._ollama_lock.release()

    async def start(self) -> None:
        if not self._settings.student_keep_warm_enabled:
            self._student_warm = True
            self._student_status = "warm"
            self._student_error = None
            logger.info("Student keep-warm disabled (STUDENT_KEEP_WARM_ENABLED=false).")
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._student_warm = False
        self._student_status = "warming"
        self._student_error = None
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

    async def warm_for_session(self, *, reason: str = "session_new") -> dict:
        """Mark warming, ping RPi with published static (+ facts), then warm/error."""
        self._session_warming = True
        self._student_warm = False
        self._student_status = "warming"
        self._student_error = None
        try:
            await self._keep_warm_once(reason=reason, force=True, include_facts=True)
            return {
                "student_status": self._student_status,
                "student_warm": self._student_warm,
                "student_error": self._student_error,
            }
        finally:
            self._session_warming = False

    async def _run_loop(self) -> None:
        interval = max(30, self._settings.student_keep_warm_interval_seconds)
        logger.info(
            "Student keep-warm loop started (interval=%ds, model=%s).",
            interval,
            self._settings.ollama_model,
        )
        await self._keep_warm_once(reason="startup", force=True, include_facts=True)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                await self._keep_warm_once(reason="interval", force=False, include_facts=False)

    async def _resolve_context(
        self, client: httpx.AsyncClient
    ) -> tuple[str, Optional[str]]:
        try:
            resp = await client.get(
                f"{self._settings.context_service_url.rstrip('/')}/api/context",
                timeout=2.0,
            )
            if resp.status_code == 200:
                body = resp.json()
                prefix = body.get("static_prefix")
                dynamic = body.get("dynamic_context")
                static = (
                    prefix
                    if isinstance(prefix, str) and prefix.strip()
                    else FALLBACK_STUDENT_STATIC_PREFIX
                )
                dyn = dynamic if isinstance(dynamic, str) and dynamic.strip() else None
                return static, dyn
        except Exception:
            logger.debug("Context Service unavailable for warmup prefix; using fallback.")
        return FALLBACK_STUDENT_STATIC_PREFIX, None

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

    async def _ping(
        self,
        client: httpx.AsyncClient,
        *,
        messages: list[dict[str, str]],
    ) -> float:
        started = time.monotonic()
        resp = await client.post(
            self._settings.ollama_url,
            json={
                "model": self._settings.ollama_model,
                "messages": messages,
                "stream": False,
                "keep_alive": self._settings.ollama_keep_alive,
                "options": {"num_predict": 1},
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

    async def _keep_warm_once(
        self,
        *,
        reason: str,
        force: bool,
        include_facts: bool,
    ) -> None:
        if not force:
            if self._chat_busy:
                logger.info("Student keep-warm (%s): skipped_busy", reason)
                return
            if self._session_warming:
                logger.info("Student keep-warm (%s): skipped_session_warming", reason)
                return

        if not force:
            self._student_status = "warming"
            self._student_error = None

        base_url = ollama_base_url(self._settings.ollama_url)
        try:
            async with self._ollama_lock:
                async with httpx.AsyncClient() as client:
                    static_prefix, dynamic = await self._resolve_context(client)
                    loaded = await self._is_model_loaded(client, base_url)
                    elapsed = await self._ping(
                        client,
                        messages=[
                            {"role": "system", "content": static_prefix},
                            {"role": "user", "content": PING_USER_MESSAGE},
                        ],
                    )
                    facts_elapsed = 0.0
                    if include_facts and dynamic:
                        facts_elapsed = await self._ping(
                            client,
                            messages=[
                                {"role": "system", "content": static_prefix},
                                {
                                    "role": "user",
                                    "content": (
                                        f"{dynamic}\n\n"
                                        "Potwierdź jednym słowem, że masz kontekst domu."
                                    ),
                                },
                            ],
                        )
                    self._student_warm = True
                    self._student_status = "warm"
                    self._student_error = None
                    self._last_ok_at = time.monotonic()
                    state = "reloaded" if not loaded else "loaded"
                    logger.info(
                        "Student keep-warm (%s): %s model=%s elapsed=%.1fs "
                        "facts_ping=%.1fs keep_alive=%s facts_chars=%d",
                        reason,
                        state,
                        self._settings.ollama_model,
                        elapsed,
                        facts_elapsed,
                        self._settings.ollama_keep_alive,
                        len(dynamic) if dynamic else 0,
                    )
        except Exception as error:
            self._student_warm = False
            self._student_status = "error"
            self._student_error = str(error)
            logger.warning(
                "Student keep-warm (%s) failed for %s: %s",
                reason,
                self._settings.ollama_model,
                error,
            )
