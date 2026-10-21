"""Keep the Student (gemma3 on the RPi) loaded and its session prefix prefilled.

- Warm-up = the exact session prefix the chat will send ([system, facts]) with
  the shared options and num_predict=1, so the first question only appends.
- No periodic inference: the model stays loaded via keep_alive=-1 on every
  request; a watchdog polls /api/ps (no inference, no KV cache change) and
  re-warms only when the model is really gone (e.g. Ollama/RPi restart) or was
  loaded with different options by someone else.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .config import Settings
from .student_prompt import GENERATION_PROMPT_TOKENS, student_options

logger = logging.getLogger(__name__)

StudentStatus = Literal["warming", "warm", "error"]
# Returns (prefix_messages, on_measured_prefix_tokens callback).
PrefixProvider = Callable[[], Awaitable[Tuple[List[dict], Callable[[int], None]]]]


def ollama_base_url(chat_url: str) -> str:
    """http://host:11434/api/chat -> http://host:11434"""
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
        self._stop: Optional[asyncio.Event] = None
        self._prefix_provider: Optional[PrefixProvider] = None
        self._student_warm = False
        self._student_status: StudentStatus = "warming"
        self._student_error: Optional[str] = None
        self._chat_busy = False
        self._ollama_lock: Optional[asyncio.Lock] = None
        self._ps_failures = 0

    # ----------------------------------------------------------------- state
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

    def _lock(self) -> asyncio.Lock:
        # Created lazily inside the running loop (asyncio objects bind to a loop on py3.9).
        if self._ollama_lock is None:
            self._ollama_lock = asyncio.Lock()
        return self._ollama_lock

    async def acquire_ollama(self) -> None:
        await self._lock().acquire()

    def release_ollama(self) -> None:
        lock = self._lock()
        if lock.locked():
            lock.release()

    # ------------------------------------------------------------- lifecycle
    async def start(self, prefix_provider: PrefixProvider) -> None:
        self._prefix_provider = prefix_provider
        if not self._settings.student_keep_warm_enabled:
            self._set_status("warm")
            logger.info("Student keep-warm disabled (STUDENT_KEEP_WARM_ENABLED=false).")
            return
        if self._task and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._set_status("warming")
        self._task = asyncio.create_task(self._run_loop(), name="student-keep-warm")

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def _set_status(self, status: StudentStatus, error: Optional[str] = None) -> None:
        self._student_status = status
        self._student_warm = status == "warm"
        self._student_error = error

    # --------------------------------------------------------------- warm-up
    async def warm_for_session(self, *, reason: str = "session_new") -> dict:
        """Prefill the current session prefix (status 'warming' while it runs)."""
        self._set_status("warming")
        await self._warm(reason=reason)
        return {
            "student_status": self._student_status,
            "student_warm": self._student_warm,
            "student_error": self._student_error,
        }

    async def _warm(self, *, reason: str) -> None:
        if self._prefix_provider is None:
            return
        started = time.monotonic()
        try:
            async with self._lock():
                messages, on_prefix_tokens = await self._prefix_provider()
                body = await self._post_chat(messages)
                prompt_eval = body.get("prompt_eval_count")
                if isinstance(prompt_eval, int) and prompt_eval > 0:
                    on_prefix_tokens(prompt_eval - GENERATION_PROMPT_TOKENS)
                self._set_status("warm")
                logger.info(
                    "Student warm-up (%s): model=%s elapsed=%.1fs prefix_tokens=%s "
                    "prefill_s=%.1f load_s=%.1f",
                    reason,
                    self._settings.ollama_model,
                    time.monotonic() - started,
                    prompt_eval,
                    (body.get("prompt_eval_duration") or 0) / 1e9,
                    (body.get("load_duration") or 0) / 1e9,
                )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}".strip(": ")
            self._set_status("error", detail)
            logger.warning("Student warm-up (%s) failed: %s", reason, detail)

    async def _post_chat(self, messages: List[dict]) -> dict:
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self._settings.ollama_read_timeout_seconds),
            write=30.0,
            pool=30.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self._settings.ollama_url,
                json={
                    "model": self._settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": self._settings.ollama_keep_alive,
                    "options": student_options(self._settings, num_predict=1),
                },
            )
            resp.raise_for_status()
            return resp.json()

    # -------------------------------------------------------------- watchdog
    async def _loaded_state(self) -> Optional[dict]:
        """Loaded Student entry from /api/ps, {} when not loaded, None when unreachable."""
        base_url = ollama_base_url(self._settings.ollama_url)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/ps")
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
        except Exception as error:
            logger.warning("Student /api/ps check failed: %s", error)
            return None
        for item in payload.get("models") or []:
            if isinstance(item, dict) and _model_matches(
                str(item.get("name") or item.get("model") or ""), self._settings.ollama_model
            ):
                return item
        return {}

    async def _run_loop(self) -> None:
        interval = max(15, self._settings.student_keep_warm_interval_seconds)
        logger.info(
            "Student keep-warm started (watchdog every %ds via /api/ps, model=%s, keep_alive=%s).",
            interval,
            self._settings.ollama_model,
            self._settings.ollama_keep_alive,
        )
        await self._warm(reason="startup")
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass
            if self._chat_busy or self._lock().locked():
                continue
            state = await self._loaded_state()
            if state is None:
                self._ps_failures += 1
                if self._ps_failures >= 3:
                    self._set_status("error", "RPi Ollama nie odpowiada (/api/ps).")
                continue
            self._ps_failures = 0
            ctx = state.get("context_length") if state else None
            if not state:
                logger.warning("Student model not loaded on the RPi; re-warming.")
                self._set_status("warming")
                await self._warm(reason="reload")
            elif ctx and ctx != self._settings.student_num_ctx:
                logger.warning(
                    "Student loaded with num_ctx=%s (expected %s), probably by another client; re-warming.",
                    ctx,
                    self._settings.student_num_ctx,
                )
                self._set_status("warming")
                await self._warm(reason="options_mismatch")
            elif self._student_status == "error":
                await self._warm(reason="recover")
