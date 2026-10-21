"""Simulated Wilga user/frontend driving the real Whisper Backend API.

Mirrors whisper-proj/src (App.tsx, chatService.ts, useStudentReady.ts):
- history = last 6 user/assistant messages from *before* the turn; the user
  message stays even when the turn fails, a partial assistant reply stays too,
- chat timeout 90 s (client side abort),
- /api/health polled every 2 s while warming, 30 s while warm; questions are
  gated on the *cached* polled status (like the UI button),
- idle rotation: every 15 s, if idle >= 600 s since the last turn start ->
  POST /api/session/new and clear the conversation.

Usage: python driver.py --scenario scenarios/s1.json --out RUN/driver.jsonl \
           --proxy-log RUN/proxy.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

CHAT_TIMEOUT_S = 90.0  # legacy frontend (before the fix)
FIRST_BYTE_TIMEOUT_S = 200.0  # fixed frontend: time to first byte
INTER_CHUNK_TIMEOUT_S = 30.0  # fixed frontend: max gap between chunks
HISTORY_MAX = 6
POLL_WARMING_S = 2.0
POLL_WARM_S = 30.0
IDLE_CHECK_S = 15.0
BOOT_MESSAGE = "__boot__"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Driver:
    def __init__(self, base: str, out: Path, proxy_log: Optional[Path], idle_rotate_s: float,
                 protocol: str = "legacy"):
        self.base = base.rstrip("/")
        self.out = out
        self.proxy_log = proxy_log
        self.idle_rotate_s = idle_rotate_s
        # "legacy": 90 s total timeout; "fixed": client_session_id + 200 s/30 s timeouts.
        self.protocol = protocol
        self.client_session_id = uuid.uuid4().hex
        self.messages: list[dict[str, str]] = []
        self.session = 0
        self.turn = 0
        self.cached_status = "warming"
        self.last_activity = time.monotonic()
        self.busy = False  # PROCESSING (no idle rotation while a turn runs)
        self.rotating = False
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        self._poll_wakeup: Optional[asyncio.Event] = None  # created inside the loop (py3.9)
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ log
    def log(self, event: str, **fields: Any) -> None:
        rec = {"ts": now_iso(), "event": event, "session": self.session, **fields}
        with self.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        short = {k: v for k, v in fields.items() if k not in ("reply", "history")}
        print(f"[{rec['ts'][11:19]}] {event} s{self.session} {short}", flush=True)

    # --------------------------------------------------------------- health
    async def poll_health(self) -> None:
        while True:
            try:
                resp = await self.client.get(f"{self.base}/api/health")
                body = resp.json()
                status = body.get("student_status", "error")
            except Exception as error:
                status = "error"
                body = {"error": repr(error)}
            if status != self.cached_status:
                self.log("health_change", old=self.cached_status, new=status,
                         student_error=body.get("student_error"))
            self.cached_status = status
            delay = POLL_WARM_S if status == "warm" else POLL_WARMING_S
            self._poll_wakeup.clear()
            try:
                await asyncio.wait_for(self._poll_wakeup.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def refresh_now(self) -> None:
        if self._poll_wakeup is not None:
            self._poll_wakeup.set()

    async def idle_rotation(self) -> None:
        while True:
            await asyncio.sleep(IDLE_CHECK_S)
            if self.cached_status != "warm" or self.rotating or self.busy:
                continue
            if time.monotonic() - self.last_activity >= self.idle_rotate_s:
                await self.new_session("idle")

    async def new_session(self, reason: str) -> None:
        if self.rotating or self.busy:
            return
        self.rotating = True
        self.cached_status = "warming"  # markWarming()
        self.messages = []
        self.session += 1
        self.turn = 0
        self.client_session_id = uuid.uuid4().hex
        started = time.monotonic()
        try:
            resp = await self.client.post(f"{self.base}/api/session/new",
                                          timeout=httpx.Timeout(200.0))
            body = resp.json()
            self.log("session_new", reason=reason, http=resp.status_code,
                     elapsed_s=round(time.monotonic() - started, 3),
                     advanced=body.get("advanced"),
                     source_timestamp=body.get("source_timestamp"))
        except Exception as error:
            self.log("session_new", reason=reason, error=repr(error),
                     elapsed_s=round(time.monotonic() - started, 3))
        finally:
            self.last_activity = time.monotonic()
            self.refresh_now()
            self.rotating = False

    # ----------------------------------------------------------------- chat
    def history(self) -> list[dict[str, str]]:
        cleaned = [m for m in self.messages if m["content"].strip()]
        return cleaned[-HISTORY_MAX:]

    async def wait_ready(self, max_wait_s: float = 600.0) -> float:
        started = time.monotonic()
        while self.cached_status != "warm" or self.rotating:
            if time.monotonic() - started > max_wait_s:
                break
            await asyncio.sleep(0.25)
        return round(time.monotonic() - started, 3)

    async def ask(self, text: str, label: str = "", *, gate: bool = True,
                  abort_after_s: Optional[float] = None,
                  abort_after_first_chunk: bool = False,
                  timeout_s: float = CHAT_TIMEOUT_S) -> dict[str, Any]:
        blocked_s = await self.wait_ready() if gate else 0.0
        self.turn += 1
        turn = self.turn
        self.busy = True
        self.last_activity = time.monotonic()  # touchActivity() at turn start
        history = self.history()
        self.messages.append({"role": "user", "content": text})
        started = time.monotonic()
        first_chunk: Optional[float] = None
        parts: list[str] = []
        http_status: Optional[int] = None
        outcome = "ok"
        detail: Optional[str] = None
        stream_error: Optional[str] = None

        async def _do() -> None:
            nonlocal first_chunk, http_status, outcome, detail, stream_error
            body = {"message": text, "history": history}
            if self.protocol == "fixed":
                body["client_session_id"] = self.client_session_id
                timeout = httpx.Timeout(connect=10.0, read=FIRST_BYTE_TIMEOUT_S, write=30.0, pool=30.0)
            else:
                timeout = httpx.Timeout(None)
            async with httpx.AsyncClient(timeout=timeout) as c:
                async with c.stream("POST", f"{self.base}/api/chat", json=body) as resp:
                    http_status = resp.status_code
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "ignore")
                        outcome = f"http_{resp.status_code}"
                        try:
                            detail = json.loads(body).get("detail")
                        except Exception:
                            detail = body[:300]
                        return
                    done = False
                    lines = resp.aiter_lines()
                    while True:
                        try:
                            if self.protocol == "fixed":
                                line = await asyncio.wait_for(lines.__anext__(), INTER_CHUNK_TIMEOUT_S)
                            else:
                                line = await lines.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        ev = json.loads(line)
                        if ev.get("type") == "chunk":
                            if first_chunk is None:
                                first_chunk = time.monotonic()
                            parts.append(ev.get("content", ""))
                            if abort_after_first_chunk:
                                outcome = "aborted_by_client"
                                return
                        elif ev.get("type") == "error":
                            stream_error = ev.get("detail")
                        elif ev.get("type") == "done":
                            done = True
                    if not done and not stream_error:
                        outcome = "stream_cut"
                    if stream_error:
                        outcome = "stream_error"

        if abort_after_s is not None:
            limit = abort_after_s
        elif self.protocol == "fixed":
            # first byte within 200 s; the 30 s inter-chunk limit is the httpx read timeout
            limit = FIRST_BYTE_TIMEOUT_S + 600.0
        else:
            limit = timeout_s
        try:
            await asyncio.wait_for(_do(), timeout=limit)
        except asyncio.TimeoutError:
            outcome = "aborted_by_client" if abort_after_s is not None else (
                "client_timeout_200s" if self.protocol == "fixed" else "client_timeout_90s")
        except httpx.ReadTimeout:
            outcome = "client_timeout_first_byte_200s"
        except Exception as error:
            outcome = "client_error"
            detail = repr(error)
        finally:
            self.busy = False

        reply = "".join(parts)
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
        rec = dict(
            label=label, turn=turn, text=text, outcome=outcome, http=http_status,
            detail=detail or stream_error, blocked_s=blocked_s,
            ttfc_s=round(first_chunk - started, 3) if first_chunk else None,
            e2e_s=round(time.monotonic() - started, 3),
            user_wait_s=round(blocked_s + (time.monotonic() - started), 3),
            history_msgs=len(history), history_chars=sum(len(m["content"]) for m in history),
            reply_chars=len(reply), reply=reply,
        )
        self.log("turn", **rec)
        if outcome.startswith("http_503"):
            self.refresh_now()
        return rec

    async def transcribe(self, path: str) -> Optional[str]:
        started = time.monotonic()
        with open(path, "rb") as fh:
            resp = await self.client.post(
                f"{self.base}/api/transcribe",
                files={"file": (Path(path).name, fh, "audio/mpeg")},
                timeout=httpx.Timeout(120.0),
            )
        body = resp.json()
        self.log("transcribe", http=resp.status_code, text=body.get("text"),
                 elapsed_s=round(time.monotonic() - started, 3))
        return body.get("text")

    # -------------------------------------------------------- proxy follow
    def _proxy_events(self) -> list[dict[str, Any]]:
        if not self.proxy_log or not self.proxy_log.exists():
            return []
        out = []
        for line in self.proxy_log.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    async def wait_for_proxy(self, kinds: list[str], phase: str, max_wait_s: float) -> Optional[dict]:
        """Wait for a new proxy record (phase 'start' or 'end') of the given kinds."""
        seen = {(e.get("id"), e.get("event")) for e in self._proxy_events()}
        started = time.monotonic()
        while time.monotonic() - started < max_wait_s:
            for e in self._proxy_events():
                key = (e.get("id"), e.get("event"))
                if key in seen:
                    continue
                seen.add(key)
                is_start = e.get("event") == "start"
                if e.get("kind") in kinds and ((phase == "start") == is_start):
                    return e
            await asyncio.sleep(0.3)
        return None

    # ------------------------------------------------------------- scenario
    async def run_step(self, step: dict[str, Any]) -> None:
        kind = step["type"]
        if kind == "ask":
            await self.ask(step["text"], step.get("label", ""),
                           abort_after_s=step.get("abort_after_s"),
                           abort_after_first_chunk=step.get("abort_after_first_chunk", False),
                           gate=step.get("gate", True))
        elif kind == "wait":
            self.log("wait", seconds=step["seconds"], note=step.get("note"))
            await asyncio.sleep(step["seconds"])
        elif kind == "new_session":
            await self.new_session(step.get("reason", "manual"))
        elif kind == "wait_ready":
            waited = await self.wait_ready()
            self.log("ready", waited_s=waited)
        elif kind == "wait_for_proxy":
            self.log("wait_for_proxy", kinds=step["kinds"], phase=step["phase"])
            ev = await self.wait_for_proxy(step["kinds"], step["phase"], step.get("max_wait_s", 900))
            self.log("proxy_seen", proxy_id=ev and ev.get("id"), kind=ev and ev.get("kind"))
            if step.get("then_delay"):
                await asyncio.sleep(step["then_delay"])
        elif kind == "parallel":
            await asyncio.gather(*(self.ask(t, step.get("label", "par"), gate=False)
                                   for t in step["texts"]))
        elif kind == "transcribe_and_ask":
            text = await self.transcribe(step["file"])
            if text:
                await self.ask(text, step.get("label", "stt"))
        elif kind == "mark":
            self.log("mark", name=step["name"])
        else:
            raise ValueError(f"unknown step {kind}")

    async def _guard(self, name: str, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as error:  # never let a background loop die silently
            self.log("driver_task_crashed", task=name, error=repr(error))
            raise

    async def run(self, steps: list[dict[str, Any]]) -> None:
        self._poll_wakeup = asyncio.Event()
        self._tasks = [asyncio.create_task(self._guard("poll_health", self.poll_health())),
                       asyncio.create_task(self._guard("idle_rotation", self.idle_rotation()))]
        self.log("driver_start", steps=len(steps), idle_rotate_s=self.idle_rotate_s)
        try:
            for step in steps:
                await self.run_step(step)
        finally:
            self.log("driver_end")
            for t in self._tasks:
                t.cancel()


def soak_steps(seed: int, minutes: float, bank: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Random realistic usage: sessions of 2-10 turns, 10-90 s gaps, long silences."""
    rng = random.Random(seed)
    pool = [q for qs in bank.values() for q in qs]
    steps: list[dict[str, Any]] = []
    budget = minutes * 60
    used = 0.0
    while used < budget:
        n_turns = rng.randint(2, 10)
        for i in range(n_turns):
            steps.append({"type": "ask", "text": rng.choice(pool), "label": f"soak"})
            gap = rng.uniform(10, 90)
            steps.append({"type": "wait", "seconds": round(gap, 1)})
            used += gap + 15  # rough per-turn cost
        if rng.random() < 0.35:
            silence = rng.uniform(610, 700)  # long enough to trigger idle rotation
            steps.append({"type": "wait", "seconds": round(silence, 1), "note": "long_silence"})
            used += silence
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proxy-log")
    ap.add_argument("--idle-rotate-s", type=float, default=600.0)
    ap.add_argument("--soak-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", choices=("legacy", "fixed"), default="legacy")
    args = ap.parse_args()

    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    if isinstance(scenario, dict) and scenario.get("generator") == "soak":
        bank = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
        steps = soak_steps(args.seed, args.soak_minutes, bank["bank"])
    else:
        steps = scenario["steps"]
    driver = Driver(args.base, Path(args.out),
                    Path(args.proxy_log) if args.proxy_log else None, args.idle_rotate_s,
                    protocol=args.protocol)
    asyncio.run(driver.run(steps))


if __name__ == "__main__":
    main()
