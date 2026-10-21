"""Transparent logging proxy: backend -> (this) -> Ollama on the RPi.

Forwards every request byte-for-byte (NDJSON streams included) and appends one
JSON line per request to $PROXY_LOG with the request shape and the final Ollama
stats (prompt_eval_count tells how many prompt tokens the RPi actually
re-computed, i.e. whether the KV prefix cache was reused).

Run:  UPSTREAM=http://192.168.1.173:11434 PROXY_LOG=run/proxy.jsonl \
      uvicorn ollama_proxy:app --host 127.0.0.1 --port 11500
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("UPSTREAM", "http://192.168.1.173:11434").rstrip("/")
PROXY_LOG = os.environ.get("PROXY_LOG", "proxy.jsonl")

_client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
_ids = itertools.count(1)
_in_flight = 0
_log_lock = asyncio.Lock()

STAT_KEYS = (
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "load_duration",
    "total_duration",
    "done_reason",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _classify(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    options = body.get("options") or {}
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    if last_user.strip() == "ping":
        return "ping_static"
    if options.get("num_predict") == 1:
        return "ping_facts"
    return "chat"


def _describe(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    return {
        "model": body.get("model"),
        "kind": _classify(body),
        "stream": body.get("stream", True),
        "keep_alive": body.get("keep_alive"),
        "options": body.get("options"),
        "n_messages": len(messages),
        "msg_roles": [m.get("role") for m in messages],
        "msg_chars": [len(m.get("content", "")) for m in messages],
        "msg_hashes": [_h(m.get("content", "")) for m in messages],
        "prompt_chars": sum(len(m.get("content", "")) for m in messages),
        "messages": messages,
    }


async def _write(record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    async with _log_lock:
        with open(PROXY_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _extract_stats(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: payload.get(k) for k in STAT_KEYS if k in payload}


async def chat(request: Request) -> Response:
    global _in_flight
    req_id = next(_ids)
    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {}
    record: dict[str, Any] = {
        "id": req_id,
        "path": request.url.path,
        "start": _now_iso(),
        "in_flight_at_start": _in_flight,
        **_describe(body),
    }
    started = time.monotonic()
    _in_flight += 1
    await _write(
        {
            "event": "start",
            "id": req_id,
            "start": record["start"],
            "kind": record["kind"],
            "n_messages": record["n_messages"],
            "in_flight_at_start": record["in_flight_at_start"],
        }
    )
    record["event"] = "end"

    headers = {"content-type": request.headers.get("content-type", "application/json")}
    upstream_req = _client.build_request(
        "POST", f"{UPSTREAM}{request.url.path}", content=raw, headers=headers
    )
    # Ollama sends response headers only after prefill, so a plain `await send()`
    # would not notice the backend hanging up meanwhile and would keep the RPi
    # working on an orphan. Watch for the downstream disconnect and drop the
    # upstream connection at the same moment the backend does (as without proxy).
    send_task = asyncio.create_task(_client.send(upstream_req, stream=True))
    try:
        while True:
            done, _ = await asyncio.wait({send_task}, timeout=0.5)
            if done:
                break
            if await request.is_disconnected():
                send_task.cancel()
                try:
                    await send_task
                except BaseException:
                    pass
                _in_flight -= 1
                record.update(
                    end=_now_iso(),
                    elapsed_s=round(time.monotonic() - started, 3),
                    outcome="client_disconnected_before_headers",
                )
                await _write(record)
                return Response(status_code=499)
        upstream = send_task.result()
    except Exception as error:  # connection refused / no route / timeout
        _in_flight -= 1
        record.update(
            end=_now_iso(),
            elapsed_s=round(time.monotonic() - started, 3),
            outcome="upstream_connect_error",
            error=repr(error),
        )
        await _write(record)
        return Response(f"proxy upstream error: {error}", status_code=502)

    record["status"] = upstream.status_code
    record["upstream_headers_s"] = round(time.monotonic() - started, 3)

    async def body_iter():
        global _in_flight
        first_byte: Optional[float] = None
        first_content: Optional[float] = None
        reply_parts: list[str] = []
        stats: dict[str, Any] = {}
        outcome = "completed"
        buffer = b""
        try:
            async for chunk in upstream.aiter_raw():
                now = time.monotonic()
                if first_byte is None:
                    first_byte = now
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = (payload.get("message") or {}).get("content", "")
                    if content:
                        if first_content is None:
                            first_content = now
                        reply_parts.append(content)
                    if payload.get("done") is True:
                        stats = _extract_stats(payload)
                    if "error" in payload:
                        record["ollama_error"] = payload["error"]
                yield chunk
            if buffer.strip():
                try:
                    payload = json.loads(buffer)
                    content = (payload.get("message") or {}).get("content", "")
                    if content:
                        if first_content is None:
                            first_content = time.monotonic()
                        reply_parts.append(content)
                    if payload.get("done") is True:
                        stats = _extract_stats(payload)
                    if "error" in payload:
                        record["ollama_error"] = payload["error"]
                except json.JSONDecodeError:
                    pass
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "client_disconnected"
            raise
        except Exception as error:
            outcome = "upstream_stream_error"
            record["error"] = repr(error)
            raise
        finally:
            _in_flight -= 1
            # anyio re-cancels every await inside a cancelled scope; shield cleanup + log.
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
                end = time.monotonic()
                reply = "".join(reply_parts)
                record.update(
                    end=_now_iso(),
                    elapsed_s=round(end - started, 3),
                    first_byte_s=round(first_byte - started, 3) if first_byte else None,
                    first_content_s=round(first_content - started, 3) if first_content else None,
                    outcome=outcome,
                    reply_chars=len(reply),
                    reply=reply,
                    **stats,
                )
                pec = stats.get("prompt_eval_count")
                ped = stats.get("prompt_eval_duration")
                ec = stats.get("eval_count")
                ed = stats.get("eval_duration")
                if pec and ped:
                    record["prefill_s"] = round(ped / 1e9, 3)
                    record["prefill_tok_s"] = round(pec / (ped / 1e9), 2)
                if ec and ed:
                    record["gen_s"] = round(ed / 1e9, 3)
                    record["gen_tok_s"] = round(ec / (ed / 1e9), 2)
                if stats.get("load_duration"):
                    record["load_s"] = round(stats["load_duration"] / 1e9, 3)
                await _write(record)

    passthrough_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in {"content-type"}
    }
    return StreamingResponse(
        body_iter(), status_code=upstream.status_code, headers=passthrough_headers
    )


async def passthrough(request: Request) -> Response:
    raw = await request.body()
    try:
        resp = await _client.request(
            request.method,
            f"{UPSTREAM}{request.url.path}",
            content=raw or None,
            params=request.query_params,
        )
    except Exception as error:
        await _write(
            {
                "path": request.url.path,
                "method": request.method,
                "start": _now_iso(),
                "outcome": "upstream_connect_error",
                "error": repr(error),
            }
        )
        return Response(f"proxy upstream error: {error}", status_code=502)
    return Response(
        resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


app = Starlette(
    routes=[
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/generate", chat, methods=["POST"]),
        Route("/{path:path}", passthrough, methods=["GET", "POST", "HEAD", "DELETE"]),
    ]
)
