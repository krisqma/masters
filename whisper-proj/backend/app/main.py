from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import AsyncGenerator, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import httpx
from pydantic import BaseModel, Field

from .config import FALLBACK_STUDENT_STATIC_PREFIX, Settings, get_settings
from .session_state import ConversationSession, SessionPin
from .student_prompt import (
    Turn,
    build_prefix_messages,
    ensure_prefix_fits,
    estimate_message_tokens,
    estimate_prefix_tokens,
    measured_turn_tokens,
    plan_request,
    student_options,
)
from .student_warmup import StudentKeepWarm
from .transcriber import WhisperTranscriber
from .tts import synthesize_speech

# INFO lines (per-turn budget log, warm-up) must be visible under plain uvicorn too.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    language: str
    student_warm: bool = False
    student_status: Literal["warming", "warm", "error"] = "warming"
    student_error: Optional[str] = None


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str
    # Only used to seed the backend ledger after a backend restart; the backend
    # keeps the authoritative history (see session_state.py).
    history: List[ChatHistoryItem] = Field(default_factory=list)
    # Changes when the UI starts a new conversation (page load, "Nowa rozmowa").
    client_session_id: Optional[str] = None


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)


settings: Settings = get_settings()
student_keep_warm = StudentKeepWarm(settings)
conversation_session = ConversationSession()
transcriber: Optional[WhisperTranscriber] = None
_warmup_task: Optional[asyncio.Task[None]] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global transcriber
    transcriber = WhisperTranscriber(settings)
    await student_keep_warm.start(_prefix_for_warmup)
    try:
        yield
    finally:
        await student_keep_warm.stop()


app = FastAPI(
    title="Whisper Local API",
    description="Local transcription, chat proxy, and edge-tts speech synthesis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "whisper-local-api",
        "docs": "/docs",
        "health": "/api/health",
        "transcribe": "/api/transcribe",
        "chat": "/api/chat",
        "tts": "/api/tts",
        "session_new": "/api/session/new",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.whisper_model_size,
        device=settings.whisper_device,
        language=settings.whisper_language,
        student_warm=student_keep_warm.student_warm,
        student_status=student_keep_warm.student_status,
        student_error=student_keep_warm.student_error,
    )


class SessionNewResponse(BaseModel):
    status: Literal["warming", "warm", "error"]
    advanced: bool = False
    source_timestamp: Optional[str] = None
    published_facts_chars: Optional[int] = None
    student_status: Literal["warming", "warm", "error"] = "warming"
    idle_rotate_seconds: int = 600


async def _advance_context_service() -> dict:
    url = f"{settings.context_service_url.rstrip('/')}/api/context/advance"
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        resp = await client.post(url)
        resp.raise_for_status()
        return resp.json()


async def _start_session_warmup(*, reason: str) -> None:
    global _warmup_task
    if _warmup_task and not _warmup_task.done():
        return

    async def _run() -> None:
        await student_keep_warm.warm_for_session(reason=reason)

    _warmup_task = asyncio.create_task(_run(), name=f"session-warmup-{reason}")


@app.post("/api/session/new", response_model=SessionNewResponse, tags=["session"])
async def session_new() -> SessionNewResponse:
    """Rotate published home facts and warm Student for a new conversation."""
    advanced = False
    source_timestamp: Optional[str] = None
    facts_chars: Optional[int] = None
    try:
        advance_body = await _advance_context_service()
        advanced = True
        source_timestamp = advance_body.get("source_timestamp")
        facts_chars = advance_body.get("published_facts_chars")
        logger.info(
            "Session new: context advanced ts=%s facts_chars=%s source=%s",
            source_timestamp,
            facts_chars,
            advance_body.get("source"),
        )
    except Exception as error:
        logger.warning("Session new: context advance failed: %s", error)

    conversation_session.clear_pin()
    conversation_session.touch()
    await _start_session_warmup(reason="session_new")

    return SessionNewResponse(
        status="warming",
        advanced=advanced,
        source_timestamp=source_timestamp,
        published_facts_chars=facts_chars,
        student_status=student_keep_warm.student_status,
        idle_rotate_seconds=settings.session_idle_rotate_seconds,
    )


@app.post("/api/transcribe", response_model=TranscriptionResponse, tags=["transcription"])
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet.")

    temp_path: Optional[Path] = None
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    total_size = 0

    try:
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio payload too large. Max is {settings.max_upload_mb} MB.",
                    )
                temp_file.write(chunk)

        if temp_path is None:
            raise HTTPException(status_code=400, detail="No audio file received.")

        if total_size < 1500:
            raise HTTPException(
                status_code=422,
                detail="Nagranie jest zbyt małe (prawie puste). Mów dłużej po aktywacji.",
            )

        logger.info("Transcribing upload: bytes=%s suffix=%s", total_size, suffix)
        result = transcriber.transcribe_file(str(temp_path))
        if not result.text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Transcription is empty. "
                    f"(bytes={total_size}, duration={result.duration_seconds!s}). "
                    "Mów głośniej / dłużej po „hej wilguś”."
                ),
            )

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            duration_seconds=result.duration_seconds,
        )
    except HTTPException:
        raise
    except Exception as error:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=f"Transcription failed: {error}") from error
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/api/tts", tags=["tts"])
async def text_to_speech(request: TtsRequest) -> Response:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > settings.tts_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Max is {settings.tts_max_chars} characters.",
        )

    try:
        audio = await synthesize_speech(
            text,
            voice=settings.tts_voice,
            rate=settings.tts_rate,
        )
    except Exception as error:  # pragma: no cover - network / edge-tts runtime
        logger.exception("TTS synthesis failed.")
        raise HTTPException(
            status_code=502,
            detail=f"Nie udało się zsyntezować mowy (edge-tts): {error}",
        ) from error

    if not audio:
        raise HTTPException(status_code=502, detail="edge-tts zwróciło pusty plik audio.")

    return Response(content=audio, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Context fetch from the external Context Service + session pin
# ---------------------------------------------------------------------------

async def _fetch_published_context() -> Tuple[str, Optional[str], Optional[str], bool]:
    """(static_prefix, dynamic_context, source_timestamp, ready) from Context Service."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{settings.context_service_url}/api/context")
            if resp.status_code != 200:
                return FALLBACK_STUDENT_STATIC_PREFIX, None, None, False
            body = resp.json()
            static_prefix = body.get("static_prefix") or FALLBACK_STUDENT_STATIC_PREFIX
            dynamic = body.get("dynamic_context")
            ts = body.get("source_timestamp")
            ts = ts if isinstance(ts, str) else None
            if body.get("status") != "ready" or not isinstance(dynamic, str) or not dynamic.strip():
                return static_prefix, None, ts, False
            return static_prefix, dynamic, ts, True
    except Exception:
        logger.debug("Context Service unavailable, proceeding with fallback static prefix.")
        return FALLBACK_STUDENT_STATIC_PREFIX, None, None, False


async def _get_or_pin_context() -> SessionPin:
    """Pin the published context for the whole conversation (stable KV prefix)."""
    pin = conversation_session.pin
    if pin is not None:
        return pin
    static_prefix, dynamic, ts, ready = await _fetch_published_context()
    dynamic, degraded = ensure_prefix_fits(static_prefix, dynamic, settings)
    if degraded:
        logger.warning("Session context degraded to fit the token budget: %s", ",".join(degraded))
    if not ready:
        # Do not pin a fallback: retry the Context Service on the next request.
        return SessionPin(static_prefix, dynamic, ts, degraded)
    return conversation_session.set_pin(
        static_prefix=static_prefix,
        dynamic_context=dynamic,
        source_timestamp=ts,
        degraded=degraded,
    )


async def _prefix_for_warmup():
    # At startup the Context Service may still be loading its first snapshot;
    # warming without facts would make the first question pay for their prefill.
    deadline = time.monotonic() + 120.0
    pin = await _get_or_pin_context()
    while conversation_session.pin is None and time.monotonic() < deadline:
        await asyncio.sleep(3.0)
        pin = await _get_or_pin_context()

    def _on_prefix_tokens(tokens: int) -> None:
        pin.prefix_tokens = tokens

    return build_prefix_messages(pin.static_prefix, pin.dynamic_context), _on_prefix_tokens


def _prefix_tokens(pin: SessionPin) -> int:
    if pin.prefix_tokens is not None:
        return pin.prefix_tokens
    return estimate_prefix_tokens(pin.static_prefix, pin.dynamic_context)


def _seed_history_from_client(history: List[ChatHistoryItem]) -> None:
    """After a backend restart the ledger is empty: seed it once from the UI."""
    seeded = conversation_session.seed_from_client([(h.role, h.content) for h in history])
    if seeded:
        logger.info("Seeded backend history from client: %d turns", seeded)


def _build_ollama_payload(messages: List[Dict[str, str]], num_predict: int) -> dict[str, object]:
    return {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
        "keep_alive": settings.ollama_keep_alive,
        "options": student_options(settings, num_predict),
    }


def _encode_stream_event(event_type: str, **payload: str) -> bytes:
    body = {"type": event_type, **payload}
    return f"{json.dumps(body, ensure_ascii=False)}\n".encode("utf-8")


class _ClientGone(Exception):
    """The browser hung up while the RPi was still prefilling."""


async def _open_ollama_stream(
    http_request: Request,
    payload: dict[str, object],
) -> tuple[httpx.AsyncClient, httpx.Response]:
    timeout = httpx.Timeout(
        connect=10.0,
        read=settings.ollama_read_timeout_seconds,
        write=30.0,
        pool=30.0,
    )
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", settings.ollama_url, json=payload)

    # Ollama sends headers only after the prefill. Watch the browser meanwhile and
    # drop the upstream request as soon as it leaves, so the RPi stops working
    # on an answer nobody will read (otherwise the next question queues behind it).
    send_task = asyncio.create_task(client.send(request, stream=True))
    try:
        while True:
            done, _ = await asyncio.wait({send_task}, timeout=0.5)
            if done:
                break
            if await http_request.is_disconnected():
                send_task.cancel()
                try:
                    await send_task
                except BaseException:
                    pass
                await client.aclose()
                raise _ClientGone()
        upstream_response = send_task.result()
    except httpx.TimeoutException as error:
        await client.aclose()
        raise HTTPException(
            status_code=504,
            detail="Ollama przekroczyła limit czasu odpowiedzi.",
        ) from error
    except httpx.HTTPError as error:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Nie udało się połączyć z Ollamą: {error}",
        ) from error

    if upstream_response.status_code >= 400:
        error_body = (await upstream_response.aread()).decode("utf-8", errors="ignore")
        await upstream_response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {upstream_response.status_code}: {error_body[:500]}",
        )

    return client, upstream_response


async def _stream_ollama_reply(
    upstream_response: httpx.Response,
    final: Dict[str, object],
) -> AsyncGenerator[str, None]:
    """Yield reply chunks; the final `done` line's stats are stored in `final`."""
    async for line in upstream_response.aiter_lines():
        raw = line.strip()
        if not raw:
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        chunk = payload.get("message", {}).get("content", "")
        if isinstance(chunk, str) and chunk:
            yield chunk

        if payload.get("done") is True:
            final.update(payload)
            break


@app.post("/api/chat", tags=["chat"])
async def chat(request: ChatRequest, http_request: Request) -> Response:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    started = time.monotonic()
    conversation_session.touch()
    if conversation_session.bind_client(request.client_session_id):
        logger.info("New client session %s: history reset.", request.client_session_id)

    # Wait for any warm-up / previous turn instead of failing with 503.
    student_keep_warm.set_chat_busy(True)
    await student_keep_warm.acquire_ollama()
    lock_wait_s = time.monotonic() - started
    try:
        pin = await _get_or_pin_context()
        _seed_history_from_client(request.history)
        prefix_tokens = _prefix_tokens(pin)
        plan = plan_request(
            settings=settings,
            prefix_messages=build_prefix_messages(pin.static_prefix, pin.dynamic_context),
            prefix_tokens=prefix_tokens,
            turns=conversation_session.turns,
            history_start=conversation_session.history_start,
            question=message,
        )
        client, upstream_response = await _open_ollama_stream(
            http_request, _build_ollama_payload(plan.messages, plan.num_predict)
        )
    except _ClientGone:
        student_keep_warm.release_ollama()
        student_keep_warm.set_chat_busy(False)
        logger.info(
            "Chat aborted: client left during prefill after %.1fs (question dropped from history).",
            time.monotonic() - started,
        )
        return Response(status_code=499)
    except Exception:
        student_keep_warm.release_ollama()
        student_keep_warm.set_chat_busy(False)
        raise

    async def event_stream() -> AsyncGenerator[bytes, None]:
        first_chunk_at: Optional[float] = None
        final: Dict[str, object] = {}
        parts: List[str] = []
        outcome = "aborted"
        try:
            async for chunk in _stream_ollama_reply(upstream_response, final):
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                parts.append(chunk)
                yield _encode_stream_event("chunk", content=chunk)
            outcome = "done" if final else "cut"
        except httpx.TimeoutException:
            outcome = "timeout"
            yield _encode_stream_event("error", detail="Ollama przekroczyła limit czasu odpowiedzi.")
        except httpx.HTTPError as error:
            outcome = "http_error"
            yield _encode_stream_event("error", detail=f"Nie udało się połączyć z Ollamą: {error}")
        except Exception:  # pragma: no cover - runtime safety
            outcome = "error"
            logger.exception("Streaming reply failed.")
            yield _encode_stream_event(
                "error",
                detail="Wystąpił nieoczekiwany błąd podczas generowania odpowiedzi.",
            )
        finally:
            prompt_eval = final.get("prompt_eval_count")
            eval_count = final.get("eval_count")
            committed = False
            # Only complete exchanges enter the history: a turn that never got
            # `done` would leave a question without its answer in the prompt.
            if outcome == "done" and isinstance(prompt_eval, int) and isinstance(eval_count, int):
                tokens = measured_turn_tokens(
                    prompt_eval_count=prompt_eval,
                    eval_count=eval_count,
                    prefix_tokens=prefix_tokens,
                    history_tokens=plan.history_tokens,
                )
                conversation_session.commit_turn(
                    Turn(user=message, reply="".join(parts), tokens=tokens, exact=True),
                    history_start=plan.history_start,
                )
                committed = True
            e2e_ms = int((time.monotonic() - started) * 1000)
            ttfb_ms = (
                int((first_chunk_at - started) * 1000) if first_chunk_at is not None else None
            )
            total_tokens = (
                prompt_eval + eval_count
                if isinstance(prompt_eval, int) and isinstance(eval_count, int)
                else None
            )
            logger.info(
                "Chat done: outcome=%s ttfb_ms=%s e2e_ms=%s lock_wait_ms=%d "
                "prompt_tok=%s est_tok=%d slot_tok=%s prefill_s=%.1f eval_tok=%s num_predict=%d "
                "done_reason=%s hist_turns=%d hist_tok=%d trimmed=%s committed=%s "
                "prefix_tok=%d prefix_exact=%s degraded=%s notes=%s context_ts=%s",
                outcome,
                ttfb_ms,
                e2e_ms,
                int(lock_wait_s * 1000),
                prompt_eval,
                plan.prompt_tokens_est,
                total_tokens,
                (final.get("prompt_eval_duration") or 0) / 1e9,
                eval_count,
                plan.num_predict,
                final.get("done_reason"),
                plan.kept_turns,
                plan.history_tokens,
                plan.trimmed,
                committed,
                prefix_tokens,
                pin.prefix_tokens is not None,
                ",".join(pin.degraded) or "-",
                ",".join(plan.notes) or "-",
                pin.source_timestamp,
            )
            if total_tokens is not None and total_tokens > settings.student_ctx_window:
                logger.warning(
                    "Student slot exceeded the %d-token window (%d tokens): estimate too low.",
                    settings.student_ctx_window,
                    total_tokens,
                )
            yield _encode_stream_event("done")
            await upstream_response.aclose()
            await client.aclose()
            student_keep_warm.release_ollama()
            student_keep_warm.set_chat_busy(False)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
