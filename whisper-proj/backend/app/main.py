from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import AsyncGenerator, Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import httpx
from pydantic import BaseModel, Field

from .config import FALLBACK_STUDENT_STATIC_PREFIX, Settings, get_settings
from .session_state import ConversationSession
from .student_warmup import StudentKeepWarm
from .transcriber import WhisperTranscriber
from .tts import synthesize_speech

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
    history: List[ChatHistoryItem] = Field(default_factory=list)


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
    await student_keep_warm.start()
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
# Context fetch from the external Context Service
# ---------------------------------------------------------------------------

class StudentContext:
    """Prefix-cache-friendly prompt parts from Context Service."""

    __slots__ = ("static_prefix", "dynamic_context", "source_timestamp")

    def __init__(
        self,
        static_prefix: str,
        dynamic_context: Optional[str],
        source_timestamp: Optional[str] = None,
    ) -> None:
        self.static_prefix = static_prefix
        self.dynamic_context = dynamic_context
        self.source_timestamp = source_timestamp


async def _fetch_published_context() -> StudentContext:
    """Fetch published context from Context Service (does not rotate)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{settings.context_service_url}/api/context")
            if resp.status_code != 200:
                return StudentContext(FALLBACK_STUDENT_STATIC_PREFIX, None)
            body = resp.json()
            static_prefix = body.get("static_prefix") or FALLBACK_STUDENT_STATIC_PREFIX
            dynamic = body.get("dynamic_context")
            ts = body.get("source_timestamp")
            if body.get("status") != "ready" or not isinstance(dynamic, str) or not dynamic.strip():
                return StudentContext(static_prefix, None, ts if isinstance(ts, str) else None)
            return StudentContext(
                static_prefix,
                dynamic,
                ts if isinstance(ts, str) else None,
            )
    except Exception:
        logger.debug("Context Service unavailable, proceeding with fallback static prefix.")
        return StudentContext(FALLBACK_STUDENT_STATIC_PREFIX, None)


async def _get_or_pin_context() -> StudentContext:
    pin = conversation_session.pin
    if pin is not None:
        return StudentContext(pin.static_prefix, pin.dynamic_context, pin.source_timestamp)
    ctx = await _fetch_published_context()
    conversation_session.set_pin(
        static_prefix=ctx.static_prefix,
        dynamic_context=ctx.dynamic_context,
        source_timestamp=ctx.source_timestamp,
    )
    return ctx


def _normalize_history(history: List[ChatHistoryItem]) -> List[Dict[str, str]]:
    """Keep last N non-empty user/assistant turns for multi-turn chat."""
    max_messages = max(0, settings.chat_history_max_messages)
    cleaned: List[Dict[str, str]] = []
    for item in history:
        content = item.content.strip()
        if not content:
            continue
        cleaned.append({"role": item.role, "content": content})
    if max_messages == 0:
        return []
    return cleaned[-max_messages:]


def _build_chat_messages(
    user_message: str,
    context: StudentContext,
    history: List[ChatHistoryItem],
) -> List[Dict[str, str]]:
    """system = immutable rules; fresh home context once; then prior turns; then question."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": context.static_prefix},
    ]
    prior = _normalize_history(history)

    if context.dynamic_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{context.dynamic_context}\n\n"
                    "Powyższy kontekst domu jest dostępny w tej turze. "
                    "Liczby i pokoje bierz z bloku „Stan mieszkania”; "
                    "Findings (jeśli są) to tylko opcjonalne anomalie. "
                    "Cytuj dane tylko, gdy pytanie dotyczy stanu domu. "
                    "Uwzględniaj też wcześniejsze wiadomości."
                ),
            }
        )
    messages.extend(prior)
    messages.append({"role": "user", "content": user_message})
    return messages


def _build_ollama_payload(messages: List[Dict[str, str]]) -> dict[str, object]:
    return {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
        "keep_alive": settings.ollama_keep_alive,
    }


def _encode_stream_event(event_type: str, **payload: str) -> bytes:
    body = {"type": event_type, **payload}
    return f"{json.dumps(body, ensure_ascii=False)}\n".encode("utf-8")


async def _open_ollama_stream(
    messages: List[Dict[str, str]],
) -> tuple[httpx.AsyncClient, httpx.Response]:
    timeout = httpx.Timeout(
        connect=10.0,
        read=settings.ollama_read_timeout_seconds,
        write=30.0,
        pool=30.0,
    )
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request(
        "POST",
        settings.ollama_url,
        json=_build_ollama_payload(messages),
    )

    try:
        upstream_response = await client.send(request, stream=True)
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


async def _stream_ollama_reply(upstream_response: httpx.Response) -> AsyncGenerator[str, None]:
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
            break


@app.post("/api/chat", tags=["chat"])
async def chat(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if student_keep_warm.student_status == "warming":
        raise HTTPException(
            status_code=503,
            detail="Model się rozgrzewa — spróbuj za chwilę.",
        )

    started = time.monotonic()
    conversation_session.touch()
    context = await _get_or_pin_context()
    messages = _build_chat_messages(message, context, request.history)
    history_msgs = len(_normalize_history(request.history))
    facts_chars = len(context.dynamic_context) if context.dynamic_context else 0

    student_keep_warm.set_chat_busy(True)
    await student_keep_warm.acquire_ollama()
    try:
        client, upstream_response = await _open_ollama_stream(messages)
    except Exception:
        student_keep_warm.release_ollama()
        student_keep_warm.set_chat_busy(False)
        raise

    async def event_stream() -> AsyncGenerator[bytes, None]:
        first_chunk_at: Optional[float] = None
        try:
            async for chunk in _stream_ollama_reply(upstream_response):
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                yield _encode_stream_event("chunk", content=chunk)
        except httpx.TimeoutException:
            yield _encode_stream_event("error", detail="Ollama przekroczyła limit czasu odpowiedzi.")
        except httpx.HTTPError as error:
            yield _encode_stream_event("error", detail=f"Nie udało się połączyć z Ollamą: {error}")
        except Exception:  # pragma: no cover - runtime safety
            logger.exception("Streaming reply failed.")
            yield _encode_stream_event(
                "error",
                detail="Wystąpił nieoczekiwany błąd podczas generowania odpowiedzi.",
            )
        finally:
            e2e_ms = int((time.monotonic() - started) * 1000)
            ttfb_ms = (
                int((first_chunk_at - started) * 1000) if first_chunk_at is not None else None
            )
            logger.info(
                "Chat done: ttfb_ms=%s e2e_ms=%s facts_chars=%d history_msgs=%d "
                "prompt_msgs=%d context_ts=%s pinned=1",
                ttfb_ms,
                e2e_ms,
                facts_chars,
                history_msgs,
                len(messages),
                context.source_timestamp,
            )
            yield _encode_stream_event("done")
            await upstream_response.aclose()
            await client.aclose()
            student_keep_warm.release_ollama()
            student_keep_warm.set_chat_busy(False)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
