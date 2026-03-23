from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel

from .config import Settings, get_settings
from .transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    language: str


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


class ChatRequest(BaseModel):
    message: str


settings: Settings = get_settings()
app = FastAPI(
    title="Whisper Local API",
    description="Local transcription API backed by faster-whisper.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transcriber: Optional[WhisperTranscriber] = None


@app.on_event("startup")
def startup_event() -> None:
    global transcriber
    transcriber = WhisperTranscriber(settings)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "whisper-local-api",
        "docs": "/docs",
        "health": "/api/health",
        "transcribe": "/api/transcribe",
        "chat": "/api/chat",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.whisper_model_size,
        device=settings.whisper_device,
        language=settings.whisper_language,
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

        result = transcriber.transcribe_file(str(temp_path))
        if not result.text:
            raise HTTPException(status_code=422, detail="Transcription is empty.")

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


# ---------------------------------------------------------------------------
# Context fetch from the external Context Service
# ---------------------------------------------------------------------------

async def _fetch_context() -> Optional[str]:
    """Fetch the latest Teacher insight from the Context Service.

    Returns the system prompt string, or None if the service is unavailable.
    Fails silently — chat works without context, just less informed.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{settings.context_service_url}/api/context")
            if resp.status_code != 200:
                return None
            body = resp.json()
            if body.get("status") != "ready":
                return None
            return body.get("system_prompt")
    except Exception:
        logger.debug("Context Service unavailable, proceeding without context.")
        return None


def _build_ollama_payload(messages: List[Dict[str, str]]) -> dict[str, object]:
    return {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
    }


async def _stream_ollama_reply(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            settings.ollama_url,
            json=_build_ollama_payload(messages),
        ) as upstream_response:
            if upstream_response.status_code >= 400:
                error_body = (await upstream_response.aread()).decode("utf-8", errors="ignore")
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama error {upstream_response.status_code}: {error_body[:500]}",
                )

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

    # Build messages: optionally inject system context from Context Service
    messages: List[Dict[str, str]] = []
    system_prompt = await _fetch_context()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in _stream_ollama_reply(messages):
                yield chunk.encode("utf-8")
        except HTTPException as error:
            yield f"[BŁĄD OLLAMA] {error.detail}".encode("utf-8")
        except httpx.HTTPError as error:
            yield f"[BŁĄD OLLAMA] Nie udało się połączyć z Ollamą: {error}".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")
