"""Context Service — standalone microservice that feeds Teacher insights to the whisper backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .config import get_settings
from .context_engine import ContextEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

engine = ContextEngine(
    teacher_url=settings.teacher_ollama_url,
    teacher_model=settings.teacher_ollama_model,
    sensor_csv_path=Path(settings.sensor_data_path),
    refresh_interval=settings.context_refresh_interval,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    logger.info("Context Engine background refresh started.")
    yield
    await engine.stop()


app = FastAPI(
    title="Context Service",
    description="Provides Teacher-analyzed smart home context for the Student LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "context-service",
        "context": "/api/context",
        "health": "/api/health",
    }


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    sim = engine.simulator_status
    return {
        "status": "ok",
        "context_ready": engine.is_ready,
        "teacher_model": settings.teacher_ollama_model,
        "refresh_interval": settings.context_refresh_interval,
        "simulation": sim,
    }


@app.get("/api/context", tags=["context"])
def get_context() -> dict:
    """Returns the current Teacher insight and pre-built system prompt.

    The whisper backend fetches this and injects system_prompt into the
    Student's Ollama messages array.
    """
    if not engine.is_ready:
        return {"status": "pending", "system_prompt": None, "insight": None}

    insight = engine.latest_insight
    return {
        "status": "ready",
        "source_timestamp": insight.source_timestamp if insight else None,
        "insight": insight.summary if insight else None,
        "system_prompt": engine.system_prompt,
    }
