"""Context Service — published session facts + staging background refresh."""

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
    description=(
        "Provides deterministic smart-home sensor facts for the Student LLM, "
        "plus optional Teacher anomaly findings. "
        "GET returns the published (session) snapshot; POST /api/context/advance "
        "rotates published from staging for a new conversation."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "context-service",
        "context": "/api/context",
        "advance": "/api/context/advance",
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
    """Returns the published Student prompt parts (does not rotate)."""
    base = {
        "static_prefix": engine.static_prefix,
        "dynamic_context": engine.dynamic_context,
        "system_prompt": engine.system_prompt,
    }
    if not engine.is_ready:
        return {
            **base,
            "status": "pending",
            "insight": None,
            "source_timestamp": None,
        }

    insight = engine.latest_insight
    return {
        **base,
        "status": "ready",
        "source_timestamp": engine.source_timestamp,
        "insight": insight.summary if insight else None,
    }


@app.post("/api/context/advance", tags=["context"])
async def advance_context() -> dict:
    """Publish staging (or a fresh snapshot) for a new conversation session."""
    result = await engine.advance()
    return result
