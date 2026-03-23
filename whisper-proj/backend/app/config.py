from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _as_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_csv_list(value: Optional[str], default: List[str]) -> List[str]:
    if value is None:
        return default
    parts = [part.strip() for part in value.split(",")]
    items = [part for part in parts if part]
    return items or default


def _load_local_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        if normalized_key:
            os.environ.setdefault(normalized_key, normalized_value)


@dataclass(frozen=True)
class Settings:
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str
    whisper_beam_size: int
    whisper_vad_filter: bool
    max_upload_mb: int
    cors_origins: List[str]
    ollama_url: str
    ollama_model: str
    context_service_url: str


DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def get_settings() -> Settings:
    _load_local_env_file()
    return Settings(
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "pl"),
        whisper_beam_size=_as_int(os.getenv("WHISPER_BEAM_SIZE"), 5),
        whisper_vad_filter=_as_bool(os.getenv("WHISPER_VAD_FILTER"), True),
        max_upload_mb=_as_int(os.getenv("MAX_UPLOAD_MB"), 25),
        cors_origins=_as_csv_list(os.getenv("CORS_ORIGINS"), DEFAULT_CORS_ORIGINS),
        ollama_url=os.getenv("OLLAMA_URL", "http://192.168.1.173:11434/api/chat"),
        ollama_model=os.getenv("OLLAMA_MODEL", "wilgus-pl"),
        context_service_url=os.getenv("CONTEXT_SERVICE_URL", "http://127.0.0.1:8001"),
    )
