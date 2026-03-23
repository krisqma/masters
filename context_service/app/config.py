from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _as_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        v = value.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


@dataclass(frozen=True)
class Settings:
    teacher_ollama_url: str
    teacher_ollama_model: str
    context_refresh_interval: int
    sensor_data_path: str


def get_settings() -> Settings:
    _load_env_file()
    default_csv = str(Path(__file__).resolve().parents[2] / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv")
    return Settings(
        teacher_ollama_url=os.getenv("TEACHER_OLLAMA_URL", "http://localhost:11434/api/chat"),
        teacher_ollama_model=os.getenv("TEACHER_OLLAMA_MODEL", "qwen2.5"),
        context_refresh_interval=_as_int(os.getenv("CONTEXT_REFRESH_INTERVAL"), 300),
        sensor_data_path=os.getenv("SENSOR_DATA_PATH", default_csv),
    )
