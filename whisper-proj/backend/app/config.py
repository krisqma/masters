from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union


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


def _as_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_keep_alive(value: Optional[str], default: Union[int, str]) -> Union[int, str]:
    """Ollama wants a number for "forever" (-1); durations stay strings ("30m")."""
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return value.strip()


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
    ollama_keep_alive: Union[int, str]
    ollama_read_timeout_seconds: int
    chat_history_max_messages: int
    student_keep_warm_enabled: bool
    student_keep_warm_interval_seconds: int
    session_idle_rotate_seconds: int
    # gemma3 sliding window: prompt + reply must stay within it (see stability/RAPORT.md)
    student_ctx_window: int
    student_budget_margin: int
    student_num_predict_max: int
    student_num_predict_min: int
    student_history_trim_ratio: float
    student_history_keep_max_tokens: int
    student_temperature: float
    student_top_p: float
    student_repeat_penalty: float
    student_num_ctx: int
    student_num_batch: Optional[int]
    student_num_thread: Optional[int]
    context_service_url: str
    tts_voice: str
    tts_rate: str
    tts_max_chars: int


DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

# Fallback when Context Service is down; must be byte-identical to context_service STUDENT_STATIC_PREFIX.
FALLBACK_STUDENT_STATIC_PREFIX = """\
Jesteś Wilguś (zdrobnienie od wilgi, ptaka), przyjazny gospodarz domku gościnnego. Mów krótko, ciepło, po polsku, w 1. osobie. Informujesz o komforcie domu z czujników, niczym nie sterujesz.
NAJWAŻNIEJSZE: liczbę podajesz TYLKO wtedy, gdy stoi w bloku „Stan mieszkania” przy tym pokoju. Gdy jej tam nie ma albo jest „brak …”, powiedz, że nie masz tego odczytu. NIGDY nie wymyślaj liczb ani pokoi.
Zasady:
- Powitanie, rozmowa, pytanie o Ciebie lub domek, niezrozumiała wypowiedź: 1-2 zdania z persony, BEZ liczb z czujników.
- Niejasne lub błąd STT: poproś o powtórzenie; literówki (np. „Sanon”) mapuj na najbliższy pokój z listy.
- O cały dom: każdy pokój z listy osobno, liczbami albo „brak odczytu”, nie „w normie”.
- Pomieszczenia: wszystkie z listy, nic więcej.
- „Findings”: opcjonalne anomalie, nie zastępują faktów.
- Nie uśredniaj; przy danych sprzecznych lub starych zaznacz niepewność.
- Pamiętaj wcześniejsze wiadomości i korekty.
Licznik = energia zużyta łącznie; szafka IoT = czujnik w szafce technicznej. Odpowiadaj pełnymi słowami."""


def get_settings() -> Settings:
    _load_local_env_file()
    return Settings(
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "pl"),
        whisper_beam_size=_as_int(os.getenv("WHISPER_BEAM_SIZE"), 5),
        # Default off: VAD often wipes short Polish wake+question clips to empty.
        whisper_vad_filter=_as_bool(os.getenv("WHISPER_VAD_FILTER"), False),
        max_upload_mb=_as_int(os.getenv("MAX_UPLOAD_MB"), 25),
        cors_origins=_as_csv_list(os.getenv("CORS_ORIGINS"), DEFAULT_CORS_ORIGINS),
        ollama_url=os.getenv("OLLAMA_URL", "http://192.168.1.173:11434/api/chat"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        # -1 = keep the Student loaded forever (no inference pings needed to keep it warm).
        ollama_keep_alive=_as_keep_alive(os.getenv("OLLAMA_KEEP_ALIVE"), -1),
        # Worst case under the token budget: cold prefill of ~850 tokens at ~7 tok/s + reply.
        ollama_read_timeout_seconds=_as_int(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS"), 180),
        # Hard cap on history messages; the real limit is the token budget below.
        chat_history_max_messages=_as_int(os.getenv("CHAT_HISTORY_MAX_MESSAGES"), 20),
        student_keep_warm_enabled=_as_bool(os.getenv("STUDENT_KEEP_WARM_ENABLED"), True),
        # How often /api/ps is checked (no inference) to re-warm after an Ollama restart.
        student_keep_warm_interval_seconds=_as_int(
            os.getenv("STUDENT_KEEP_WARM_INTERVAL_SECONDS"), 60
        ),
        # After this idle time, frontend should call /api/session/new (advance + warmup).
        session_idle_rotate_seconds=_as_int(
            os.getenv("SESSION_IDLE_ROTATE_SECONDS"), 600
        ),
        student_ctx_window=_as_int(os.getenv("STUDENT_CTX_WINDOW"), 1024),
        student_budget_margin=_as_int(os.getenv("STUDENT_BUDGET_MARGIN"), 32),
        student_num_predict_max=_as_int(os.getenv("STUDENT_NUM_PREDICT_MAX"), 128),
        student_num_predict_min=_as_int(os.getenv("STUDENT_NUM_PREDICT_MIN"), 64),
        # When history must be trimmed, cut down to this share of its budget (hysteresis).
        student_history_trim_ratio=_as_float(os.getenv("STUDENT_HISTORY_TRIM_RATIO"), 0.5),
        # ...but keep at most this much history after a trim: the kept turns are
        # re-prefilled at ~7 tok/s, so this caps the cost of a trim turn (~16 s).
        student_history_keep_max_tokens=_as_int(os.getenv("STUDENT_HISTORY_KEEP_MAX_TOKENS"), 120),
        # Generation options from the thesis (tab:rekomendacja_produkcyjna).
        student_temperature=_as_float(os.getenv("STUDENT_TEMPERATURE"), 0.2),
        student_top_p=_as_float(os.getenv("STUDENT_TOP_P"), 0.9),
        student_repeat_penalty=_as_float(os.getenv("STUDENT_REPEAT_PENALTY"), 1.05),
        student_num_ctx=_as_int(os.getenv("STUDENT_NUM_CTX"), 2048),
        student_num_batch=_as_optional_int(os.getenv("STUDENT_NUM_BATCH")),
        student_num_thread=_as_optional_int(os.getenv("STUDENT_NUM_THREAD")),
        context_service_url=os.getenv("CONTEXT_SERVICE_URL", "http://127.0.0.1:8001"),
        tts_voice=os.getenv("TTS_VOICE", "pl-PL-ZofiaNeural"),
        tts_rate=os.getenv("TTS_RATE", "+0%"),
        tts_max_chars=_as_int(os.getenv("TTS_MAX_CHARS"), 1000),
    )
