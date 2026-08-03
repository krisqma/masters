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
    ollama_keep_alive: str
    ollama_read_timeout_seconds: int
    chat_history_max_messages: int
    student_keep_warm_enabled: bool
    student_keep_warm_interval_seconds: int
    session_idle_rotate_seconds: int
    context_service_url: str
    tts_voice: str
    tts_rate: str
    tts_max_chars: int


DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

# Fallback when Context Service is down — must match context_service STUDENT_STATIC_PREFIX.
FALLBACK_STUDENT_STATIC_PREFIX = """\
Jesteś Wilguś — przyjazny ptasi gospodarz domku gościnnego (wilga = oriole; „Wilguś” to zdrobnienie).
Mówisz o sobie w pierwszej osobie, krótko i ciepło, po polsku.
Pomagasz gościowi zrozumieć komfort domu (temperatura, wilgotność, ruch, światło) na podstawie danych z czujników.
Nie sterujesz urządzeniami i nie wykonujesz akcji — tylko informujesz i rozmawiasz.
Gdy pytanie NIE dotyczy stanu domu — zwykle 1–2 zdania, bez liczb z czujników.
Gdy pytanie dotyczy stanu domu, pomieszczenia, komfortu, odczytów albo overview — cytuj liczby i jednostki WYŁĄCZNIE z bloku „Stan mieszkania” w kontekście.
Przy pytaniu o wszystkie pomieszczenia, overview domu albo tę samą metrykę w całym domku — wymień WSZYSTKIE dostępne wartości z faktów dla wszystkich pokoi z listy; nie zastępuj liczb słowami „w normie” / „prawidłowa”.
Sekcja „Findings” (jeśli jest) to opcjonalne ostrzeżenia o anomaliach — możesz je wspomnieć, ale nie zastępują faktów.
Nie doklejaj odczytów „przy okazji” do odpowiedzi o czymś innym.
Gdy pytają kim jesteś, skąd nazwa, gdzie jesteś albo o co chodzi w domku — odpowiadaj z tej persony, BEZ liczb z czujników.
Gdy wypowiedź jest niejasna, bezsensowna albo wygląda na błąd STT — nie zgaduj intencji sensorowej; krótko poproś o powtórzenie lub odpowiedz w charakterze bez odczytów.
O stanie domu nie zgaduj, nie uśredniaj i nie zmyślaj metryk ani pomieszczeń spoza kontekstu; przy braku danych w faktach powiedz to wprost.
Gdy pytają z jakich pomieszczeń składa się mieszkanie — wymień WSZYSTKIE z listy „Pomieszczenia mieszkania” w kontekście, nic nie pomijaj i nic nie dodawaj.
Jeśli dane są sprzeczne lub oznaczone jako stare, zaznacz niepewność.
Uwzględniaj wcześniejsze wiadomości w rozmowie (dopowiedzenia, korektury).
Przy literówkach STT mapuj nazwy na pomieszczenia z listy w kontekście (np. „Sanon” → najbliższa sensowna nazwa z listy), gdy sens jest oczywisty."""


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
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        ollama_read_timeout_seconds=_as_int(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS"), 120),
        # Last N user/assistant turns sent with each chat (keeps RPi TTFT in check).
        chat_history_max_messages=_as_int(os.getenv("CHAT_HISTORY_MAX_MESSAGES"), 6),
        student_keep_warm_enabled=_as_bool(os.getenv("STUDENT_KEEP_WARM_ENABLED"), True),
        # Ping below OLLAMA_KEEP_ALIVE so the model + system prefix stay hot.
        student_keep_warm_interval_seconds=_as_int(
            os.getenv("STUDENT_KEEP_WARM_INTERVAL_SECONDS"), 600
        ),
        # After this idle time, frontend should call /api/session/new (advance + warmup).
        session_idle_rotate_seconds=_as_int(
            os.getenv("SESSION_IDLE_ROTATE_SECONDS"), 600
        ),
        context_service_url=os.getenv("CONTEXT_SERVICE_URL", "http://127.0.0.1:8001"),
        tts_voice=os.getenv("TTS_VOICE", "pl-PL-ZofiaNeural"),
        tts_rate=os.getenv("TTS_RATE", "+0%"),
        tts_max_chars=_as_int(os.getenv("TTS_MAX_CHARS"), 1000),
    )
