from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import Settings

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError as import_error:  # pragma: no cover - handled at runtime
    WhisperModel = None  # type: ignore[assignment]
    FAST_WHISPER_IMPORT_ERROR = import_error
else:
    FAST_WHISPER_IMPORT_ERROR = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration_seconds: Optional[float]


class WhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                f"Failed to import faster-whisper runtime dependencies: {FAST_WHISPER_IMPORT_ERROR}"
            ) from FAST_WHISPER_IMPORT_ERROR

        self._settings = settings
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def _run(self, file_path: str, *, vad_filter: bool) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            file_path,
            language=self._settings.whisper_language,
            beam_size=self._settings.whisper_beam_size,
            vad_filter=vad_filter,
        )

        chunks = [segment.text.strip() for segment in segments]
        text = " ".join(chunk for chunk in chunks if chunk).strip()

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=getattr(info, "duration", None),
        )

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        use_vad = self._settings.whisper_vad_filter
        result = self._run(file_path, vad_filter=use_vad)

        # Short / quiet wake-word clips often become empty with VAD — retry once without it.
        if not result.text and use_vad:
            logger.info(
                "Empty transcript with VAD (duration=%s); retrying without VAD.",
                result.duration_seconds,
            )
            result = self._run(file_path, vad_filter=False)

        return result
