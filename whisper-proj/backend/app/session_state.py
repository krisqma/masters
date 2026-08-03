"""In-process conversation session pin for Student context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionPin:
    static_prefix: str
    dynamic_context: Optional[str]
    source_timestamp: Optional[str]


class ConversationSession:
    def __init__(self) -> None:
        self.pin: Optional[SessionPin] = None
        self.last_activity_at: float = time.monotonic()

    def touch(self) -> None:
        self.last_activity_at = time.monotonic()

    def clear_pin(self) -> None:
        self.pin = None

    def set_pin(
        self,
        *,
        static_prefix: str,
        dynamic_context: Optional[str],
        source_timestamp: Optional[str],
    ) -> None:
        self.pin = SessionPin(
            static_prefix=static_prefix,
            dynamic_context=dynamic_context,
            source_timestamp=source_timestamp,
        )
        self.touch()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity_at
