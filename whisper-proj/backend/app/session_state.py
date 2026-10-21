"""In-process conversation session: pinned Student context + turn ledger.

The backend (not the frontend) is the source of truth for history, so the
Student always receives exactly the text it generated (no trimming/stripping)
and every turn has a known token size for the budget in student_prompt.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .student_prompt import Turn, estimate_message_tokens


@dataclass
class SessionPin:
    static_prefix: str
    dynamic_context: Optional[str]
    source_timestamp: Optional[str]
    degraded: List[str] = field(default_factory=list)
    # Exact prefix tokens (without the generation prompt), from the warm-up's
    # prompt_eval_count; None until measured (an estimate is used meanwhile).
    prefix_tokens: Optional[int] = None


class ConversationSession:
    def __init__(self) -> None:
        self.pin: Optional[SessionPin] = None
        self.turns: List[Turn] = []
        # Turns before this index were trimmed away and are never re-sent.
        self.history_start: int = 0
        self.client_session_id: Optional[str] = None
        # True until the first chat after a backend (re)start: only then may the
        # UI's history seed the ledger (otherwise aborted turns could sneak back in).
        self.awaiting_first_client: bool = True
        self.last_activity_at: float = time.monotonic()

    def touch(self) -> None:
        self.last_activity_at = time.monotonic()

    def reset_history(self) -> None:
        self.turns = []
        self.history_start = 0

    def clear_pin(self) -> None:
        self.pin = None
        self.reset_history()

    def set_pin(
        self,
        *,
        static_prefix: str,
        dynamic_context: Optional[str],
        source_timestamp: Optional[str],
        degraded: Optional[List[str]] = None,
    ) -> SessionPin:
        self.pin = SessionPin(
            static_prefix=static_prefix,
            dynamic_context=dynamic_context,
            source_timestamp=source_timestamp,
            degraded=list(degraded or []),
        )
        self.reset_history()
        self.touch()
        return self.pin

    def bind_client(self, client_session_id: Optional[str]) -> bool:
        """Reset history when the UI starts a new conversation (e.g. page reload).

        Returns True when the history was reset.
        """
        if not client_session_id or client_session_id == self.client_session_id:
            return False
        changed = self.client_session_id is not None
        self.client_session_id = client_session_id
        if changed:
            self.reset_history()
        return changed

    def seed_from_client(self, history: List[Tuple[str, str]]) -> int:
        """Seed the ledger from the UI history, only on the first chat after a backend start.

        Later resets (new session, page reload, aborted turns) never re-seed, so
        an aborted or partial exchange cannot sneak back into the prompt.
        Returns the number of seeded turns.
        """
        first_contact = self.awaiting_first_client
        self.awaiting_first_client = False
        if not first_contact or self.turns or not history:
            return 0
        pending_user: Optional[str] = None
        for role, content in history:
            if role == "user":
                pending_user = content
            elif pending_user is not None:
                tokens = estimate_message_tokens(pending_user) + estimate_message_tokens(content)
                self.turns.append(Turn(pending_user, content, tokens))
                pending_user = None
        return len(self.turns)

    def commit_turn(self, turn: Turn, *, history_start: int) -> None:
        self.history_start = history_start
        self.turns.append(turn)

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity_at
