"""Token-budgeted prompts for the Student (gemma3:4b on the RPi).

Why: gemma3 has a 1024-token sliding window. Once the single Ollama KV slot holds
more than that (prompt + generated), any prompt that diverges from the cached
tokens forces a full recompute (~7 tok/s on the RPi -> minutes). Below the
window, diverging is a cheap rollback. So every request is planned to satisfy

    prompt_tokens + num_predict + margin <= window

and history is dynamic: as many recent turns as fit, trimmed with hysteresis.
See stability/RAPORT.md for the measurements behind this.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import Settings

# Template tokens around one message: "<start_of_turn>user\n" ... "<end_of_turn>\n".
MESSAGE_OVERHEAD_TOKENS = 6
# "<start_of_turn>model\n" appended after the last user message.
GENERATION_PROMPT_TOKENS = 3
# Rooms dropped (in this order) when the pinned prefix alone exceeds the budget.
DEGRADE_LINE_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("findings", "Findings (anomalie):"),
    ("counters", " (licznik "),  # cumulative energy counters; current power stays
    ("instalacja", "Instalacja ogólna:"),
    ("outside", "Na zewnątrz:"),
)
COUNTER_RE = re.compile(r" \(licznik [^)]*\)")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Polish text, calibrated on the RPi.

    Gemma splits numbers into single digits. Sensor facts (numbers, units,
    abbreviations) run ~2.0-2.2 chars/token for the rest; plain Polish ~3.5.
    Measured: stability/runs/phase0/token_calibration.jsonl.
    """
    if not text:
        return 0
    digits = sum(ch.isdigit() for ch in text)
    dense = digits / len(text) > 0.06
    return digits + math.ceil((len(text) - digits) / (1.95 if dense else 2.8))


def estimate_message_tokens(text: str) -> int:
    return estimate_tokens(text) + MESSAGE_OVERHEAD_TOKENS


def student_options(settings: Settings, num_predict: int) -> Dict[str, Union[int, float]]:
    """Single source of options for EVERY Student request (chat, warm-up, refresh).

    Load-time options (num_ctx, num_batch, num_thread) must never differ between
    requests, otherwise Ollama reloads the model and drops the KV cache.
    """
    options: Dict[str, Union[int, float]] = {
        "temperature": settings.student_temperature,
        "top_p": settings.student_top_p,
        "repeat_penalty": settings.student_repeat_penalty,
        "num_ctx": settings.student_num_ctx,
        "num_predict": num_predict,
    }
    if settings.student_num_batch:
        options["num_batch"] = settings.student_num_batch
    if settings.student_num_thread:
        options["num_thread"] = settings.student_num_thread
    return options


def build_prefix_messages(static_prefix: str, dynamic_context: Optional[str]) -> List[Dict[str, str]]:
    """Session prefix shared byte-for-byte by the warm-up and every chat request."""
    messages = [{"role": "system", "content": static_prefix}]
    if dynamic_context:
        messages.append({"role": "user", "content": dynamic_context})
    return messages


def prompt_budget(settings: Settings) -> int:
    """Max prompt tokens so that prompt + max reply + margin fit in the window."""
    return settings.student_ctx_window - settings.student_budget_margin - settings.student_num_predict_max


def estimate_prefix_tokens(static_prefix: str, dynamic_context: Optional[str]) -> int:
    return sum(
        estimate_message_tokens(m["content"])
        for m in build_prefix_messages(static_prefix, dynamic_context)
    )


def ensure_prefix_fits(
    static_prefix: str,
    dynamic_context: Optional[str],
    settings: Settings,
    *,
    min_history_tokens: int = 120,
) -> Tuple[Optional[str], List[str]]:
    """Shrink the facts block until prefix + a minimal turn fits the budget.

    Drops, in order: Teacher findings, cumulative energy counters, whole-house
    meters, outdoor readings.
    Returns (dynamic_context, degraded_parts). Never touches the static prefix.
    """
    degraded: List[str] = []
    limit = prompt_budget(settings) - min_history_tokens
    current = dynamic_context
    for name, line_prefix in DEGRADE_LINE_PREFIXES:
        if current is None or estimate_prefix_tokens(static_prefix, current) <= limit:
            break
        lines = current.split("\n")
        if name == "findings" and line_prefix in lines:
            current = "\n".join(lines[: lines.index(line_prefix)])
        elif name == "counters":
            stripped = COUNTER_RE.sub("", current)
            if stripped == current:
                continue
            current = stripped
        else:
            kept = [ln for ln in lines if not ln.startswith(line_prefix)]
            if len(kept) == len(lines):
                continue
            current = "\n".join(kept)
        degraded.append(name)
    return current, degraded


@dataclass
class Turn:
    """One completed exchange, stored exactly as sent and as generated."""

    user: str
    reply: str
    tokens: int  # exact when measured from Ollama stats, else estimated
    exact: bool = False


@dataclass
class RequestPlan:
    messages: List[Dict[str, str]]
    num_predict: int
    prompt_tokens_est: int
    kept_turns: int
    history_start: int
    trimmed: bool
    question_tokens_est: int
    history_tokens: int = 0
    notes: List[str] = field(default_factory=list)


def plan_request(
    *,
    settings: Settings,
    prefix_messages: Sequence[Dict[str, str]],
    prefix_tokens: int,
    turns: Sequence[Turn],
    history_start: int,
    question: str,
) -> RequestPlan:
    """Pick the history window and num_predict so the request fits the window.

    History is dynamic: all turns from `history_start` that fit. When they no
    longer fit, older turns are dropped until history uses at most
    `student_history_trim_ratio` of its budget (and at most
    `student_history_keep_max_tokens`), so trims (each costs a recompute of the
    kept history on the RPi) happen every few turns and stay cheap.
    """
    window_left = settings.student_ctx_window - settings.student_budget_margin
    question_tokens = estimate_message_tokens(question) + GENERATION_PROMPT_TOKENS
    history_budget = window_left - settings.student_num_predict_max - prefix_tokens - question_tokens

    start = max(0, min(history_start, len(turns)))
    if settings.chat_history_max_messages > 0:
        start = max(start, len(turns) - settings.chat_history_max_messages // 2)
    kept = list(turns[start:])
    history_tokens = sum(t.tokens for t in kept)
    trimmed = start != history_start
    notes: List[str] = []

    if history_tokens > max(0, history_budget):
        target = max(
            0,
            min(
                int(history_budget * settings.student_history_trim_ratio),
                settings.student_history_keep_max_tokens,
            ),
        )
        while kept and history_tokens > target:
            history_tokens -= kept[0].tokens
            kept.pop(0)
            start += 1
        trimmed = True

    prompt_tokens = prefix_tokens + history_tokens + question_tokens
    num_predict = min(settings.student_num_predict_max, window_left - prompt_tokens)
    if num_predict < settings.student_num_predict_min:
        # Prefix is huge (should have been degraded at pin time): drop all history.
        start += len(kept)
        kept = []
        history_tokens = 0
        prompt_tokens = prefix_tokens + question_tokens
        num_predict = max(
            settings.student_num_predict_min,
            min(settings.student_num_predict_max, window_left - prompt_tokens),
        )
        trimmed = True
        notes.append("history_dropped_for_reply_budget")
        if prompt_tokens + num_predict > window_left:
            notes.append("over_budget")

    messages = list(prefix_messages)
    for turn in kept:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.reply})
    messages.append({"role": "user", "content": question})

    return RequestPlan(
        messages=messages,
        num_predict=num_predict,
        prompt_tokens_est=prompt_tokens,
        kept_turns=len(kept),
        history_start=start,
        trimmed=trimmed,
        question_tokens_est=question_tokens,
        history_tokens=history_tokens,
        notes=notes,
    )


def measured_turn_tokens(
    *,
    prompt_eval_count: int,
    eval_count: int,
    prefix_tokens: int,
    history_tokens: int,
) -> int:
    """Exact size of the new (question, reply) pair from Ollama's own counts.

    prompt_eval_count = prefix + history + question block + generation prompt.
    The stored reply is re-rendered as "<start_of_turn>model\\n{reply}<end_of_turn>\\n".
    """
    question_block = prompt_eval_count - prefix_tokens - history_tokens
    return max(1, question_block + eval_count + 2)
