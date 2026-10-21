"""Token budget, dynamic history and session ledger for the Student."""

from __future__ import annotations

import ast
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Process env wins over the local .env (config uses setdefault): pin the defaults.
os.environ.update({
    "CHAT_HISTORY_MAX_MESSAGES": "20",
    "OLLAMA_KEEP_ALIVE": "-1",
    "STUDENT_CTX_WINDOW": "1024",
    "STUDENT_BUDGET_MARGIN": "32",
    "STUDENT_NUM_PREDICT_MAX": "128",
    "STUDENT_NUM_PREDICT_MIN": "64",
    "STUDENT_HISTORY_TRIM_RATIO": "0.5",
    "STUDENT_HISTORY_KEEP_MAX_TOKENS": "120",
    "STUDENT_TEMPERATURE": "0.2",
    "STUDENT_NUM_CTX": "2048",
})

from app.config import FALLBACK_STUDENT_STATIC_PREFIX, get_settings  # noqa: E402
from app.session_state import ConversationSession  # noqa: E402
from app.student_prompt import (  # noqa: E402
    Turn,
    build_prefix_messages,
    ensure_prefix_fits,
    estimate_prefix_tokens,
    measured_turn_tokens,
    plan_request,
    prompt_budget,
    student_options,
)

SETTINGS = get_settings()
CONTEXT_SERVICE = ROOT.parents[1] / "context_service"


def test_fallback_prefix_is_byte_identical_to_context_service():
    # Both services name their package `app`, so read the constant from source.
    source = (CONTEXT_SERVICE / "app" / "context_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    value = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "STUDENT_STATIC_PREFIX" for t in node.targets)
    )
    assert FALLBACK_STUDENT_STATIC_PREFIX == value


def _plan(turns, history_start=0, prefix_tokens=600, question="Jaka jest temperatura w łazience?"):
    return plan_request(
        settings=SETTINGS,
        prefix_messages=build_prefix_messages("S", "F"),
        prefix_tokens=prefix_tokens,
        turns=turns,
        history_start=history_start,
        question=question,
    )


def test_request_always_fits_window_randomized():
    rng = random.Random(7)
    window = SETTINGS.student_ctx_window - SETTINGS.student_budget_margin
    for _ in range(2000):
        prefix = rng.randint(300, 700)
        turns = [Turn("q", "a", rng.randint(20, 220)) for _ in range(rng.randint(0, 15))]
        start = rng.randint(0, len(turns))
        question = "x" * rng.randint(5, 300)
        plan = _plan(turns, start, prefix, question)
        assert plan.prompt_tokens_est + plan.num_predict <= window, plan
        assert plan.num_predict >= SETTINGS.student_num_predict_min
        assert plan.history_start >= start  # never re-adds trimmed turns


def test_no_trim_when_history_fits():
    turns = [Turn("q", "a", 40) for _ in range(3)]
    plan = _plan(turns)
    assert plan.kept_turns == 3 and not plan.trimmed
    assert plan.num_predict == SETTINGS.student_num_predict_max


def test_trim_uses_hysteresis():
    prefix = 600
    turns = [Turn("q", "a", 60) for _ in range(6)]  # 360 tokens of history
    plan = _plan(turns, prefix_tokens=prefix)
    assert plan.trimmed
    history_budget = (
        SETTINGS.student_ctx_window - SETTINGS.student_budget_margin
        - SETTINGS.student_num_predict_max - prefix - plan.question_tokens_est
    )
    assert plan.history_tokens <= history_budget * SETTINGS.student_history_trim_ratio
    assert plan.history_tokens <= SETTINGS.student_history_keep_max_tokens
    # Next turn with one more pair appended must not trim again.
    turns.append(Turn("q", "a", 60))
    nxt = _plan(turns, history_start=plan.history_start, prefix_tokens=prefix)
    assert not nxt.trimmed
    assert nxt.kept_turns == plan.kept_turns + 1


def test_messages_are_prefix_then_history_then_question():
    turns = [Turn("u1", "a1\n\n", 30)]
    plan = _plan(turns)
    assert [m["role"] for m in plan.messages] == ["system", "user", "user", "assistant", "user"]
    assert plan.messages[3]["content"] == "a1\n\n"  # raw reply, never stripped


def test_options_identical_for_every_request_kind():
    a = student_options(SETTINGS, 1)
    b = student_options(SETTINGS, 128)
    a.pop("num_predict"), b.pop("num_predict")
    assert a == b
    assert a["num_ctx"] == SETTINGS.student_num_ctx
    assert a["temperature"] == 0.2


def test_ensure_prefix_fits_degrades_in_order():
    small = replace(SETTINGS, student_ctx_window=700)
    dynamic = "\n".join([
        "Stan mieszkania na 2025-12-27 10:10 UTC.",
        "Łazienka: temp. 19.7 °C",
        "Na zewnątrz: temp. 0.9 °C; wilg. 96.2 %",
        "Instalacja ogólna: światła: 0 W (licznik 1 Wh)",
        "Findings (anomalie):",
        "- " + "długie ostrzeżenie " * 20,
    ])
    out, degraded = ensure_prefix_fits("S" * 600, dynamic, small)
    assert degraded[0] == "findings"
    assert "Findings" not in out
    assert "Łazienka" in out


def test_measured_turn_tokens():
    # prompt = prefix 500 + history 100 + question block 15 (incl. generation prompt)
    assert measured_turn_tokens(prompt_eval_count=615, eval_count=40, prefix_tokens=500, history_tokens=100) == 57


def test_session_ledger_and_client_binding():
    s = ConversationSession()
    assert not s.bind_client("a")  # first id: nothing to reset
    s.commit_turn(Turn("q", "r", 10), history_start=0)
    assert not s.bind_client("a")
    assert len(s.turns) == 1
    assert s.bind_client("b")  # page reload / new conversation
    assert s.turns == [] and s.history_start == 0


def test_budget_constants():
    assert prompt_budget(SETTINGS) == 1024 - 32 - 128


def _load_context_service_module(name: str):
    """Load a stdlib-only Context Service module by path (both services use package `app`)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"cs_{name}", CONTEXT_SERVICE / "app" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_csv_row_fits_after_degradation():
    """Real data, worst-case findings: the pinned prefix always leaves room for a turn."""
    ds = _load_context_service_module("data_source")
    sim = ds.SensorSimulator(CONTEXT_SERVICE.parent / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv")
    sim.next_snapshot()
    findings = "- " + ("Duży pokój: VOC 261 ppb, wysoko. " * 6)[:156]
    worst_left = None
    degraded_rows = 0
    for row in range(1, len(sim._rows) + 1, 3):
        sim._cursor = row
        facts = sim.next_snapshot().to_student_facts_text()
        dynamic = "\n".join([facts, "Findings (anomalie):", findings])
        out, degraded = ensure_prefix_fits(FALLBACK_STUDENT_STATIC_PREFIX, dynamic, SETTINGS)
        degraded_rows += bool(degraded)
        left = prompt_budget(SETTINGS) - estimate_prefix_tokens(FALLBACK_STUDENT_STATIC_PREFIX, out)
        worst_left = left if worst_left is None else min(worst_left, left)
        nothing_left_to_drop = not any(
            line.startswith(("Findings", "Instalacja ogólna:", "Na zewnątrz:")) or "(licznik " in line
            for line in out.split("\n")
        )
        assert left >= 120 or nothing_left_to_drop, (row, left, degraded)
        # Hard guarantee: even the largest row still fits a long question + a reply.
        prefix_tokens = estimate_prefix_tokens(FALLBACK_STUDENT_STATIC_PREFIX, out)
        plan = plan_request(
            settings=SETTINGS,
            prefix_messages=build_prefix_messages(FALLBACK_STUDENT_STATIC_PREFIX, out),
            prefix_tokens=prefix_tokens,
            turns=[Turn("q", "a", 60)] * 4,
            history_start=0,
            question="x" * 150,
        )
        window = SETTINGS.student_ctx_window - SETTINGS.student_budget_margin
        assert plan.prompt_tokens_est + plan.num_predict <= window, (row, plan.notes)
        assert plan.num_predict >= SETTINGS.student_num_predict_min
    assert worst_left is not None and worst_left >= 0, worst_left


def test_history_seeded_only_on_first_contact_after_restart():
    s = ConversationSession()
    hist = [("user", "q1"), ("assistant", "a1"), ("user", "q2 bez odpowiedzi")]
    assert s.seed_from_client(hist) == 1  # backend restart: seeded once, dangling question dropped
    s.reset_history()
    assert s.seed_from_client(hist) == 0  # later resets (new session, abort) never re-seed
    assert s.turns == []
