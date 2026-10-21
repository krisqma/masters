"""Measure exact gemma3 token counts (via Ollama prompt_eval_count) for the new
compact prefix/facts and a few questions, to calibrate estimate_tokens().

Each message is measured as the difference of two requests that share a cached
prefix, with the production options (so no model reload).

Usage (from stability/): python token_calibration.py OUT.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "context_service"))
from app.context_engine import STUDENT_STATIC_PREFIX, build_dynamic_context  # noqa: E402
from app.data_source import SensorSimulator  # noqa: E402

URL = "http://192.168.1.173:11434/api/chat"
OPTS = {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.05, "num_ctx": 2048, "num_predict": 1}
QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))


def prompt_tokens(messages: list[dict]) -> int:
    r = httpx.post(URL, json={"model": "gemma3:4b", "messages": messages, "stream": False,
                              "keep_alive": -1, "options": OPTS},
                   timeout=httpx.Timeout(600.0, connect=10.0))
    r.raise_for_status()
    return r.json()["prompt_eval_count"]


def est(text: str, div: float) -> int:
    d = sum(c.isdigit() for c in text)
    return d + math.ceil((len(text) - d) / div)


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    system = {"role": "system", "content": STUDENT_STATIC_PREFIX}
    base = prompt_tokens([system])
    rows.append({"kind": "static_prefix_request", "chars": len(STUDENT_STATIC_PREFIX), "tokens": base})
    print(rows[-1], flush=True)

    sim = SensorSimulator(ROOT / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv")
    sim.next_snapshot()
    picks = [1, 3000, 6000, 9000, 12000, 15000, 18000, 21000]
    for row in picks:
        sim._cursor = row
        snap = sim.next_snapshot()
        facts = build_dynamic_context(facts_text=snap.to_student_facts_text(), findings=None)
        total = prompt_tokens([system, {"role": "user", "content": facts}])
        rows.append({"kind": "facts", "row": row, "chars": len(facts),
                     "digits": sum(c.isdigit() for c in facts), "tokens": total - base})
        print(rows[-1], flush=True)
    # questions appended after the last facts block (cached prefix -> fast)
    last_facts = {"role": "user", "content": facts}
    prefix_total = total
    for q in QUESTIONS["replay_c"][:6] + QUESTIONS["replay_a"][:3]:
        t = prompt_tokens([system, last_facts, {"role": "user", "content": q}])
        rows.append({"kind": "question", "chars": len(q), "digits": sum(c.isdigit() for c in q),
                     "tokens": t - prefix_total, "text": q})
        print(rows[-1], flush=True)

    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # how a letters-per-token divisor + per-message overhead would fare
    for div in (2.6, 2.8, 3.0, 3.2):
        errs = []
        for r in rows:
            if r["kind"] in ("facts", "question"):
                text_est = r["digits"] + math.ceil((r["chars"] - r["digits"]) / div) + 6
                errs.append(text_est - r["tokens"])
        print(f"div={div}: estimate-minus-actual min={min(errs)} max={max(errs)}")
    print("static prefix est at 2.6/2.8/3.0/3.2:",
          [est(STUDENT_STATIC_PREFIX, d) for d in (2.6, 2.8, 3.0, 3.2)], "request tokens:", base)


if __name__ == "__main__":
    main()
