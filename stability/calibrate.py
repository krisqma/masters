"""Direct RPi calibration: prefill/generation speed and KV prefix-cache behaviour.

Sends production-shaped prompts (same static prefix, facts and wrapper text as
whisper-proj/backend/app/main.py::_build_chat_messages, no `options`, like
production) straight to Ollama on the RPi and records Ollama's own stats.

Usage (from stability/): python calibrate.py OUT.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "context_service"))
from app.context_engine import STUDENT_STATIC_PREFIX, build_dynamic_context  # noqa: E402
from app.data_source import SensorSimulator  # noqa: E402

URL = "http://192.168.1.173:11434/api/chat"
MODEL = "gemma3:4b"
WRAP = (
    "\n\nPowyższy kontekst domu jest dostępny w tej turze. "
    "Liczby i pokoje bierz z bloku „Stan mieszkania”; "
    "Findings (jeśli są) to tylko opcjonalne anomalie. "
    "Cytuj dane tylko, gdy pytanie dotyczy stanu domu. "
    "Uwzględniaj też wcześniejsze wiadomości."
)


def facts() -> str:
    sim = SensorSimulator(ROOT / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv")
    snap = sim.next_snapshot()
    return build_dynamic_context(facts_text=snap.to_student_facts_text(), findings=None)


def call(label: str, messages: list[dict], out: Path, options: dict | None = None) -> dict:
    body = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "30m"}
    if options:
        body["options"] = options
    t0 = time.monotonic()
    r = httpx.post(URL, json=body, timeout=httpx.Timeout(600.0, connect=10.0))
    wall = time.monotonic() - t0
    d = r.json()
    rec = {
        "label": label,
        "wall_s": round(wall, 2),
        "n_messages": len(messages),
        "prompt_chars": sum(len(m["content"]) for m in messages),
        "prompt_eval_count": d.get("prompt_eval_count"),
        "prefill_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 2),
        "eval_count": d.get("eval_count"),
        "gen_s": round((d.get("eval_duration") or 0) / 1e9, 2),
        "load_s": round((d.get("load_duration") or 0) / 1e9, 2),
        "reply": (d.get("message") or {}).get("content", ""),
    }
    if rec["prompt_eval_count"] and rec["prefill_s"]:
        rec["prefill_tok_s"] = round(rec["prompt_eval_count"] / rec["prefill_s"], 1)
    if rec["eval_count"] and rec["gen_s"]:
        rec["gen_tok_s"] = round(rec["eval_count"] / rec["gen_s"], 2)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print({k: v for k, v in rec.items() if k != "reply"}, "| reply:", rec["reply"][:120].replace("\n", " "), flush=True)
    return rec


def main() -> None:
    out = Path(sys.argv[1])
    dyn = facts()
    system = {"role": "system", "content": STUDENT_STATIC_PREFIX}
    ctx = {"role": "user", "content": dyn + WRAP}
    q1 = {"role": "user", "content": "Jaka jest temperatura w łazience?"}
    q2 = {"role": "user", "content": "Cześć powiedz mi coś o temperaturze w mieszkaniu."}

    # 0. unload so we start cold (API call, no config change)
    httpx.post(URL.replace("/api/chat", "/api/generate"), json={"model": MODEL, "keep_alive": 0}, timeout=60)
    time.sleep(3)
    r1 = call("1_cold_full", [system, ctx, q1], out)
    call("2_identical_repeat", [system, ctx, q1], out)
    call("3_same_prefix_new_question", [system, ctx, q2], out)
    # 4. append a turn (history grows): prefix = previous request + reply
    a1 = {"role": "assistant", "content": r1["reply"]}
    call("4_append_turn", [system, ctx, q1, a1, q2], out)
    # 5. keep-warm static ping (what the backend does every 600 s)
    call("5_static_ping", [system, {"role": "user", "content": "ping"}], out, {"num_predict": 1})
    # 6. normal question right after the ping
    call("6_question_after_ping", [system, ctx, q1], out)
    # 7. sliding window: drop the oldest history turn (prefix after ctx changes)
    q3 = {"role": "user", "content": "A w dużym pokoju?"}
    call("7_slid_history", [system, ctx, q2, q3], out)


if __name__ == "__main__":
    main()
