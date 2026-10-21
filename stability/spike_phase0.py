"""Phase 0 spike: verify assumptions of the fix plan directly against the RPi.

1. A prompt-less /api/generate {model, keep_alive, options} refreshes the TTL
   without touching the KV cache (same chat before/after must stay cached).
2. keep_alive: -1 (number) is accepted -> /api/ps expires_at far in the future.
3. Identical options across requests do not reload the model.
4. Prefill speed vs num_batch / num_thread (each variant reloads once).

Usage (from stability/): python spike_phase0.py OUT.jsonl [--skip-batch]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import STUDENT_STATIC_PREFIX, WRAP, facts  # noqa: E402

BASE = "http://192.168.1.173:11434"
MODEL = "gemma3:4b"
OPTS = {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.05, "num_ctx": 2048}


def post(path: str, body: dict, out: Path, label: str) -> dict:
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}{path}", json=body, timeout=httpx.Timeout(900.0, connect=10.0))
    wall = time.monotonic() - t0
    d = r.json() if r.content else {}
    rec = {
        "label": label, "http": r.status_code, "wall_s": round(wall, 2),
        "prompt_eval_count": d.get("prompt_eval_count"),
        "prefill_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 2),
        "eval_count": d.get("eval_count"),
        "load_s": round((d.get("load_duration") or 0) / 1e9, 2),
        "done_reason": d.get("done_reason"),
        "options": body.get("options"), "keep_alive": body.get("keep_alive"),
    }
    if rec["prompt_eval_count"] and rec["prefill_s"]:
        rec["prefill_tok_s"] = round(rec["prompt_eval_count"] / rec["prefill_s"], 1)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(rec, flush=True)
    return rec


def ps(out: Path, label: str) -> dict:
    d = httpx.get(f"{BASE}/api/ps", timeout=10).json()
    m = (d.get("models") or [{}])[0]
    rec = {"label": label, "ps_model": m.get("name"), "expires_at": m.get("expires_at"),
           "context_length": m.get("context_length")}
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
    return rec


def chat(messages: list[dict], options: dict, out: Path, label: str, num_predict: int = 8) -> dict:
    return post("/api/chat", {"model": MODEL, "messages": messages, "stream": False,
                              "keep_alive": -1, "options": {**options, "num_predict": num_predict}},
                out, label)


def main() -> None:
    out = Path(sys.argv[1])
    skip_batch = "--skip-batch" in sys.argv
    system = {"role": "system", "content": STUDENT_STATIC_PREFIX}
    ctx = {"role": "user", "content": facts() + WRAP}
    q = {"role": "user", "content": "Jaka jest temperatura w łazience?"}
    conv = [system, ctx, q]

    # --- 1-3: refresh without prompt, keep_alive -1, stable options
    chat(conv, OPTS, out, "A1_first_with_new_options (reload expected: num_ctx 2048)")
    chat(conv, OPTS, out, "A2_identical_repeat (expect cached)")
    ps(out, "A3_ps_after_keep_alive_-1")
    post("/api/generate", {"model": MODEL, "keep_alive": -1, "options": OPTS}, out,
         "A4_promptless_refresh_same_options")
    chat(conv, OPTS, out, "A5_identical_after_refresh (expect cached, no reload)")
    post("/api/generate", {"model": MODEL, "keep_alive": -1}, out,
         "A6_promptless_refresh_WITHOUT_options (reload?)")
    chat(conv, OPTS, out, "A7_identical_after_refresh_without_options")
    ps(out, "A8_ps_end")

    if skip_batch:
        return
    # --- 4: prefill speed vs num_batch / num_thread (unique prefix defeats cache)
    for extra in ({"num_batch": 512}, {"num_batch": 128}, {"num_batch": 32},
                  {"num_batch": 512, "num_thread": 3}):
        opts = {**OPTS, **extra}
        tag = "_".join(f"{k}{v}" for k, v in extra.items())
        # 1st request after an options change reloads; then measure a fresh prefix
        chat([{"role": "user", "content": f"warm {tag}"}], opts, out, f"B_{tag}_load", num_predict=1)
        uniq = {"role": "system", "content": f"[{tag}] " + STUDENT_STATIC_PREFIX}
        chat([uniq, ctx, q], opts, out, f"B_{tag}_prefill_fresh", num_predict=8)
    # restore the production options so the model is left in the expected state
    chat(conv, OPTS, out, "C_restore_production_options", num_predict=8)


if __name__ == "__main__":
    main()
