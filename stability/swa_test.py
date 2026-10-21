"""Controlled test: does the KV prefix cache survive once a conversation exceeds
gemma3's 1024-token sliding window?

For several conversation sizes: send a production-shaped chat (system + facts +
padded history + question, fixed num_predict), then the backend's static
keep-warm ping (shares the 596-token system prefix). Record the ping prefill.
Hypothesis: total (prompt + generated) <= ~1024 -> ping prefill ~1 s (cache
rolled back); > 1024 -> full recompute (~70 s), because the sliding-window
layers no longer hold the earliest positions.

Usage (from stability/): python swa_test.py OUT.jsonl
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import STUDENT_STATIC_PREFIX, URL, MODEL, WRAP, call, facts  # noqa: E402

PAIRS = [
    ("Jaka jest temperatura w łazience?", "Temperatura w łazience wynosi 15,4°C."),
    ("A w dużym pokoju?", "W dużym pokoju jest 23,0°C, a oświetlenie to 76 lx."),
    ("Czy w małym pokoju ktoś jest?", "Nie, czujnik obecności w małym pokoju nie wykrywa nikogo."),
    ("Jaka jest wilgotność w łazience?", "Wilgotność w łazience wynosi 61,8 %, to dość wysoko."),
    ("A ciśnienie?", "Ciśnienie w łazience to 1010,3 hPa, w normie dla tej pory."),
    ("Powiedz coś o sobie.", "Jestem Wilguś, ptasi gospodarz tego domku. Pomagam zrozumieć, co dzieje się w środku."),
]


def main() -> None:
    out = Path(sys.argv[1])
    system = {"role": "system", "content": STUDENT_STATIC_PREFIX}
    ctx = {"role": "user", "content": facts() + WRAP}
    ping = [system, {"role": "user", "content": "ping"}]
    q = {"role": "user", "content": "Dziękuję, to wszystko."}
    # history sizes (number of user/assistant pairs) to walk across the 1024 boundary
    for n_pairs in (0, 1, 2, 3, 4, 6):
        hist = []
        for u, a in PAIRS[:n_pairs]:
            hist += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        # reset to a known state: cache = static ping prefix only
        call(f"reset_ping_before_{n_pairs}", ping, out, {"num_predict": 1})
        r = call(f"conv_{n_pairs}pairs", [system, ctx, *hist, q], out, {"num_predict": 12})
        total = (r["prompt_eval_count"] or 0) + (r["eval_count"] or 0)
        print(f"   -> conversation total tokens = {total}", flush=True)
        p = call(f"ping_after_{n_pairs}pairs_total{total}", ping, out, {"num_predict": 1})
        verdict = "CACHE REUSED" if p["prefill_s"] < 5 else "FULL RECOMPUTE"
        print(f"   => total={total}: ping prefill {p['prefill_s']} s  [{verdict}]", flush=True)


if __name__ == "__main__":
    main()
