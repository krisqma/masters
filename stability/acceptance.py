"""Check the fix's acceptance criteria on post-fix runs (backend 'Chat done' lines + driver + RPi logs).

Usage: python acceptance.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHAT_RE = re.compile(r"^(\S+) INFO \[app\.main\] Chat done: (.*)$")
WARM_RE = re.compile(r"Student warm-up \((\w+)\): .* elapsed=([\d.]+)s prefix_tokens=(\d+)")


def kv(text: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S+)", text))


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main() -> None:
    chats, warms, turns, journal_loads, warnings = [], [], [], 0, []
    for run in map(Path, sys.argv[1:]):
        for line in (run / "backend.log").read_text(errors="ignore").splitlines():
            m = CHAT_RE.match(line)
            if m:
                d = kv(m.group(2))
                d["run"] = run.name
                chats.append(d)
            w = WARM_RE.search(line)
            if w:
                warms.append((w.group(1), float(w.group(2)), int(w.group(3))))
            if "exceeded the" in line or "Traceback" in line:
                warnings.append(line[:200])
        for line in (run / "driver.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("event") == "turn":
                turns.append(d)
        jl = run / "rpi_ollama_journal.log"
        if jl.exists():
            journal_loads += sum("starting runner" in l and "--model" in l for l in jl.read_text(errors="ignore").splitlines())

    done = [c for c in chats if c.get("outcome") == "done"]
    slot = [int(c["slot_tok"]) for c in done if c.get("slot_tok", "None") != "None"]
    est_err = [
        (int(c["est_tok"]) - int(c["prompt_tok"])) / int(c["prompt_tok"])
        for c in done if c.get("prompt_tok", "None") != "None"
    ]
    under = [e for e in est_err if e < 0]
    trim = [c for c in done if c.get("trimmed") == "True"]
    normal = [c for c in done if c.get("trimmed") == "False"]
    e2e = lambda cs: [int(c["e2e_ms"]) / 1000 for c in cs]
    outcomes = {}
    for t in turns:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    deliberate_aborts = sum(1 for t in turns if t["outcome"] == "aborted_by_client")

    rows = [
        ("Tury (driver)", len(turns)),
        ("Wyniki tur", outcomes),
        ("Timeouty klienta", sum(v for k, v in outcomes.items() if "timeout" in k)),
        ("HTTP 5xx / 503 do klienta", sum(v for k, v in outcomes.items() if k.startswith("http_5"))),
        ("Celowe aborty (scenariusz)", deliberate_aborts),
        ("Slot max (tokeny, limit 1024)", max(slot) if slot else None),
        ("Żądania ze slotem > 1024", sum(s > 1024 for s in slot)),
        ("Zwykłe tury: E2E p50 / p95 / max [s]", (pct(e2e(normal), .5), pct(e2e(normal), .95), max(e2e(normal), default=None))),
        ("Tury z przycięciem: liczba, E2E p50 / max [s]", (len(trim), pct(e2e(trim), .5), max(e2e(trim), default=None))),
        ("Rozgrzewanie: liczba, max [s]", (len(warms), max((w[1] for w in warms), default=None))),
        ("Błąd estymacji tokenów: max |%|, niedoszacowania", (round(100 * max(map(abs, est_err), default=0), 1), len(under))),
        ("Przeładowania modelu na RPi (journal 'starting runner')", journal_loads),
        ("Ostrzeżenia / tracebacki", len(warnings)),
    ]
    width = max(len(r[0]) for r in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    for w in warnings[:5]:
        print("  !", w)


if __name__ == "__main__":
    main()
