"""Join all logs of a stability run into per-request / per-turn tables + plots.

Usage: python analyze.py RUN_DIR      -> RUN_DIR/analysis.md, RUN_DIR/timeline.png

Key derived signals per Ollama request (single KV slot on the RPi):
- prefill_s: Ollama's prompt_eval_duration. NOTE: prompt_eval_count reports the
  whole prompt even when cached, so prefill_s is the cache-miss signal.
- lcp_chars: common prefix (chars) of this request's rendered messages vs the
  previous request -> where the cache *should* be reusable from.
- prev_total_tok: prompt+generated tokens of the previous request (did the
  conversation in the slot exceed gemma3's 1024-token sliding window?).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SWA_WINDOW = 1024


def ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def render(messages: list[dict[str, str]]) -> str:
    return "".join(f"<{m.get('role')}>{m.get('content', '')}" for m in messages)


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_backend(path: Path) -> list[dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+) (\w+) \[([^\]]+)\] (.*)", line)
        if not m:
            continue
        t, _lvl, logger, msg = m.groups()
        if "uvicorn.access" in logger and "/api/health" in msg:
            continue
        if any(k in msg for k in ("Chat done", "keep-warm", "Session new", "Context advanced",
                                  "Traceback", "ERROR", "failed", "skipped", " 503", " 504", " 502")):
            events.append({"ts": ts(t + "+00:00"), "logger": logger, "msg": msg})
    return events


def main() -> None:
    run = Path(sys.argv[1])
    proxy = [r for r in jsonl(run / "proxy.jsonl") if r.get("event") == "end"]
    proxy.sort(key=lambda r: r["start"])
    driver = jsonl(run / "driver.jsonl")
    metrics = jsonl(run / "rpi_metrics.jsonl")
    backend = load_backend(run / "backend.log")

    # ---- per Ollama request
    prev = None
    rows = []
    for r in proxy:
        rendered = render(r.get("messages") or [])
        row = {
            "id": r["id"], "start": r["start"][11:19], "kind": r["kind"],
            "n_msgs": r.get("n_messages"), "prompt_chars": r.get("prompt_chars"),
            "prompt_tok": r.get("prompt_eval_count"), "gen_tok": r.get("eval_count"),
            "prefill_s": r.get("prefill_s"), "gen_s": r.get("gen_s"),
            "elapsed_s": r.get("elapsed_s"), "outcome": r.get("outcome"),
            "in_flight": r.get("in_flight_at_start"),
        }
        if prev is not None:
            row["lcp_chars"] = lcp(rendered, prev["_rendered"])
            row["new_chars"] = len(rendered) - row["lcp_chars"]
            row["prev_total_tok"] = (prev.get("prompt_eval_count") or 0) + (prev.get("eval_count") or 0)
            row["prev_over_swa"] = row["prev_total_tok"] > SWA_WINDOW
        r["_rendered"] = rendered
        rows.append(row)
        prev = r

    # ---- per driver turn, matched to the proxy chat request carrying the same question
    turns = [d for d in driver if d.get("event") == "turn"]
    chat_reqs = [r for r in proxy if r["kind"] == "chat"]
    used: set[int] = set()
    for t in turns:
        t_end = ts(t["ts"])
        cands = [
            r for r in chat_reqs
            if r["id"] not in used and (r.get("messages") or [{}])[-1].get("content") == t["text"]
            and abs((ts(r["end"]) - t_end).total_seconds()) < 130
        ]
        if cands:
            best = min(cands, key=lambda r: abs((ts(r["end"]) - t_end).total_seconds()))
            used.add(best["id"])
            t["proxy_id"] = best["id"]
            match = next(x for x in rows if x["id"] == best["id"])
            t.update({k: match.get(k) for k in ("prefill_s", "gen_s", "prompt_tok", "gen_tok",
                                                  "new_chars", "prev_total_tok", "prev_over_swa")})
        # RPi temp at turn end
        near = [m for m in metrics if isinstance(m.get("m"), dict) and "cpu_temp_c" in m["m"]
                and abs((ts(m["ts"] + "+00:00") - t_end).total_seconds()) < 10]
        if near:
            t["temp_c"] = near[-1]["m"]["cpu_temp_c"]

    # ---- write markdown
    lines = [f"# Analiza runu `{run.name}`", ""]
    lines += ["## Tury użytkownika (driver)", "",
              "| czas | s | tura | label | wynik | czekanie | TTFC | E2E | hist | prefill | gen tok | gen s | prev tot tok | >1024 | temp |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in turns:
        lines.append(
            f"| {t['ts'][11:19]} | {t['session']} | {t['turn']} | {t.get('label','')} | {t['outcome']} "
            f"| {t.get('blocked_s')} | {t.get('ttfc_s')} | {t.get('e2e_s')} | {t.get('history_msgs')} "
            f"| {t.get('prefill_s')} | {t.get('gen_tok')} | {t.get('gen_s')} | {t.get('prev_total_tok')} "
            f"| {'TAK' if t.get('prev_over_swa') else ''} | {t.get('temp_c','')} |"
        )
    lines += ["", "## Żądania do Ollamy na RPi (proxy)", "",
              "| id | start | rodzaj | msgs | prompt tok | nowe znaki vs poprzednie | prev total tok | >1024 | prefill s | gen tok | gen s | razem s | wynik | w locie |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['start']} | {r['kind']} | {r['n_msgs']} | {r['prompt_tok']} | {r.get('new_chars','')} "
            f"| {r.get('prev_total_tok','')} | {'TAK' if r.get('prev_over_swa') else ''} | {r['prefill_s']} "
            f"| {r['gen_tok']} | {r['gen_s']} | {r['elapsed_s']} | {r['outcome']} | {r['in_flight']} |"
        )
    other = [d for d in driver if d.get("event") in ("session_new", "health_change", "driver_task_crashed", "mark")]
    lines += ["", "## Zdarzenia drivera", ""] + [
        f"- {d['ts'][11:19]} {d['event']} " + json.dumps({k: v for k, v in d.items() if k not in ('ts', 'event')}, ensure_ascii=False)
        for d in other]
    lines += ["", "## Zdarzenia backendu", ""] + [f"- {e['ts'].strftime('%H:%M:%S')} {e['msg'][:220]}" for e in backend]

    # ping / wifi summary
    ping_log = run / "ping.log"
    if ping_log.exists():
        txt = ping_log.read_text(errors="ignore").splitlines()
        rtts = [float(m.group(1)) for l in txt if (m := re.search(r"time=([\d.]+) ms", l))]
        lost = [l[:19] for l in txt if "timeout" in l.lower() or "no route" in l.lower() or "down" in l.lower()]
        if rtts:
            rtts_sorted = sorted(rtts)
            lines += ["", "## Wi-Fi (ping Mac -> RPi)", "",
                      f"- odpowiedzi: {len(rtts)}, utracone: {len(lost)}",
                      f"- RTT mediana {rtts_sorted[len(rtts)//2]:.1f} ms, p95 {rtts_sorted[int(len(rtts)*0.95)]:.1f} ms, max {max(rtts):.1f} ms"]
            if lost:
                lines.append(f"- utraty (pierwsze 20): {', '.join(lost[:20])}")
    temps = [m["m"]["cpu_temp_c"] for m in metrics if isinstance(m.get("m"), dict) and "cpu_temp_c" in m["m"]]
    thr = [m for m in metrics if isinstance(m.get("m"), dict) and m["m"].get("throttling")]
    if temps:
        lines += ["", "## RPi termika", "", f"- temp min {min(temps)} / max {max(temps)} °C, próbki z throttlingiem: {len(thr)}"]
    (run / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6))
        for kind, color, marker in (("chat", "tab:blue", "o"), ("ping_static", "tab:orange", "^"), ("ping_facts", "tab:red", "s")):
            xs = [ts(r["start"]) for r in proxy if r["kind"] == kind and r.get("prefill_s") is not None]
            ys = [r["prefill_s"] for r in proxy if r["kind"] == kind and r.get("prefill_s") is not None]
            ax.scatter(xs, ys, c=color, marker=marker, label=f"prefill {kind}", zorder=3)
        tx = [ts(t["ts"]) for t in turns]
        ty = [t.get("e2e_s") or 0 for t in turns]
        ax.plot(tx, ty, "-", c="tab:green", alpha=0.6, label="E2E tury (driver)")
        for t in turns:
            if t["outcome"] != "ok":
                ax.annotate(t["outcome"], (ts(t["ts"]), t.get("e2e_s") or 0), fontsize=7, color="red")
        for d in driver:
            if d.get("event") == "session_new":
                ax.axvline(ts(d["ts"]), color="gray", ls="--", lw=0.8)
        ax.set_ylabel("sekundy")
        ax.legend(loc="upper left")
        if temps:
            ax2 = ax.twinx()
            ax2.plot([ts(m["ts"] + "+00:00") for m in metrics if isinstance(m.get("m"), dict) and "cpu_temp_c" in m["m"]],
                     temps, c="tab:purple", alpha=0.3, label="temp RPi")
            ax2.set_ylabel("°C")
        ax.set_title(f"Wilga stability run {run.name}: prefill per request, E2E per turn (szare linie = session/new)")
        fig.tight_layout()
        fig.savefig(run / "timeline.png", dpi=110)
    except Exception as error:  # plotting is optional
        print("plot failed:", error)
    print((run / "analysis.md").read_text(encoding="utf-8")[:4000])


if __name__ == "__main__":
    main()
