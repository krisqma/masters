"""Quality A/B: old Student prompt (git HEAD, before the fix) vs the new compact one.

Same model on the RPi, same generation options (thesis), same CSV rows and
questions (from the real 26.07/30.07 conversations). Only the prompt differs:
  OLD = old static prefix + old verbose facts + old wrapper text
  NEW = compact static prefix + compact facts (device chip temperatures dropped)

Golden answers: Teacher (qwen2.5 on the Mac) from the full verbose facts.
Judge: benchmark/evaluate.py (LLM via OpenRouter), context = full verbose facts
for both variants, so a correct statement counts no matter which prompt saw it.
Deterministic check: every number in an answer must appear in the facts.

Usage (from stability/): python quality_ab.py OUT_DIR
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "context_service"))
sys.path.insert(0, str(ROOT / "benchmark"))
from app.context_engine import STUDENT_STATIC_PREFIX as NEW_PREFIX, build_dynamic_context  # noqa: E402
from app.data_source import SensorSimulator  # noqa: E402

RPI = "http://192.168.1.173:11434/api/chat"
# Name under which the CURRENT prompt code is recorded (e.g. "new2" after a prompt fix).
NEW_NAME = os.environ.get("AB_NEW_NAME", "new")
TEACHER = "http://localhost:11434/api/chat"
# Same options as whisper-proj/backend/app/student_prompt.student_options (no reload).
OPTS = {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.05, "num_ctx": 2048, "num_predict": 128}
ROWS = [1200, 6000, 12000, 18000, 23000]
QUESTIONS = [
    ("status_single", "Jaka jest temperatura w łazience?"),
    ("status_single", "Czy w małym pokoju ktoś jest?"),
    ("status_all_rooms", "Cześć powiedz mi coś o temperaturze w mieszkaniu."),
    ("status_all_rooms", "Ok, a jakie są ciśnienia w innych pomieszczeniach w takim razie? Daj mi ciśnienie dla każdego pomieszczenia, poproszę."),
    ("anomalies", "Ok, jakieś anomalie, coś dziwnego się dzieje teraz w danych, czy sytuacja jest spokojna? Oceń."),
    ("persona", "Coś o tym, kim jesteś, dlaczego jesteś takim ptakiem, gdzie ty się znajdujesz, o co chodzi w ogóle, skąd nazywa Wilgo i Wilga, co tutaj dzieje, co ja robię w tym danku."),
    ("rooms", "Powiedz mi, z jakich pomieszczeń się składa to moje mieszkanie?"),
    ("stt_garbage", "Glocka! Zostawiony serce! No, no, no!"),
]
OLD_INTRO = "Użyj poniższych danych tylko, gdy pytanie dotyczy stanu domu. W przeciwnym razie ich nie cytuj."
OLD_WRAP = (
    "\n\nPowyższy kontekst domu jest dostępny w tej turze. "
    "Liczby i pokoje bierz z bloku „Stan mieszkania”; "
    "Findings (jeśli są) to tylko opcjonalne anomalie. "
    "Cytuj dane tylko, gdy pytanie dotyczy stanu domu. "
    "Uwzględniaj też wcześniejsze wiadomości."
)
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def old_prefix() -> str:
    src = subprocess.run(
        ["git", "show", "HEAD:context_service/app/context_engine.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STUDENT_STATIC_PREFIX" for t in node.targets):
            return node.value.value
    raise SystemExit("old prefix not found")


def chat(url: str, model: str, messages: list[dict], options: dict | None) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": -1}
    if options:
        body["options"] = options
    t0 = time.monotonic()
    r = httpx.post(url, json=body, timeout=httpx.Timeout(900.0, connect=10.0))
    r.raise_for_status()
    d = r.json()
    d["_wall_s"] = round(time.monotonic() - t0, 2)
    return d


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM_RE.findall(text):
        v = m.replace(",", ".")
        try:
            f = float(v)
        except ValueError:
            continue
        out.add(f"{f:.1f}")
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "ab_answers.jsonl"
    done = set()
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.add((d["row"], d["qi"], d["variant"]))

    old_static = old_prefix()
    sim = SensorSimulator(ROOT / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv")
    sim.next_snapshot()

    for row in ROWS:
        sim._cursor = row
        snap = sim.next_snapshot()
        verbose = snap.to_student_facts_text_verbose()
        variants = {
            NEW_NAME: [{"role": "system", "content": NEW_PREFIX},
                    {"role": "user", "content": build_dynamic_context(facts_text=snap.to_student_facts_text(), findings=None)}],
            "old": [{"role": "system", "content": old_static},
                    {"role": "user", "content": f"{OLD_INTRO}\n{verbose}{OLD_WRAP}"}],
        }
        for variant in (NEW_NAME, "old"):  # new first: its prefix stays cached across questions
            for qi, (category, question) in enumerate(QUESTIONS):
                if (row, qi, variant) in done:
                    continue
                d = chat(RPI, "gemma3:4b", variants[variant] + [{"role": "user", "content": question}], OPTS)
                rec = {
                    "row": row, "qi": qi, "variant": variant, "category": category, "question": question,
                    "answer": d["message"]["content"], "wall_s": d["_wall_s"],
                    "prompt_tokens": d.get("prompt_eval_count"), "eval_tokens": d.get("eval_count"),
                    "prefill_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 1),
                    "done_reason": d.get("done_reason"), "facts_verbose": verbose,
                }
                with answers_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{time.strftime('%H:%M:%S')} row={row} q={qi} {variant} {rec['wall_s']}s "
                      f"prompt={rec['prompt_tokens']} | {rec['answer'][:90]!r}", flush=True)

    # ---- golden answers (Teacher on the Mac) + judge
    from evaluate import evaluate_response  # noqa: E402  (loads benchmark/.env)

    golden_path = out_dir / "ab_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.exists() else {}
    answers = [json.loads(l) for l in answers_path.read_text(encoding="utf-8").splitlines()]
    for a in answers:
        key = f"{a['row']}:{a['qi']}"
        if key in golden:
            continue
        g = chat(TEACHER, "qwen2.5", [
            {"role": "system", "content": (
                "Napisz wzorcową, krótką (1-3 zdania) odpowiedź asystenta głosowego Wilguś "
                "(przyjazny ptasi gospodarz domku) na pytanie gościa, po polsku. "
                "Liczby tylko z danych poniżej. Pytania o personę lub niezwiązane z domem: bez liczb. "
                "Niezrozumiała wypowiedź: poproś o powtórzenie.\n\nDane:\n" + a["facts_verbose"])},
            {"role": "user", "content": a["question"]},
        ], {"temperature": 0.0})
        golden[key] = g["message"]["content"]
        golden_path.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")

    judged_path = out_dir / "ab_judged.jsonl"
    judged_keys = set()
    if judged_path.exists():
        for line in judged_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            judged_keys.add((d["row"], d["qi"], d["variant"]))
    for a in answers:
        if (a["row"], a["qi"], a["variant"]) in judged_keys:
            continue
        context = ("Asystent to Wilguś, ptasi gospodarz domku gościnnego. Dane z czujników:\n" + a["facts_verbose"])
        j = evaluate_response(context, a["question"], golden[f"{a['row']}:{a['qi']}"], a["answer"], a["category"])
        fact_numbers = numbers(a["facts_verbose"])
        stray = sorted(n for n in numbers(a["answer"]) if n not in fact_numbers and n not in {"1.0", "2.0"})
        rec = {k: a[k] for k in ("row", "qi", "variant", "category", "question", "answer", "eval_tokens", "done_reason")}
        rec.update({k: j[k] for k in ("faithfulness", "answer_relevancy", "conciseness", "polish_quality", "hallucination_flag")})
        rec["stray_numbers"] = stray
        rec["has_numbers"] = bool(numbers(a["answer"]))
        with judged_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary
    rows = [json.loads(l) for l in judged_path.read_text(encoding="utf-8").splitlines()]
    names = list(dict.fromkeys(["old"] + [r["variant"] for r in rows if r["variant"] != "old"]))
    lines = ["| metryka | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]

    def stat(variant, fn):
        vals = [fn(r) for r in rows if r["variant"] == variant]
        return sum(vals) / len(vals) if vals else float("nan")

    for name, fn in [
        ("faithfulness", lambda r: r["faithfulness"]),
        ("answer_relevancy", lambda r: r["answer_relevancy"]),
        ("conciseness", lambda r: r["conciseness"]),
        ("polish_quality", lambda r: r["polish_quality"]),
        ("średnia 4 ocen", lambda r: (r["faithfulness"] + r["answer_relevancy"] + r["conciseness"] + r["polish_quality"]) / 4),
        ("hallucination_flag (udział)", lambda r: float(r["hallucination_flag"])),
        ("odpowiedzi z liczbą spoza danych (udział)", lambda r: float(bool(r["stray_numbers"]))),
        ("śr. długość odpowiedzi [tokeny]", lambda r: r["eval_tokens"] or 0),
    ]:
        lines.append(f"| {name} | " + " | ".join(f"{stat(v, fn):.3f}" for v in names) + " |")
    no_num = lambda v: [not r["has_numbers"] for r in rows if r["variant"] == v and r["category"] in ("persona", "stt_garbage")]
    lines.append("| persona/off-topic bez liczb | " + " | ".join(
        f"{sum(no_num(v))}/{len(no_num(v))}" for v in names) + " |")
    per_cat = ["", "| kategoria (śr. 4 ocen) | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for cat in dict.fromkeys(r["category"] for r in rows):
        def cs(v):
            xs = [(r["faithfulness"] + r["answer_relevancy"] + r["conciseness"] + r["polish_quality"]) / 4
                  for r in rows if r["variant"] == v and r["category"] == cat]
            return sum(xs) / len(xs) if xs else float("nan")
        per_cat.append(f"| {cat} | " + " | ".join(f"{cs(v):.3f}" for v in names) + " |")
    summary = "\n".join(lines + per_cat)
    (out_dir / "ab_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
