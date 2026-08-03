#!/usr/bin/env python3
"""Wznawialny dogrywacz latencji kwantyzacji: najpierw WARM, potem COLD → n=100.

Dokłada powtórki tych samych 20 pytań (repeat=0..4) do istniejących zbiorów n≈20.
Bezpieczne przy Ctrl+C: każdy udany rekord jest append+fsync do stałego JSONL.
Nieudane pomiary (null TTFT / pusta odpowiedź / błąd) nie są zapisywane —
ten sam case jest ponawiany do skutku; completed liczy tylko valid rekordy.

Przykłady:
  python run_quant_latency_extend.py --status
  python run_quant_latency_extend.py --continue --target-n 100
  python run_quant_latency_extend.py --continue --phase warm --target-n 100
  caffeinate -i python run_quant_latency_extend.py --continue --target-n 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

import test_cold_cache_generation as cold
import test_warm_cache_generation as warm

SCRIPT_ROOT = Path(__file__).resolve().parent

MODELS = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
VARIANTS = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]

WARM_SEED = SCRIPT_ROOT / "analysis" / "quant-warm-cache" / "warm_cache_generation_full_20260618_20260621.jsonl"
COLD_SEED = SCRIPT_ROOT / "results" / "cold_cache_generation_cold_full_20260729.jsonl"
WARM_OUT = SCRIPT_ROOT / "results" / "quant_latency_warm_n100.jsonl"
COLD_OUT = SCRIPT_ROOT / "results" / "quant_latency_cold_n100.jsonl"
DEFAULT_CASES = SCRIPT_ROOT / "warm_cache_cases.json"

Key = tuple[str, str, str, int]  # model, variant, case_id, repeat


def parse_csv(raw: str | None, allowed: list[str], label: str) -> list[str]:
    if raw is None:
        return list(allowed)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def record_key(row: dict[str, Any]) -> Key | None:
    model = row.get("model")
    variant = row.get("quant_variant")
    case_id = row.get("case_id")
    repeat = row.get("repeat", 0)
    if not isinstance(model, str) or not isinstance(variant, str) or not isinstance(case_id, str):
        return None
    try:
        repeat_i = int(repeat)
    except (TypeError, ValueError):
        repeat_i = 0
    return model, variant, case_id, repeat_i


def is_valid_latency_row(row: dict[str, Any]) -> bool:
    """Rekord nadaje się do n=100: ma TTFT, generation i niepustą odpowiedź."""
    if row.get("ttft_ms") is None or row.get("generation_ms") is None:
        return False
    try:
        return int(row.get("answer_chars") or 0) > 0
    except (TypeError, ValueError):
        return False


def completed_keys(rows: list[dict[str, Any]]) -> set[Key]:
    keys: set[Key] = set()
    for row in rows:
        if not is_valid_latency_row(row):
            continue
        key = record_key(row)
        if key is not None:
            keys.add(key)
    return keys


def measurement_ok(answer: str, ttft_ms: float | None, error: str | None) -> tuple[bool, str]:
    """Czy pomiar jest kompletny; przy failu zwraca powód do RETRY."""
    if error:
        return False, error
    if ttft_ms is None:
        return False, "empty_stream_no_ttft"
    if not answer:
        return False, "empty_answer"
    return True, ""


def call_ollama_until_valid(
    *,
    prefix: str,
    repeat: int,
    q_idx: int,
    ollama_urls: list[str],
    model_tag: str,
    api: str,
    system_prompt: str,
    question: str,
    request_timeout: int,
    keep_alive: str,
) -> tuple[str, float, float]:
    """Woła Ollamę aż dostanie TTFT + niepustą odpowiedź (bez błędu). Nie zapisuje faili."""
    attempt = 0
    while True:
        attempt += 1
        answer, ttft_ms, e2e_ms, error = cold.call_ollama(
            ollama_urls=ollama_urls,
            model_tag=model_tag,
            api=api,
            system_prompt=system_prompt,
            question=question,
            request_timeout=request_timeout,
            keep_alive=keep_alive,
        )
        ok, reason = measurement_ok(answer, ttft_ms, error)
        if ok:
            assert ttft_ms is not None
            return answer, ttft_ms, e2e_ms
        print(
            f"{prefix} RETRY r{repeat} q{q_idx} attempt={attempt} "
            f"reason={reason} e2e={cold.fmt_s(e2e_ms)}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(2.0)


def counts_per_cell(keys: set[Key], models: list[str], variants: list[str]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {(m, v): 0 for m in models for v in variants}
    for model, variant, _case, _rep in keys:
        cell = (model, variant)
        if cell in counts:
            counts[cell] += 1
    return counts


def seed_output_if_needed(
    *,
    output_path: Path,
    seed_path: Path,
    cache_mode: str,
) -> int:
    """Jeśli output pusty — skopiuj seed jako repeat=0. Zwraca liczbę dopisanych rekordów."""
    existing = read_jsonl(output_path)
    if existing:
        return 0
    if not seed_path.exists():
        print(f"WARN: brak seed {seed_path} — start od zera", file=sys.stderr, flush=True)
        return 0

    seeded = 0
    for row in read_jsonl(seed_path):
        if not isinstance(row.get("model"), str) or not isinstance(row.get("quant_variant"), str):
            continue
        if not isinstance(row.get("case_id"), str):
            continue
        out = dict(row)
        out["repeat"] = int(out.get("repeat", 0) or 0)
        out["cache_mode"] = cache_mode
        out["source"] = "seed"
        append_jsonl(output_path, out)
        seeded += 1
    print(f"SEED {cache_mode}: skopiowano {seeded} rekordów z {seed_path.name} → {output_path.name}", flush=True)
    return seeded


def iter_missing_jobs(
    *,
    model: str,
    variant: str,
    questions: list[dict[str, str]],
    completed: set[Key],
    target_n: int,
) -> Iterator[tuple[int, int, dict[str, str]]]:
    """Yield (repeat, question_idx, case) aż cell osiągnie target_n."""
    done = sum(1 for m, v, _c, _r in completed if m == model and v == variant)
    need = target_n - done
    if need <= 0:
        return
    n_q = len(questions)
    max_repeats = max(1, math.ceil(target_n / n_q))
    emitted = 0
    for repeat in range(max_repeats):
        for idx, case in enumerate(questions, start=1):
            key = (model, variant, case["case_id"], repeat)
            if key in completed:
                continue
            yield repeat, idx, case
            emitted += 1
            if emitted >= need:
                return


def print_status(
    *,
    phase: str,
    output_path: Path,
    models: list[str],
    variants: list[str],
    target_n: int,
) -> int:
    rows = read_jsonl(output_path)
    keys = completed_keys(rows)
    counts = counts_per_cell(keys, models, variants)
    remaining = 0
    print(f"\n=== STATUS {phase.upper()} → {output_path.name} (target n={target_n}) ===")
    print(f"{'model':<14}{'variant':<10}{'done':>6}{'need':>6}")
    for model in models:
        for variant in variants:
            done = counts[(model, variant)]
            need = max(0, target_n - done)
            remaining += need
            print(f"{model:<14}{variant:<10}{done:>6}{need:>6}")
    print(f"RAZEM brakuje: {remaining} requestów | już w pliku: {len(rows)}")
    return remaining


def run_warm_cell(
    *,
    model: str,
    variant: str,
    jobs: list[tuple[int, int, dict[str, str]]],
    system_prompt: str,
    ollama_urls: list[str],
    rpi_ssh: str,
    remote_models_dir: str,
    remote_ollama_host: str,
    output_path: Path,
    completed: set[Key],
    target_n: int,
    request_timeout: int,
) -> int:
    if not jobs:
        return 0

    api = warm.MODEL_API[model]
    template_key = warm.template_for_model(model)
    remote_path = warm.remote_gguf_path(model, variant, remote_models_dir)
    model_tag = warm.model_to_safe(f"warm-ext-{model}-{variant}")
    prefix = f"[WARM {model}/{variant}]"
    gguf_size_gb = warm.remote_file_size_gb(rpi_ssh, remote_path)

    print(f"{prefix} loading tag={model_tag} jobs={len(jobs)}", flush=True)
    warm.ensure_ollama_model(
        model_tag=model_tag,
        remote_path=remote_path,
        template_key=template_key,
        rpi_ssh=rpi_ssh,
        remote_ollama_host=remote_ollama_host,
    )
    measured = 0
    try:
        print(f"{prefix} warmup...", flush=True)
        _, cold_ttft_ms, cold_e2e_ms, warmup_error = cold.call_ollama(
            ollama_urls=ollama_urls,
            model_tag=model_tag,
            api=api,
            system_prompt=system_prompt,
            question="OK",
            request_timeout=request_timeout,
            keep_alive="30m",
        )
        if warmup_error:
            print(f"{prefix} WARN warmup: {warmup_error}", file=sys.stderr, flush=True)
        print(
            f"{prefix} cold_baseline ttft={warm.fmt_s(cold_ttft_ms)} e2e={warm.fmt_s(cold_e2e_ms)}",
            flush=True,
        )

        for repeat, q_idx, case in jobs:
            answer, ttft_ms, e2e_ms = call_ollama_until_valid(
                prefix=prefix,
                repeat=repeat,
                q_idx=q_idx,
                ollama_urls=ollama_urls,
                model_tag=model_tag,
                api=api,
                system_prompt=system_prompt,
                question=case["question"],
                request_timeout=request_timeout,
                keep_alive="30m",
            )
            generation_ms = max(e2e_ms - ttft_ms, 0.0)
            cache_hit = cold_ttft_ms is not None and ttft_ms < 0.30 * cold_ttft_ms
            record = {
                "cache_mode": "warm",
                "source": "extend",
                "model": model,
                "quant_variant": variant,
                "quant_method": warm.QUANT_METHOD[variant],
                "gguf_size_gb": gguf_size_gb,
                "repeat": repeat,
                "question_idx": q_idx,
                "case_id": case["case_id"],
                "ttft_ms": round(ttft_ms, 1),
                "e2e_ms": round(e2e_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "answer_chars": len(answer),
                "cache_hit": cache_hit,
            }
            append_jsonl(output_path, record)
            key = (model, variant, case["case_id"], repeat)
            completed.add(key)
            measured += 1
            done_cell = sum(1 for m, v, _c, _r in completed if m == model and v == variant)
            print(
                f"{prefix} repeat={repeat} q={q_idx} "
                f"ttft={warm.fmt_s(ttft_ms)} e2e={warm.fmt_s(e2e_ms)} "
                f"gen={warm.fmt_s(generation_ms)} cache={'HIT' if cache_hit else 'MISS'} "
                f"done={done_cell}/{target_n}",
                flush=True,
            )
    finally:
        print(f"{prefix} removing model...", flush=True)
        warm.remove_ollama_model(rpi_ssh, remote_ollama_host, model_tag)

    return measured


def run_cold_cell(
    *,
    model: str,
    variant: str,
    jobs: list[tuple[int, int, dict[str, str]]],
    system_prompt: str,
    ollama_urls: list[str],
    rpi_ssh: str,
    remote_models_dir: str,
    remote_ollama_host: str,
    output_path: Path,
    completed: set[Key],
    target_n: int,
    request_timeout: int,
) -> int:
    if not jobs:
        return 0

    api = cold.MODEL_API[model]
    template_key = cold.template_for_model(model)
    remote_path = cold.remote_gguf_path(model, variant, remote_models_dir)
    model_tag = cold.model_to_safe(f"cold-ext-{model}-{variant}")
    prefix = f"[COLD {model}/{variant}]"
    gguf_size_gb = cold.remote_file_size_gb(rpi_ssh, remote_path)

    print(f"{prefix} create tag={model_tag} jobs={len(jobs)}", flush=True)
    cold.remove_ollama_model(rpi_ssh, remote_ollama_host, model_tag)
    measured = 0
    try:
        cold.create_ollama_model(
            model_tag=model_tag,
            remote_path=remote_path,
            template_key=template_key,
            rpi_ssh=rpi_ssh,
            remote_ollama_host=remote_ollama_host,
        )
        for repeat, q_idx, case in jobs:
            answer, ttft_ms, e2e_ms = call_ollama_until_valid(
                prefix=prefix,
                repeat=repeat,
                q_idx=q_idx,
                ollama_urls=ollama_urls,
                model_tag=model_tag,
                api=api,
                system_prompt=system_prompt,
                question=case["question"],
                request_timeout=request_timeout,
                keep_alive="0s",
            )
            generation_ms = max(e2e_ms - ttft_ms, 0.0)
            record = {
                "cache_mode": "cold",
                "source": "extend",
                "model": model,
                "quant_variant": variant,
                "quant_method": cold.QUANT_METHOD[variant],
                "gguf_size_gb": gguf_size_gb,
                "repeat": repeat,
                "question_idx": q_idx,
                "case_id": case["case_id"],
                "ttft_ms": round(ttft_ms, 1),
                "e2e_ms": round(e2e_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "answer_chars": len(answer),
            }
            append_jsonl(output_path, record)
            completed.add((model, variant, case["case_id"], repeat))
            measured += 1
            done_cell = sum(1 for m, v, _c, _r in completed if m == model and v == variant)
            print(
                f"{prefix} repeat={repeat} q={q_idx} "
                f"ttft={cold.fmt_s(ttft_ms)} e2e={cold.fmt_s(e2e_ms)} "
                f"gen={cold.fmt_s(generation_ms)} "
                f"done={done_cell}/{target_n}",
                flush=True,
            )
    finally:
        print(f"{prefix} removing model...", flush=True)
        cold.remove_ollama_model(rpi_ssh, remote_ollama_host, model_tag)

    return measured


def run_phase(
    *,
    phase: str,
    output_path: Path,
    seed_path: Path,
    models: list[str],
    variants: list[str],
    questions: list[dict[str, str]],
    system_prompt: str,
    target_n: int,
    ollama_urls: list[str],
    rpi_ssh: str,
    remote_models_dir: str,
    remote_ollama_host: str,
    request_timeout: int,
) -> None:
    seed_output_if_needed(output_path=output_path, seed_path=seed_path, cache_mode=phase)
    rows = read_jsonl(output_path)
    completed = completed_keys(rows)
    remaining_before = print_status(
        phase=phase,
        output_path=output_path,
        models=models,
        variants=variants,
        target_n=target_n,
    )
    if remaining_before == 0:
        print(f"{phase.upper()}: komplet — nic do roboty.", flush=True)
        return

    total_measured = 0
    for model in models:
        for variant in variants:
            jobs = list(
                iter_missing_jobs(
                    model=model,
                    variant=variant,
                    questions=questions,
                    completed=completed,
                    target_n=target_n,
                )
            )
            if not jobs:
                continue
            if phase == "warm":
                total_measured += run_warm_cell(
                    model=model,
                    variant=variant,
                    jobs=jobs,
                    system_prompt=system_prompt,
                    ollama_urls=ollama_urls,
                    rpi_ssh=rpi_ssh,
                    remote_models_dir=remote_models_dir,
                    remote_ollama_host=remote_ollama_host,
                    output_path=output_path,
                    completed=completed,
                    target_n=target_n,
                    request_timeout=request_timeout,
                )
            else:
                total_measured += run_cold_cell(
                    model=model,
                    variant=variant,
                    jobs=jobs,
                    system_prompt=system_prompt,
                    ollama_urls=ollama_urls,
                    rpi_ssh=rpi_ssh,
                    remote_models_dir=remote_models_dir,
                    remote_ollama_host=remote_ollama_host,
                    output_path=output_path,
                    completed=completed,
                    target_n=target_n,
                    request_timeout=request_timeout,
                )

    print_status(
        phase=phase,
        output_path=output_path,
        models=models,
        variants=variants,
        target_n=target_n,
    )
    print(f"{phase.upper()}: w tej sesji zmierzono {total_measured} requestów.", flush=True)


def main() -> None:
    load_dotenv(SCRIPT_ROOT / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Wznawialny dogrywacz warm→cold latencji kwantyzacji do target-n"
    )
    parser.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        default=True,
        help="Wznawia i dokłada brakujące (domyślnie włączone)",
    )
    parser.add_argument(
        "--no-continue",
        dest="do_continue",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--status", action="store_true", help="Tylko pokaż braki, nic nie mierz")
    parser.add_argument("--target-n", type=int, default=100, help="Docelowa liczba rekordów na model×wariant")
    parser.add_argument(
        "--phase",
        choices=("all", "warm", "cold"),
        default="all",
        help="warm → cold (all), albo jedna faza",
    )
    parser.add_argument("--models", default=None, help="Comma-separated model ids")
    parser.add_argument("--variants", default=None, help="Comma-separated quant variants")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--warm-out", type=Path, default=WARM_OUT)
    parser.add_argument("--cold-out", type=Path, default=COLD_OUT)
    parser.add_argument("--warm-seed", type=Path, default=WARM_SEED)
    parser.add_argument("--cold-seed", type=Path, default=COLD_SEED)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "180")),
    )
    parser.add_argument("--rpi-ssh", default=os.environ.get("QUANT_RPI_SSH", cold.DEFAULT_RPI_SSH))
    parser.add_argument(
        "--remote-models-dir",
        default=os.environ.get("QUANT_REMOTE_MODELS_DIR", cold.DEFAULT_REMOTE_MODELS_DIR),
    )
    parser.add_argument(
        "--remote-ollama-host",
        default=os.environ.get("QUANT_REMOTE_OLLAMA_HOST", cold.DEFAULT_REMOTE_OLLAMA_HOST),
    )
    args = parser.parse_args()

    if args.target_n < 1:
        raise SystemExit("--target-n must be >= 1")

    models = parse_csv(args.models, MODELS, "model(s)")
    variants = parse_csv(args.variants, VARIANTS, "variant(s)")
    system_prompt, questions = cold.load_cases(args.cases, None)
    if not questions:
        raise SystemExit("Brak pytań w cases")

    # Status / seed preview — seedujemy przy statusie też, żeby liczby miały sens.
    if args.phase in ("all", "warm"):
        seed_output_if_needed(output_path=args.warm_out, seed_path=args.warm_seed, cache_mode="warm")
    if args.phase in ("all", "cold"):
        seed_output_if_needed(output_path=args.cold_out, seed_path=args.cold_seed, cache_mode="cold")

    if args.status or not args.do_continue:
        remaining = 0
        if args.phase in ("all", "warm"):
            remaining += print_status(
                phase="warm",
                output_path=args.warm_out,
                models=models,
                variants=variants,
                target_n=args.target_n,
            )
        if args.phase in ("all", "cold"):
            remaining += print_status(
                phase="cold",
                output_path=args.cold_out,
                models=models,
                variants=variants,
                target_n=args.target_n,
            )
        print(f"\nSUMA braków (wybrane fazy): {remaining}")
        print("Uruchom z --continue żeby mierzyć, np.:")
        print("  caffeinate -i python run_quant_latency_extend.py --continue --target-n 100")
        return

    ollama_urls = cold.configured_urls()
    print("=== QUANT LATENCY EXTEND (resumable) ===", flush=True)
    print(f"target_n={args.target_n}", flush=True)
    print(f"phase={args.phase}", flush=True)
    print(f"models={','.join(models)}", flush=True)
    print(f"variants={','.join(variants)}", flush=True)
    print(f"questions={len(questions)} (× repeats do target_n)", flush=True)
    print(f"warm_out={args.warm_out}", flush=True)
    print(f"cold_out={args.cold_out}", flush=True)
    print(f"ollama_urls={', '.join(ollama_urls)}", flush=True)
    print(f"rpi_ssh={args.rpi_ssh}", flush=True)

    try:
        if args.phase in ("all", "warm"):
            run_phase(
                phase="warm",
                output_path=args.warm_out,
                seed_path=args.warm_seed,
                models=models,
                variants=variants,
                questions=questions,
                system_prompt=system_prompt,
                target_n=args.target_n,
                ollama_urls=ollama_urls,
                rpi_ssh=args.rpi_ssh,
                remote_models_dir=args.remote_models_dir,
                remote_ollama_host=args.remote_ollama_host,
                request_timeout=args.request_timeout,
            )
        if args.phase in ("all", "cold"):
            run_phase(
                phase="cold",
                output_path=args.cold_out,
                seed_path=args.cold_seed,
                models=models,
                variants=variants,
                questions=questions,
                system_prompt=system_prompt,
                target_n=args.target_n,
                ollama_urls=ollama_urls,
                rpi_ssh=args.rpi_ssh,
                remote_models_dir=args.remote_models_dir,
                remote_ollama_host=args.remote_ollama_host,
                request_timeout=args.request_timeout,
            )
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl+C). Stan zapisany — odpal ponownie z --continue.", flush=True)
        raise SystemExit(130) from None

    print("\nGotowe (lub w pełni domknięte dla wybranych faz).", flush=True)


if __name__ == "__main__":
    main()
