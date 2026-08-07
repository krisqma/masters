#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


MODELS = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
VARIANTS = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]
MODEL_API = {
    "gemma3-4b": "generate",
    "gemma4-e2b": "chat",
    "bielik-4.5b": "generate",
}
QUANT_METHOD = {
    "Q2_K": "K-Quant",
    "Q4_0": "RTN",
    "Q4_K_M": "K-Quant",
    "Q8_0": "RTN",
}

FIXED_PARAMS = {
    "temperature": 0.2,
    "num_ctx": 2048,
    "num_predict": 128,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "keep_alive": "30m",
    "stream": True,
}

REQUEST_TIMEOUT = 120
MODEL_LOAD_TIMEOUT = 900
# Hotspot: 172.20.10.2 / mill-56-rpi.local
DEFAULT_RPI_OLLAMA_URLS = (
    "http://192.168.1.173:11434",
)
DEFAULT_RPI_SSH = "krisqma@192.168.1.173"
DEFAULT_REMOTE_MODELS_DIR = "/home/krisqma/quant/models"
DEFAULT_REMOTE_OLLAMA_HOST = "127.0.0.1:11434"

TEMPLATES = {
    "gemma": """<bos><start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
""",
    "bielik": """<s>{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
}

GEMMA4_MODELFILE_SUFFIX = """TEMPLATE {{ .Prompt }}
SYSTEM You are a helpful AI assistant.
RENDERER gemma4
PARSER gemma4
PARAMETER stop <turn|>
"""


def configured_urls() -> list[str]:
    urls: list[str] = []
    for env_name in ("OLLAMA_URL", "RPI_OLLAMA_URL"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            urls.extend(part.strip().rstrip("/") for part in raw.split(",") if part.strip())
    urls.extend(DEFAULT_RPI_OLLAMA_URLS)

    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def parse_csv(raw: str | None, allowed: list[str], label: str) -> list[str]:
    if raw is None:
        return list(allowed)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")
    return values


def model_to_safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).replace(".", "_").lower()


def template_for_model(model: str) -> str:
    if model.startswith("bielik"):
        return "bielik"
    if model.startswith("gemma4"):
        return "gemma4"
    return "gemma"


def remote_gguf_path(model: str, variant: str, remote_models_dir: str) -> str:
    return f"{remote_models_dir.rstrip('/')}/{model}/{model}-{variant}.gguf"


def ssh(
    rpi_ssh: str,
    command: str,
    *,
    input_text: str | None = None,
    timeout: int = 30,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", rpi_ssh, command],
        input=input_text,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise SystemExit(f"{action} failed: {detail}")


def remote_file_exists(rpi_ssh: str, path: str) -> bool:
    result = ssh(rpi_ssh, f"test -s {shlex.quote(path)}", timeout=30)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    require_success(result, f"remote file check for {path}")
    return False


def remote_file_size_gb(rpi_ssh: str, path: str) -> float:
    result = ssh(rpi_ssh, f"stat -c %s {shlex.quote(path)}", timeout=30)
    require_success(result, f"remote file size for {path}")
    size_bytes = int(result.stdout.strip())
    return round(size_bytes / (1024**3), 3)


def existing_ollama_models(rpi_ssh: str, remote_ollama_host: str) -> set[str]:
    command = f"OLLAMA_HOST={shlex.quote(remote_ollama_host)} ollama list"
    result = ssh(rpi_ssh, command, timeout=30)
    require_success(result, "remote ollama list")
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0].split(":", 1)[0])
            names.add(parts[0])
    return names


def modelfile_for(gguf_path: str, template_key: str) -> str:
    if template_key == "gemma4":
        return f"FROM {gguf_path}\n{GEMMA4_MODELFILE_SUFFIX}"
    return f'FROM {gguf_path}\nTEMPLATE """{TEMPLATES[template_key]}"""\n'


def ensure_ollama_model(
    *,
    model_tag: str,
    remote_path: str,
    template_key: str,
    rpi_ssh: str,
    remote_ollama_host: str,
) -> bool:
    if not remote_file_exists(rpi_ssh, remote_path):
        raise SystemExit(f"Remote GGUF not found on RPi: {remote_path}")

    if model_tag in existing_ollama_models(rpi_ssh, remote_ollama_host):
        return False

    remote_modelfile = f"/tmp/{model_tag}.Modelfile"
    write_result = ssh(
        rpi_ssh,
        f"cat > {shlex.quote(remote_modelfile)}",
        input_text=modelfile_for(remote_path, template_key),
        timeout=30,
    )
    require_success(write_result, "remote Modelfile write")

    try:
        create_cmd = (
            f"OLLAMA_HOST={shlex.quote(remote_ollama_host)} "
            f"ollama create {shlex.quote(model_tag)} -f {shlex.quote(remote_modelfile)}"
        )
        result = ssh(
            rpi_ssh,
            create_cmd,
            timeout=MODEL_LOAD_TIMEOUT,
            capture_output=False,
        )
        require_success(result, "remote ollama create")
    finally:
        ssh(rpi_ssh, f"rm -f {shlex.quote(remote_modelfile)}", timeout=30)

    return True


def remove_ollama_model(rpi_ssh: str, remote_ollama_host: str, model_tag: str) -> None:
    rm_cmd = f"OLLAMA_HOST={shlex.quote(remote_ollama_host)} ollama rm {shlex.quote(model_tag)}"
    ssh(rpi_ssh, rm_cmd, timeout=60)


def load_cases(path: Path, limit: int | None) -> tuple[str, list[dict[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    system_prompt = data.get("system_prompt")
    questions = data.get("questions")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise SystemExit(f"{path} must contain a non-empty system_prompt")
    if not isinstance(questions, list) or len(questions) < 1:
        raise SystemExit(f"{path} must contain a non-empty questions array")

    parsed: list[dict[str, str]] = []
    for idx, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"{path}: question {idx} is not an object")
        case_id = item.get("case_id")
        question = item.get("question")
        if not isinstance(case_id, str) or not isinstance(question, str):
            raise SystemExit(f"{path}: question {idx} must contain case_id and question strings")
        parsed.append({"case_id": case_id, "question": question})

    return system_prompt, parsed[:limit] if limit is not None else parsed


def build_payload(model_tag: str, api: str, system_prompt: str, question: str) -> tuple[str, dict[str, Any]]:
    options = {
        key: value
        for key, value in FIXED_PARAMS.items()
        if key not in {"stream", "keep_alive"}
    }
    if api == "chat":
        return "/api/chat", {
            "model": model_tag,
            "messages": [
                {"role": "system", "content": f"/no_think\n{system_prompt}"},
                {"role": "user", "content": question},
            ],
            "think": False,
            "stream": True,
            "keep_alive": FIXED_PARAMS["keep_alive"],
            "options": options,
        }
    return "/api/generate", {
        "model": model_tag,
        "prompt": question,
        "system": system_prompt,
        "stream": True,
        "keep_alive": FIXED_PARAMS["keep_alive"],
        "options": options,
    }


def call_ollama(
    *,
    ollama_urls: list[str],
    model_tag: str,
    api: str,
    system_prompt: str,
    question: str,
) -> tuple[str, float | None, float, str | None]:
    endpoint, payload = build_payload(model_tag, api, system_prompt, question)
    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    started = time.monotonic()
    last_error: str | None = None

    for ollama_url in ollama_urls:
        answer_parts: list[str] = []
        ttft_ms: float | None = None
        final_data: dict[str, Any] | None = None
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", f"{ollama_url}{endpoint}", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        final_data = chunk
                        if api == "chat":
                            raw_piece = chunk.get("message", {}).get("content", "")
                        else:
                            raw_piece = chunk.get("response", "")
                        piece = "" if raw_piece is None else str(raw_piece)
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = (time.monotonic() - started) * 1000
                            answer_parts.append(piece)
            e2e_ms = (time.monotonic() - started) * 1000
            answer = "".join(answer_parts).strip()
            if final_data and final_data.get("error"):
                return answer, ttft_ms, e2e_ms, f"ollama_error: {final_data['error']}"
            return answer, ttft_ms, e2e_ms, None
        except httpx.TransportError as exc:
            last_error = f"{ollama_url}: {exc}"
            continue
        except Exception as exc:
            e2e_ms = (time.monotonic() - started) * 1000
            answer = "".join(answer_parts).strip()
            return answer, ttft_ms, e2e_ms, f"ollama_error: {exc}"

    e2e_ms = (time.monotonic() - started) * 1000
    return "", None, e2e_ms, f"ollama_transport_error: {last_error}"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fmt_s(ms: float | None) -> str:
    if ms is None:
        return "null"
    return f"{ms / 1000:.1f}s"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def run_combination(
    *,
    model: str,
    variant: str,
    system_prompt: str,
    questions: list[dict[str, str]],
    ollama_urls: list[str],
    rpi_ssh: str,
    remote_models_dir: str,
    remote_ollama_host: str,
    output_path: Path,
) -> dict[str, Any]:
    api = MODEL_API[model]
    template_key = template_for_model(model)
    remote_path = remote_gguf_path(model, variant, remote_models_dir)
    model_tag = model_to_safe(f"warm-cache-{model}-{variant}")
    prefix = f"[{model} / {variant}]"

    print(f"{prefix} loading model tag={model_tag}", flush=True)
    gguf_size_gb = remote_file_size_gb(rpi_ssh, remote_path)

    ttft_values: list[float] = []
    e2e_values: list[float] = []
    gen_values: list[float] = []
    cache_hits = 0
    completed = 0

    try:
        created = ensure_ollama_model(
            model_tag=model_tag,
            remote_path=remote_path,
            template_key=template_key,
            rpi_ssh=rpi_ssh,
            remote_ollama_host=remote_ollama_host,
        )
        print(f"{prefix} model {'created' if created else 'already exists'}", flush=True)
        print(f"{prefix} warmup...", flush=True)
        _, cold_ttft_ms, cold_e2e_ms, warmup_error = call_ollama(
            ollama_urls=ollama_urls,
            model_tag=model_tag,
            api=api,
            system_prompt=system_prompt,
            question="OK",
        )
        if warmup_error:
            print(f"{prefix} WARN warmup: {warmup_error}", file=sys.stderr, flush=True)
        print(
            f"{prefix} cold_baseline ttft={fmt_s(cold_ttft_ms)} e2e={fmt_s(cold_e2e_ms)}",
            flush=True,
        )

        for idx, case in enumerate(questions, start=1):
            answer, ttft_ms, e2e_ms, error = call_ollama(
                ollama_urls=ollama_urls,
                model_tag=model_tag,
                api=api,
                system_prompt=system_prompt,
                question=case["question"],
            )
            generation_ms = None if ttft_ms is None else max(e2e_ms - ttft_ms, 0.0)
            cache_hit = (
                ttft_ms is not None
                and cold_ttft_ms is not None
                and ttft_ms < 0.30 * cold_ttft_ms
            )

            record = {
                "model": model,
                "quant_variant": variant,
                "quant_method": QUANT_METHOD[variant],
                "gguf_size_gb": gguf_size_gb,
                "question_idx": idx,
                "case_id": case["case_id"],
                "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
                "e2e_ms": round(e2e_ms, 1),
                "generation_ms": None if generation_ms is None else round(generation_ms, 1),
                "answer_chars": len(answer),
                "cache_hit": cache_hit,
            }
            if error:
                record["error"] = error
                print(f"{prefix} WARN q{idx}: {error}", file=sys.stderr, flush=True)
            append_jsonl(output_path, record)

            completed += 1
            if ttft_ms is not None:
                ttft_values.append(ttft_ms)
            e2e_values.append(e2e_ms)
            if generation_ms is not None:
                gen_values.append(generation_ms)
            if cache_hit:
                cache_hits += 1

            print(
                f"{prefix} q{idx}/{len(questions)}  "
                f"ttft={fmt_s(ttft_ms)}  e2e={fmt_s(e2e_ms)}  "
                f"gen={fmt_s(generation_ms)}  cache={'HIT' if cache_hit else 'MISS'}",
                flush=True,
            )

        med_ttft = median(ttft_values)
        med_gen = median(gen_values)
        med_e2e = median(e2e_values)
        cache_hit_pct = (cache_hits / completed * 100.0) if completed else 0.0
        print(
            f"{prefix} DONE — median_gen={fmt_s(med_gen)}  median_ttft={fmt_s(med_ttft)}",
            flush=True,
        )
        return {
            "model": model,
            "variant": variant,
            "med_ttft": med_ttft,
            "med_gen": med_gen,
            "med_e2e": med_e2e,
            "cache_hit_pct": cache_hit_pct,
            "completed": completed,
        }
    finally:
        print(f"{prefix} removing model...", flush=True)
        remove_ollama_model(rpi_ssh, remote_ollama_host, model_tag)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print("Model          Variant    med_ttft   med_gen    med_e2e    cache_hit%")
    for row in rows:
        print(
            f"{row['model']:<14}"
            f"{row['variant']:<11}"
            f"{fmt_s(row['med_ttft']):>8}   "
            f"{fmt_s(row['med_gen']):>7}   "
            f"{fmt_s(row['med_e2e']):>7}   "
            f"{row['cache_hit_pct']:>8.0f}%"
        )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Warm-cache generation benchmark for quantized models")
    parser.add_argument("--models", default=None, help="Comma-separated model ids")
    parser.add_argument("--variants", default=None, help="Comma-separated quant variants")
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--cases", type=Path, default=Path("warm_cache_cases.json"))
    args = parser.parse_args()

    selected_models = parse_csv(args.models, MODELS, "model(s)")
    selected_variants = parse_csv(args.variants, VARIANTS, "variant(s)")
    if args.limit_questions is not None and args.limit_questions < 1:
        raise SystemExit("--limit-questions must be >= 1")

    system_prompt, questions = load_cases(args.cases, args.limit_questions)
    ollama_urls = configured_urls()
    rpi_ssh = os.environ.get("QUANT_RPI_SSH", DEFAULT_RPI_SSH)
    remote_models_dir = os.environ.get("QUANT_REMOTE_MODELS_DIR", DEFAULT_REMOTE_MODELS_DIR)
    remote_ollama_host = os.environ.get("QUANT_REMOTE_OLLAMA_HOST", DEFAULT_REMOTE_OLLAMA_HOST)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path("results") / f"warm_cache_generation_{timestamp}.jsonl"

    print("=== WARM CACHE GENERATION BENCHMARK ===", flush=True)
    print(f"models={','.join(selected_models)}", flush=True)
    print(f"variants={','.join(selected_variants)}", flush=True)
    print(f"questions={len(questions)}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"ollama_urls={', '.join(ollama_urls)}", flush=True)
    print(f"rpi_ssh={rpi_ssh}", flush=True)
    print(f"remote_models_dir={remote_models_dir}", flush=True)
    print(f"remote_ollama_host={remote_ollama_host}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for model in selected_models:
        for variant in selected_variants:
            summary_rows.append(
                run_combination(
                    model=model,
                    variant=variant,
                    system_prompt=system_prompt,
                    questions=questions,
                    ollama_urls=ollama_urls,
                    rpi_ssh=rpi_ssh,
                    remote_models_dir=remote_models_dir,
                    remote_ollama_host=remote_ollama_host,
                    output_path=output_path,
                )
            )

    print_summary(summary_rows)
    print(f"\nJSONL: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
