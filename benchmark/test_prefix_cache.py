#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SYSTEM_PROMPT_STATIC = """Jesteś asystentem smart-home. Aktualny stan mieszkania:
- Temperatura salon: 21.4°C
- Wilgotność łazienka: 29.2%
- Okno sypialnia: otwarte
- Bojler: 1289W
Odpowiadaj krótko po polsku."""

SYSTEM_PROMPT_DYNAMIC = """Stan na 2026-06-18 21:15:00 UTC.
Jesteś asystentem smart-home. Aktualny stan mieszkania:
- Temperatura salon: 21.4°C
- Wilgotność łazienka: 29.2%
- Okno sypialnia: otwarte
- Bojler: 1289W
Odpowiadaj krótko po polsku."""

QUESTIONS = [
    "Jaka jest temperatura w salonie?",
    "Jaka jest wilgotność w łazience?",
    "Czy okno w sypialni jest otwarte?",
    "Jaka jest moc bojlera?",
]

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
DEFAULT_OLLAMA_URL = "http://mill-56-rpi.local:11434"
DEFAULT_RPI_SSH = "krisqma@mill-56-rpi"
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


def model_to_safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).replace(".", "_").lower()


def template_for_model(model: str) -> str:
    if model.startswith("bielik"):
        return "bielik"
    if model.startswith("gemma4"):
        return "gemma4"
    return "gemma"


def resolve_local_gguf_path(model: str, variant: str) -> Path | None:
    relative = Path(model) / f"{model}-{variant}.gguf"
    candidates = [
        Path("models") / relative,
        Path("quant") / "models" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


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


def call_ollama(
    *,
    ollama_url: str,
    model_tag: str,
    system_prompt: str,
    question: str,
) -> tuple[str, float | None, float, str | None]:
    options = {
        key: value
        for key, value in FIXED_PARAMS.items()
        if key not in {"stream", "keep_alive"}
    }
    payload = {
        "model": model_tag,
        "prompt": question,
        "system": system_prompt,
        "stream": True,
        "keep_alive": FIXED_PARAMS["keep_alive"],
        "options": options,
    }
    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    started = time.monotonic()
    ttft_ms: float | None = None
    answer_parts: list[str] = []
    final_data: dict[str, Any] | None = None

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{ollama_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    final_data = chunk
                    raw_piece = chunk.get("response", "")
                    piece = "" if raw_piece is None else str(raw_piece)
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.monotonic() - started) * 1000
                        answer_parts.append(piece)
        e2e_ms = (time.monotonic() - started) * 1000
        if final_data and final_data.get("error"):
            return "".join(answer_parts).strip(), ttft_ms, e2e_ms, str(final_data["error"])
        return "".join(answer_parts).strip(), ttft_ms, e2e_ms, None
    except Exception as exc:
        e2e_ms = (time.monotonic() - started) * 1000
        return "".join(answer_parts).strip(), ttft_ms, e2e_ms, str(exc)


def truncate(value: str, width: int = 30) -> str:
    if len(value) <= width:
        return value
    return value[: width - 3].rstrip() + "..."


def seconds(ms: float | None) -> str:
    if ms is None:
        return "null"
    return f"{ms / 1000:.1f}s"


def cache_label(index: int, ttft_ms: float | None, baseline_ms: float | None) -> str:
    if index == 1:
        return "MISS (zimny)"
    if index == 3:
        return "MISS (inny prefix)"
    if ttft_ms is None or baseline_ms is None:
        return "MISS"
    return "HIT" if ttft_ms < baseline_ms * 0.30 else "MISS"


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(f"{'#':<3}{'Prompt':<9}{'Pytanie':<32}{'TTFT':>9}{'E2E':>10}    Cache")
    for row in rows:
        print(
            f"{row['index']:<3}"
            f"{row['prompt_type']:<9}"
            f"{truncate(row['question']):<32}"
            f"{seconds(row['ttft_ms']):>9}"
            f"{seconds(row['e2e_ms']):>10}    "
            f"{row['cache']}"
        )


def print_conclusions(rows: list[dict[str, Any]]) -> None:
    ttft_1 = rows[0]["ttft_ms"]
    ttft_2 = rows[1]["ttft_ms"]
    ttft_3 = rows[2]["ttft_ms"]

    print()
    print("=== WNIOSKI ===")
    if ttft_1 is not None and ttft_2 is not None:
        reduction = (1.0 - ttft_2 / ttft_1) * 100 if ttft_1 > 0 else 0.0
        print(
            "Cache HIT redukuje TTFT: "
            f"{seconds(ttft_1)} → {seconds(ttft_2)}  ({reduction:+.1f}%)"
        )
    else:
        print("Cache HIT redukuje TTFT: brak danych TTFT dla requestu 1 lub 2")

    if ttft_2 is not None and ttft_3 is not None:
        print(f"Zmiana prefiksu niszczy cache: {seconds(ttft_2)} → {seconds(ttft_3)}")
    else:
        print("Zmiana prefiksu niszczy cache: brak danych TTFT dla requestu 2 lub 3")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test Ollama prefix/KV cache via TTFT")
    parser.add_argument("--model", default="gemma3-4b")
    parser.add_argument("--variant", default="Q4_K_M")
    args = parser.parse_args()

    ollama_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    rpi_ssh = os.environ.get("QUANT_RPI_SSH", DEFAULT_RPI_SSH)
    remote_models_dir = os.environ.get("QUANT_REMOTE_MODELS_DIR", DEFAULT_REMOTE_MODELS_DIR)
    remote_ollama_host = os.environ.get("QUANT_REMOTE_OLLAMA_HOST", DEFAULT_REMOTE_OLLAMA_HOST)
    model_tag = model_to_safe(f"quant-{args.model}-{args.variant}")
    local_gguf_path = resolve_local_gguf_path(args.model, args.variant)
    rpi_gguf_path = remote_gguf_path(args.model, args.variant, remote_models_dir)
    template_key = template_for_model(args.model)

    print("=== TEST PREFIX CACHING ===")
    print(f"Model: {args.model} / {args.variant}")
    print(f"Ollama: {ollama_url}")
    print(f"RPI SSH: {rpi_ssh}")
    print(f"Remote Ollama Host: {remote_ollama_host}")
    print(f"Tag: {model_tag}")
    if local_gguf_path:
        print(f"Local GGUF: {local_gguf_path}")
    print(f"Remote GGUF: {rpi_gguf_path}")

    print("Checking Ollama model on RPi...", flush=True)
    created = ensure_ollama_model(
        model_tag=model_tag,
        remote_path=rpi_gguf_path,
        template_key=template_key,
        rpi_ssh=rpi_ssh,
        remote_ollama_host=remote_ollama_host,
    )
    print("Ollama model: created" if created else "Ollama model: already exists")

    sequence = [
        ("STATIC", SYSTEM_PROMPT_STATIC, QUESTIONS[0]),
        ("STATIC", SYSTEM_PROMPT_STATIC, QUESTIONS[1]),
        ("DYNAMIC", SYSTEM_PROMPT_DYNAMIC, QUESTIONS[2]),
        ("STATIC", SYSTEM_PROMPT_STATIC, QUESTIONS[3]),
    ]

    rows: list[dict[str, Any]] = []
    baseline_ms: float | None = None
    for index, (prompt_type, system_prompt, question) in enumerate(sequence, start=1):
        answer, ttft_ms, e2e_ms, error = call_ollama(
            ollama_url=ollama_url,
            model_tag=model_tag,
            system_prompt=system_prompt,
            question=question,
        )
        if index == 1:
            baseline_ms = ttft_ms
        label = cache_label(index, ttft_ms, baseline_ms)
        rows.append(
            {
                "index": index,
                "prompt_type": prompt_type,
                "question": question,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "cache": label,
                "answer": answer,
                "error": error,
            }
        )
        if error:
            print(f"WARN request {index}: {error}", file=sys.stderr)

    print_table(rows)
    print_conclusions(rows)


if __name__ == "__main__":
    main()
