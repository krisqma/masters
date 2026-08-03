#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

import evaluate

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    {"id": "gemma3-4b", "template": "gemma", "api": "generate"},
    {"id": "gemma4-e2b", "template": "gemma4", "api": "chat"},
    {"id": "bielik-4.5b", "template": "bielik", "api": "generate"},
]

MODEL_GROUPS = {
    "gemma3": ["gemma3-4b"],
    "gemma4": ["gemma4-e2b"],
    "bielik": ["bielik-4.5b"],
    "all": [model["id"] for model in MODELS],
}

QUANT_VARIANTS = [
    {"name": "Q8_0", "method": "RTN"},
    {"name": "Q4_K_M", "method": "K-Quant"},
    {"name": "Q4_0", "method": "RTN"},
    {"name": "Q2_K", "method": "K-Quant"},
]

FIXED_PARAMS = {
    "temperature": 0.2,
    "num_ctx": 2048,
    "num_predict": 128,
    "top_p": 0.90,
    "repeat_penalty": 1.05,
    "stream": True,
}

# Stara sieć (dom): 192.168.1.173 / mill-56-rpi.local / mill-56-rpi
DEFAULT_RPI_OLLAMA_URLS = (
    "http://172.20.10.2:11434",
)
# Stara sieć (dom): 192.168.1.173:9000 / mill-56-rpi.local:9000 / mill-56-rpi:9000
DEFAULT_RPI_METRICS_URLS = (
    "http://172.20.10.2:9000",
)

REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "120"))
MODEL_LOAD_TIMEOUT = int(os.environ.get("OLLAMA_MODEL_LOAD_TIMEOUT", "900"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

E2E_NORM_CAP_MS = int(
    os.environ.get(
        "E2E_NORM_CAP_MS",
        str(max(90_000, REQUEST_TIMEOUT * 1000 + 30_000)),
    )
)
TTFT_NORM_CAP_MS = int(
    os.environ.get(
        "TTFT_NORM_CAP_MS",
        str(max(45_000, REQUEST_TIMEOUT * 1000)),
    )
)

W_FAITHFULNESS = 0.40
W_RELEVANCY = 0.30
W_CONCISENESS = 0.20
W_LATENCY = 0.10
W_LATENCY_E2E = W_LATENCY / 2.0
W_LATENCY_TTFT = W_LATENCY / 2.0

# Stara sieć: krisqma@mill-56-rpi
DEFAULT_RPI_SSH = "krisqma@172.20.10.2"
DEFAULT_REMOTE_MODELS_DIR = "/home/krisqma/quant/models"
REMOTE_OLLAMA_HOST = os.environ.get("QUANT_REMOTE_OLLAMA_HOST", "127.0.0.1:11434")

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


# ── Helpers copied/adapted from run_benchmark.py ──────────────────────────────

def composite_score(
    faithfulness: float,
    answer_relevancy: float,
    conciseness: float,
    e2e_ms: float,
    ttft_ms: float | None,
    hallucination_flag: bool,
) -> float:
    if hallucination_flag:
        return 0.0
    norm_e2e = min(e2e_ms / E2E_NORM_CAP_MS, 1.0)
    norm_ttft = 1.0 if ttft_ms is None else min(ttft_ms / TTFT_NORM_CAP_MS, 1.0)
    latency_bonus = (
        W_LATENCY_E2E * (1.0 - norm_e2e) + W_LATENCY_TTFT * (1.0 - norm_ttft)
    )
    return (
        W_FAITHFULNESS * faithfulness
        + W_RELEVANCY * answer_relevancy
        + W_CONCISENESS * conciseness
        + latency_bonus
    )


def configured_urls(env_name: str, defaults: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(env_name, "").strip()
    urls: list[str] = list(defaults)
    if raw:
        for url in (part.strip().rstrip("/") for part in raw.split(",") if part.strip()):
            if url not in urls:
                urls.append(url)
    return urls


def rpi_ollama_urls() -> list[str]:
    return configured_urls("RPI_OLLAMA_URL", DEFAULT_RPI_OLLAMA_URLS)


def rpi_metrics_urls() -> list[str]:
    return configured_urls("RPI_METRICS_URL", DEFAULT_RPI_METRICS_URLS)


def get_rpi_metrics(metrics_urls: list[str]) -> dict[str, Any]:
    for metrics_url in metrics_urls:
        try:
            resp = httpx.get(f"{metrics_url}/metrics", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {
                "ram_mb": round(float(data.get("ram_mb", 0.0)), 1),
                "cpu_temp_c": round(float(data.get("cpu_temp_c", 0.0)), 1),
                "throttling": bool(data.get("throttling", False)),
            }
        except Exception:
            continue
    return {"ram_mb": 0.0, "cpu_temp_c": 0.0, "throttling": False}


def _model_to_safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


# ── Quant runner ──────────────────────────────────────────────────────────────

class QuantBenchmarkRunner:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        dry_run: bool = False,
        models: list[str] | None = None,
        variants: list[str] | None = None,
    ):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.run_id = run_id or os.environ.get("QUANT_RUN_ID") or datetime.now(
            timezone.utc
        ).strftime("quant_%Y%m%d_%H%M%S")
        self.dry_run = dry_run
        self.ollama_urls = rpi_ollama_urls()
        self.metrics_urls = rpi_metrics_urls()
        self.rpi_ssh = os.environ.get("QUANT_RPI_SSH", DEFAULT_RPI_SSH)
        self.remote_models_dir = os.environ.get(
            "QUANT_REMOTE_MODELS_DIR", DEFAULT_REMOTE_MODELS_DIR
        ).rstrip("/")
        self.golden = self._load_golden()
        if dry_run:
            self.golden = self.golden[:2]

        selected_models = set(models) if models else None
        selected_variants = set(variants) if variants else None
        self.models = [m for m in MODELS if selected_models is None or m["id"] in selected_models]
        variant_by_name = {v["name"]: v for v in QUANT_VARIANTS}
        self.variants = (
            [variant_by_name[name] for name in variants if name in variant_by_name]
            if variants
            else list(QUANT_VARIANTS)
        )

        if selected_models:
            missing = selected_models - {m["id"] for m in self.models}
            if missing:
                raise SystemExit(f"Unknown model(s): {', '.join(sorted(missing))}")
        if selected_variants:
            missing = selected_variants - {v["name"] for v in self.variants}
            if missing:
                raise SystemExit(f"Unknown variant(s): {', '.join(sorted(missing))}")

        if dry_run:
            self.models = self.models[:1]
            self.variants = self.variants[:1]

    def _load_golden(self) -> list[dict[str, Any]]:
        golden_path = self.base_dir / "golden.jsonl"
        if not golden_path.exists():
            raise SystemExit("Błąd: golden.jsonl nie istnieje. Uruchom najpierw generate_golden.py")
        records = []
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _ssh(
        self,
        command: str,
        *,
        input_text: str | None = None,
        timeout: int = 30,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["ssh", self.rpi_ssh, command],
            input=input_text,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            detail = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(f"ssh {self.rpi_ssh} {command!r} failed: {detail}")
        return result

    def _remote_gguf_path(self, model_id: str, variant_name: str) -> str:
        return f"{self.remote_models_dir}/{model_id}/{model_id}-{variant_name}.gguf"

    def _remote_file_exists(self, remote_path: str) -> bool:
        result = self._ssh(f"test -s {shlex.quote(remote_path)}", check=False)
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"could not check remote GGUF {remote_path}: {detail}")

    def _remote_file_size_gb(self, remote_path: str) -> float | None:
        try:
            result = self._ssh(f"stat -c %s {shlex.quote(remote_path)}", check=True)
            size_bytes = int(result.stdout.strip())
            return round(size_bytes / (1024**3), 3)
        except Exception as exc:
            logging.warning("Could not read remote GGUF size for %s: %s", remote_path, exc)
            return None

    def _create_ollama_model(self, remote_gguf_path: str, model_tag: str, template_key: str) -> None:
        remote_modelfile = f"/tmp/{model_tag}.Modelfile"
        if template_key == "gemma4":
            modelfile = f"FROM {remote_gguf_path}\n{GEMMA4_MODELFILE_SUFFIX}"
        else:
            modelfile = f'FROM {remote_gguf_path}\nTEMPLATE """{TEMPLATES[template_key]}"""\n'
        self._ssh(
            f"cat > {shlex.quote(remote_modelfile)}",
            input_text=modelfile,
            timeout=30,
            check=True,
        )
        try:
            create_cmd = (
                f"OLLAMA_HOST={shlex.quote(REMOTE_OLLAMA_HOST)} "
                f"ollama create {shlex.quote(model_tag)} -f {shlex.quote(remote_modelfile)}"
            )
            self._ssh(create_cmd, timeout=MODEL_LOAD_TIMEOUT, check=True, capture_output=False)
        finally:
            self._ssh(f"rm -f {shlex.quote(remote_modelfile)}", check=False)

    def _remove_ollama_model(self, model_tag: str) -> None:
        rm_cmd = f"OLLAMA_HOST={shlex.quote(REMOTE_OLLAMA_HOST)} ollama rm {shlex.quote(model_tag)}"
        self._ssh(rm_cmd, timeout=60, check=False)

    def _call_ollama(
        self,
        model_tag: str,
        system_prompt: str,
        question: str,
        api: str,
    ) -> tuple[str, float, float | None, dict[str, Any] | None, str | None]:
        options = {k: v for k, v in FIXED_PARAMS.items() if k != "stream"}
        if api == "chat":
            endpoint = "/api/chat"
            payload = {
                "model": model_tag,
                "messages": [
                    {"role": "system", "content": f"/no_think\n{system_prompt}"},
                    {"role": "user", "content": question},
                ],
                "think": False,
                "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": options,
            }
        else:
            endpoint = "/api/generate"
            payload = {
                "model": model_tag,
                "prompt": question,
                "system": system_prompt,
                "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": options,
            }
        timeout = httpx.Timeout(REQUEST_TIMEOUT)
        started = time.monotonic()
        last_error: Exception | None = None

        for ollama_url in self.ollama_urls:
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
                if final_data and final_data.get("error"):
                    answer = "".join(answer_parts).strip()
                    return answer, e2e_ms, ttft_ms, final_data, f"ollama_error: {final_data['error']}"
                return "".join(answer_parts).strip(), e2e_ms, ttft_ms, final_data, None
            except httpx.TimeoutException as exc:
                answer = "".join(answer_parts).strip()
                return answer, float(REQUEST_TIMEOUT * 1000), ttft_ms, final_data, f"ollama_timeout: {exc}"
            except httpx.TransportError as exc:
                if answer_parts or ttft_ms is not None:
                    e2e_ms = (time.monotonic() - started) * 1000
                    answer = "".join(answer_parts).strip()
                    return answer, e2e_ms, ttft_ms, final_data, f"ollama_transport_error: {exc}"
                last_error = exc
                continue
            except Exception as exc:
                e2e_ms = (time.monotonic() - started) * 1000
                answer = "".join(answer_parts).strip()
                return answer, e2e_ms, ttft_ms, final_data, f"ollama_error: {exc}"

        e2e_ms = (time.monotonic() - started) * 1000
        return "", e2e_ms, None, None, f"ollama_transport_error: {last_error}"

    def _warmup(self, model_tag: str, api: str) -> None:
        first_case = self.golden[0]
        answer, _, _, _, error = self._call_ollama(
            model_tag,
            first_case["system_prompt"],
            first_case["question"],
            api,
        )
        if error or not answer:
            logging.warning("Warmup failed for %s: %s", model_tag, error or "empty answer")

    @staticmethod
    def _timeout_eval() -> dict[str, Any]:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "conciseness": 0.0,
            "polish_quality": 0.0,
            "hallucination_flag": True,
            "haiku_raw": "",
        }

    def _append_record(self, out_file, record: dict[str, Any]) -> None:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        os.fsync(out_file.fileno())

    def _completed_case_ids(self, output_path: Path, model_id: str, variant_name: str) -> set[str]:
        if not output_path.exists():
            return set()
        completed: set[str] = set()
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("run_id") == self.run_id
                    and record.get("model") == model_id
                    and record.get("quant_variant") == variant_name
                    and isinstance(record.get("case_id"), str)
                ):
                    completed.add(record["case_id"])
        return completed

    def _run_case(
        self,
        *,
        model_id: str,
        model_tag: str,
        api: str,
        variant: dict[str, str],
        gguf_size_gb: float | None,
        case: dict[str, Any],
        index: int,
        total: int,
        out_file,
    ) -> tuple[float, float, bool]:
        answer, e2e_ms, ttft_ms, _, error = self._call_ollama(
            model_tag,
            case["system_prompt"],
            case["question"],
            api,
        )
        metrics = get_rpi_metrics(self.metrics_urls)

        timed_out = e2e_ms >= float(REQUEST_TIMEOUT * 1000) - 0.001
        if timed_out or not answer.strip():
            eval_result = self._timeout_eval()
        else:
            eval_result = evaluate.evaluate_response(
                system_prompt=case["system_prompt"],
                question=case["question"],
                golden_answer=case["golden_answer"],
                model_answer=answer,
                category=case["category"],
            )

        score = composite_score(
            eval_result["faithfulness"],
            eval_result["answer_relevancy"],
            eval_result["conciseness"],
            e2e_ms,
            ttft_ms,
            eval_result["hallucination_flag"],
        )

        record = {
            "run_id": self.run_id,
            "model": model_id,
            "quant_variant": variant["name"],
            "quant_method": variant["method"],
            "gguf_size_gb": gguf_size_gb,
            "temperature": FIXED_PARAMS["temperature"],
            "num_ctx": FIXED_PARAMS["num_ctx"],
            "num_predict": FIXED_PARAMS["num_predict"],
            "top_p": FIXED_PARAMS["top_p"],
            "repeat_penalty": FIXED_PARAMS["repeat_penalty"],
            "case_id": case["case_id"],
            "category": case["category"],
            "question": case["question"],
            "golden_answer": case["golden_answer"],
            "model_answer": answer,
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "ram_mb": metrics["ram_mb"],
            "cpu_temp_c": metrics["cpu_temp_c"],
            "throttling": metrics["throttling"],
            "faithfulness": eval_result["faithfulness"],
            "answer_relevancy": eval_result["answer_relevancy"],
            "conciseness": eval_result["conciseness"],
            "polish_quality": eval_result["polish_quality"],
            "hallucination_flag": eval_result["hallucination_flag"],
            "composite_score": round(score, 4),
        }
        if error:
            record["error"] = error

        self._append_record(out_file, record)
        ttft_text = "null" if ttft_ms is None else f"{ttft_ms:.0f}ms"
        gen_text = "null" if ttft_ms is None else f"{max(e2e_ms - ttft_ms, 0.0):.0f}ms"
        print(
            f"[{model_id} / {variant['name']}] case {index}/{total} — "
            f"ttft={ttft_text} e2e={e2e_ms:.0f}ms gen={gen_text} "
            f"hall={eval_result['hallucination_flag']} score={score:.3f}",
            flush=True,
        )
        return e2e_ms, score, bool(eval_result["hallucination_flag"])

    def benchmark_variant(
        self,
        model: dict[str, str],
        variant: dict[str, str],
        output_path: Path,
        out_file,
    ) -> None:
        model_id = model["id"]
        variant_name = variant["name"]
        prefix = f"[{model_id} / {variant_name}]"
        remote_gguf_path = self._remote_gguf_path(model_id, variant_name)

        if not self._remote_file_exists(remote_gguf_path):
            print(f"{prefix} WARN: missing GGUF on RPi: {remote_gguf_path}", flush=True)
            return

        gguf_size_gb = self._remote_file_size_gb(remote_gguf_path)
        model_tag = _model_to_safe(f"quant-{model_id}-{variant_name}-{self.run_id}").lower()
        e2e_values: list[float] = []
        scores: list[float] = []
        hallucinations = 0
        completed_case_ids = self._completed_case_ids(output_path, model_id, variant_name)
        if completed_case_ids:
            print(
                f"{prefix} Resume: skipping {len(completed_case_ids)} completed case(s)",
                flush=True,
            )

        try:
            print(f"{prefix} Loading GGUF into Ollama: {remote_gguf_path}", flush=True)
            self._create_ollama_model(remote_gguf_path, model_tag, model["template"])
            print(f"{prefix} Warmup...", flush=True)
            self._warmup(model_tag, model["api"])

            total = len(self.golden)
            for index, case in enumerate(self.golden, start=1):
                if case["case_id"] in completed_case_ids:
                    continue
                e2e_ms, score, hallucinated = self._run_case(
                    model_id=model_id,
                    model_tag=model_tag,
                    api=model["api"],
                    variant=variant,
                    gguf_size_gb=gguf_size_gb,
                    case=case,
                    index=index,
                    total=total,
                    out_file=out_file,
                )
                e2e_values.append(e2e_ms)
                scores.append(score)
                if hallucinated:
                    hallucinations += 1

            median_e2e = statistics.median(e2e_values) if e2e_values else 0.0
            mean_score = mean(scores) if scores else 0.0
            hall_rate = (hallucinations / len(e2e_values) * 100.0) if e2e_values else 0.0
            print(
                f"{prefix} DONE — median_e2e={median_e2e:.0f}ms "
                f"mean_score={mean_score:.3f} hall_rate={hall_rate:.1f}%",
                flush=True,
            )
        finally:
            print(f"{prefix} Removing Ollama model...", flush=True)
            self._remove_ollama_model(model_tag)

    def run_all(self) -> Path:
        out_path = self.results_dir / f"quant_results_{self.run_id}.jsonl"
        print(f"[quant] run_id={self.run_id}", flush=True)
        print(f"[quant] output={out_path}", flush=True)
        print(f"[quant] rpi_ssh={self.rpi_ssh}", flush=True)
        print(f"[quant] remote_models_dir={self.remote_models_dir}", flush=True)
        print(f"[quant] ollama_urls={', '.join(self.ollama_urls)}", flush=True)
        print(f"[quant] metrics_urls={', '.join(self.metrics_urls)}", flush=True)
        print(f"[quant] golden_cases={len(self.golden)}", flush=True)

        with out_path.open("a", encoding="utf-8") as out_file:
            for model in self.models:
                for variant in self.variants:
                    self.benchmark_variant(model, variant, out_path, out_file)
        print("[quant] finished", flush=True)
        return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Wilga quantization benchmark")
    parser.add_argument("--run-id", type=str, default=None, help="ID runu do nazwy pliku wynikowego")
    parser.add_argument(
        "--model-group",
        choices=sorted(MODEL_GROUPS),
        default=None,
        help="Wygodny wybór rodziny modeli: gemma3, gemma4, bielik albo all",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Podzbiór modeli po przecinku, np. gemma3-4b,bielik-4.5b",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Podzbiór wariantów po przecinku, np. Q8_0,Q4_K_M",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pierwszy model, pierwszy wariant, pierwsze 2 golden cases",
    )
    args = parser.parse_args()

    if args.model_group and args.models:
        raise SystemExit("Use either --model-group or --models, not both")
    models = MODEL_GROUPS[args.model_group] if args.model_group else parse_csv(args.models)

    runner = QuantBenchmarkRunner(
        run_id=args.run_id,
        dry_run=args.dry_run,
        models=models,
        variants=parse_csv(args.variants),
    )
    runner.run_all()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
