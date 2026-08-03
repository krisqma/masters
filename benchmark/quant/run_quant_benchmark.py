#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


GOLDEN_PATH = ROOT.parent / "data" / "golden.jsonl"

# Stara sieć (dom): http://mill-56-rpi.local:11434
OLLAMA_URL = os.environ.get("QUANT_OLLAMA_URL", "http://172.20.10.2:11434").rstrip("/")
# Stara sieć (dom): http://mill-56-rpi.local:9090/metrics
METRICS_URL = os.environ.get("QUANT_METRICS_URL", "http://172.20.10.2:9090/metrics")
RESULTS_DIR = Path(os.environ.get("QUANT_RESULTS_DIR", str(ROOT / "results")))
RUN_ID = os.environ.get("QUANT_RUN_ID", time.strftime("quant_%Y%m%d_%H%M%S"))

# Stara sieć (dom): mill-56-rpi.local
OLLAMA_HOSTNAME = urlparse(OLLAMA_URL).hostname or "172.20.10.2"
RPI_SSH = os.environ.get("QUANT_RPI_SSH", f"pi@{OLLAMA_HOSTNAME}")
REMOTE_MODELS_DIR = os.environ.get("QUANT_REMOTE_MODELS_DIR", "/home/pi/quant/models").rstrip("/")
REMOTE_OLLAMA_HOST = os.environ.get("QUANT_REMOTE_OLLAMA_HOST", "127.0.0.1:11434")

REQUEST_TIMEOUT_S = 120
JUDGE_TIMEOUT_S = 60
JUDGE_RETRIES = 2
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
HAIKU_MODEL = os.environ.get("HAIKU_MODEL", "anthropic/claude-3-5-haiku")

MODELS = [
    {"id": "gemma3-4b", "template": "gemma"},
    {"id": "gemma4-e2b", "template": "gemma"},
    {"id": "bielik-4.5b", "template": "bielik"},
]

QUANT_VARIANTS = [
    {"name": "Q8_0", "method": "legacy", "nominal_bits": 8, "analysis_role": "high_precision_baseline"},
    {"name": "Q5_0", "method": "legacy", "nominal_bits": 5, "analysis_role": "method_comparison"},
    {"name": "Q5_K_M", "method": "K-Quant", "nominal_bits": 5, "analysis_role": "method_comparison"},
    {"name": "Q4_0", "method": "legacy", "nominal_bits": 4, "analysis_role": "method_comparison"},
    {"name": "Q4_K_M", "method": "K-Quant", "nominal_bits": 4, "analysis_role": "method_comparison"},
    {"name": "Q2_K", "method": "K-Quant", "nominal_bits": 2, "analysis_role": "aggressive_compression"},
]

FIXED_PARAMS = {
    "temperature": 0.2,
    "num_ctx": 2048,
    "num_predict": 128,
    "top_p": 0.90,
    "repeat_penalty": 1.05,
    "stream": False,
}

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


class OllamaTimeoutError(RuntimeError):
    pass


class JudgeEvaluationError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run_checked(command: list[str], input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=input_text is not None, check=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise OllamaTimeoutError(str(exc)) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise OllamaTimeoutError(str(exc)) from exc
        raise


def get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def send_request(model_tag: str, prompt: str, system: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_tag,
        "prompt": prompt,
        "stream": FIXED_PARAMS["stream"],
        "options": {k: v for k, v in FIXED_PARAMS.items() if k != "stream"},
    }
    if system:
        payload["system"] = system
    return post_json(f"{OLLAMA_URL}/api/generate", payload, REQUEST_TIMEOUT_S)


def remote_model_path(model_id: str, quant_variant: str) -> str:
    return f"{REMOTE_MODELS_DIR}/{model_id}/{model_id}-{quant_variant}.gguf"


def remote_file_exists(remote_path: str) -> bool:
    result = subprocess.run(["ssh", RPI_SSH, f"test -s {shlex.quote(remote_path)}"], check=False)
    return result.returncode == 0


def ollama_create_on_rpi(remote_gguf_path: str, model_tag: str, template_key: str) -> None:
    remote_modelfile = f"/tmp/{model_tag}.Modelfile"
    modelfile = f'FROM {remote_gguf_path}\nTEMPLATE """{TEMPLATES[template_key]}"""\n'
    run_checked(["ssh", RPI_SSH, f"cat > {shlex.quote(remote_modelfile)}"], input_text=modelfile)
    try:
        create_cmd = (
            f"OLLAMA_HOST={shlex.quote(REMOTE_OLLAMA_HOST)} "
            f"ollama create {shlex.quote(model_tag)} -f {shlex.quote(remote_modelfile)}"
        )
        run_checked(["ssh", RPI_SSH, create_cmd])
    finally:
        run_checked(["ssh", RPI_SSH, f"rm -f {shlex.quote(remote_modelfile)}"])


def ollama_rm_on_rpi(model_tag: str) -> None:
    rm_cmd = f"OLLAMA_HOST={shlex.quote(REMOTE_OLLAMA_HOST)} ollama rm {shlex.quote(model_tag)}"
    subprocess.run(["ssh", RPI_SSH, rm_cmd], check=False)


def tokens_per_second(ollama_response: dict[str, Any]) -> float | None:
    eval_count = ollama_response.get("eval_count")
    eval_duration = ollama_response.get("eval_duration")
    if not eval_count or not eval_duration:
        return None
    seconds = float(eval_duration) / 1_000_000_000
    return None if seconds <= 0 else float(eval_count) / seconds


def judge_message(system: str, user: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise JudgeEvaluationError("OPENROUTER_API_KEY is not set")
    payload = {
        "model": HAIKU_MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost/quant-benchmark",
        "X-Title": "Quant Benchmark",
    }
    last_error: Exception | None = None
    for attempt in range(JUDGE_RETRIES + 1):
        try:
            result = post_json(f"{OPENROUTER_BASE_URL}/chat/completions", payload, JUDGE_TIMEOUT_S, headers)
            choices = result.get("choices", [])
            if not choices:
                raise JudgeEvaluationError(f"OpenRouter returned no choices: {result}")
            return str(choices[0].get("message", {}).get("content", "")).strip()
        except Exception as exc:
            last_error = exc
            if attempt < JUDGE_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise JudgeEvaluationError(str(last_error))


def parse_score(text: str) -> float:
    cleaned = text.strip().replace(",", ".")
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for key in ("score", "value", "result"):
                if key in data:
                    return max(0.0, min(1.0, float(data[key])))
    except json.JSONDecodeError:
        pass
    for token in cleaned.replace(":", " ").replace("=", " ").split():
        try:
            return max(0.0, min(1.0, float(token)))
        except ValueError:
            continue
    raise JudgeEvaluationError(f"cannot parse score from: {text!r}")


def parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    try:
        data = json.loads(lowered)
        if isinstance(data, bool):
            return data
        if isinstance(data, dict):
            for key in ("hallucination_flag", "hallucination", "flag", "result"):
                if key in data:
                    return bool(data[key])
    except json.JSONDecodeError:
        pass
    if "true" in lowered or "tak" in lowered or "yes" in lowered:
        return True
    if "false" in lowered or "nie" in lowered or "no" in lowered:
        return False
    raise JudgeEvaluationError(f"cannot parse bool from: {text!r}")


def evaluate_with_haiku(question: str, context: str, answer: str) -> dict[str, Any]:
    judge_system = "You are a strict benchmark judge. Return only valid JSON."
    faithfulness = parse_score(judge_message(judge_system, f'Return {{"score": number}} for faithfulness to context.\nQuestion:\n{question}\nContext:\n{context}\nAnswer:\n{answer}'))
    answer_relevancy = parse_score(judge_message(judge_system, f'Return {{"score": number}} for answer relevancy.\nQuestion:\n{question}\nAnswer:\n{answer}'))
    conciseness = parse_score(judge_message(judge_system, f'Return {{"score": number}} for conciseness.\nAnswer:\n{answer}'))
    hallucination_flag = parse_bool(judge_message(judge_system, f'Return {{"hallucination_flag": true|false}}. Flag true if answer invents or contradicts context.\nContext:\n{context}\nAnswer:\n{answer}'))
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "conciseness": conciseness,
        "hallucination_flag": hallucination_flag,
    }


def get_rpi_metrics() -> dict[str, Any]:
    try:
        data = get_json(METRICS_URL)
    except Exception as exc:
        log(f"[metrics] WARN: {exc}")
        return {"ram_mb": None, "cpu_temp_c": None, "throttling": None}
    return {
        "ram_mb": data.get("ram_mb", data.get("memory_mb", data.get("ram"))),
        "cpu_temp_c": data.get("cpu_temp_c", data.get("temperature_c", data.get("temp_c"))),
        "throttling": data.get("throttling", data.get("is_throttling")),
    }


def compute_composite(faithfulness: float | None, answer_relevancy: float | None, conciseness: float | None, e2e_ms: float, hallucination_flag: bool | None) -> float | None:
    if hallucination_flag:
        return 0.0
    if faithfulness is None or answer_relevancy is None or conciseness is None:
        return None
    norm_latency = min(e2e_ms / 20000, 1.0)
    return 0.40 * faithfulness + 0.30 * answer_relevancy + 0.20 * conciseness + 0.10 * (1 - norm_latency)


def case_value(case: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in case:
            return case[key]
    return default


def timeout_scores() -> dict[str, Any]:
    return {"faithfulness": 0.0, "answer_relevancy": 0.0, "conciseness": 0.0, "hallucination_flag": True}


def null_scores() -> dict[str, Any]:
    return {"faithfulness": None, "answer_relevancy": None, "conciseness": None, "hallucination_flag": None}


def benchmark_variant(model: dict[str, str], variant: dict[str, str], golden_cases: list[dict[str, Any]], output_path: Path) -> None:
    model_id = model["id"]
    variant_name = variant["name"]
    remote_gguf_path = remote_model_path(model_id, variant_name)
    if not remote_file_exists(remote_gguf_path):
        log(f"[{model_id} / {variant_name}] WARN: missing remote GGUF, run upload_models_to_rpi.py first: {remote_gguf_path}")
        return

    model_tag = f"quant-{model_id}-{variant_name}".lower()
    prefix = f"[{model_id} / {variant_name}]"
    scores_for_summary: list[float] = []
    e2e_for_summary: list[float] = []
    hallucinations = 0
    completed = 0

    try:
        log(f"{prefix} Ładowanie modelu na RPi...")
        ollama_create_on_rpi(remote_gguf_path, model_tag, model["template"])
        log(f"{prefix} Warmup...")
        try:
            send_request(model_tag, "OK")
        except Exception as exc:
            log(f"{prefix} WARN: warmup failed: {exc}")

        for index, case in enumerate(golden_cases, start=1):
            question = str(case_value(case, "question", default=""))
            context = str(case_value(case, "context", "system", default=""))
            golden_answer = case_value(case, "golden_answer", "golden", "answer", default=None)
            start = time.time()
            answer = ""
            tps = None
            error: str | None = None
            try:
                response = send_request(model_tag, question, system=context)
                e2e_ms = (time.time() - start) * 1000
                answer = response.get("response", "")
                tps = tokens_per_second(response)
                scores = evaluate_with_haiku(question=question, context=context, answer=answer)
            except OllamaTimeoutError as exc:
                e2e_ms = 120000.0
                scores = timeout_scores()
                error = f"ollama_timeout: {exc}"
            except JudgeEvaluationError as exc:
                e2e_ms = (time.time() - start) * 1000
                scores = null_scores()
                error = f"judge_error: {exc}"
            except Exception as exc:
                e2e_ms = (time.time() - start) * 1000
                scores = null_scores()
                error = f"unexpected_error: {exc}"

            metrics = get_rpi_metrics()
            composite_score = compute_composite(scores["faithfulness"], scores["answer_relevancy"], scores["conciseness"], e2e_ms, scores["hallucination_flag"])
            record = {
                "run_id": RUN_ID,
                "model_name": model_id,
                "quant_variant": variant_name,
                "quant_method": variant["method"],
                "nominal_bits": variant["nominal_bits"],
                "analysis_role": variant["analysis_role"],
                "case_id": case_value(case, "case_id", "id", default=f"case_{index:02d}"),
                "category": case_value(case, "category", default=None),
                "question": question,
                "answer": answer,
                "golden_answer": golden_answer,
                "e2e_ms": e2e_ms,
                "tokens_per_s": tps,
                "ram_mb": metrics["ram_mb"],
                "cpu_temp_c": metrics["cpu_temp_c"],
                "throttling": metrics["throttling"],
                **scores,
                "polish_quality": None,
                "composite_score": composite_score,
            }
            if error:
                record["error"] = error
            append_jsonl(output_path, record)

            completed += 1
            e2e_for_summary.append(e2e_ms)
            if composite_score is not None:
                scores_for_summary.append(composite_score)
            if scores["hallucination_flag"]:
                hallucinations += 1
            score_text = "null" if composite_score is None else f"{composite_score:.3f}"
            log(f"{prefix} case {index:2d}/{len(golden_cases)} — e2e={e2e_ms:.0f}ms  score={score_text}  hall={scores['hallucination_flag']}")

        median_e2e = statistics.median(e2e_for_summary) if e2e_for_summary else 0.0
        mean_score = statistics.mean(scores_for_summary) if scores_for_summary else 0.0
        hall_rate = (hallucinations / completed * 100) if completed else 0.0
        log(f"{prefix} DONE — median_e2e={median_e2e:.0f}ms  mean_score={mean_score:.3f}  hall_rate={hall_rate:.1f}%")
    finally:
        log(f"{prefix} Usuwam model z Ollama na RPi...")
        ollama_rm_on_rpi(model_tag)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"quant_results_{RUN_ID}.jsonl"
    golden_cases = load_jsonl(GOLDEN_PATH)
    log(f"[quant] run_id={RUN_ID}")
    log(f"[quant] ollama_url={OLLAMA_URL}")
    log(f"[quant] metrics_url={METRICS_URL}")
    log(f"[quant] judge_base_url={OPENROUTER_BASE_URL}")
    log(f"[quant] judge_model={HAIKU_MODEL}")
    log(f"[quant] results={output_path}")
    log(f"[quant] rpi_ssh={RPI_SSH}")
    log(f"[quant] remote_models_dir={REMOTE_MODELS_DIR}")
    log(f"[quant] remote_ollama_host={REMOTE_OLLAMA_HOST}")
    log(f"[quant] golden_cases={len(golden_cases)}")
    for model in MODELS:
        for variant in QUANT_VARIANTS:
            benchmark_variant(model, variant, golden_cases, output_path)
    log("[quant] finished")


if __name__ == "__main__":
    main()
