#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import httpx
import optuna
from tqdm import tqdm

import evaluate
import report

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    "batiai/gemma4-e2b:q4",
    "gemma4:e2b",
    "gemma2:2b",
    "gemma3:4b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "phi3.5",
]

N_TRIALS = 30

DEFAULT_NUM_CTX_CHOICES = [1024, 2048, 4096]
MODEL_NUM_CTX_CHOICES = {
    # gemma4:e2b on RPi repeatedly times out at larger contexts and can poison
    # the following trial until the model is unloaded.
    "gemma4:e2b": [1024, 2048],
}
MAX_CONSECUTIVE_EMPTY_TIMEOUTS = 3

DEFAULT_RPI_OLLAMA_URLS = (
    "http://192.168.1.173:11434",
    "http://mill-56-rpi.local:11434",
    "http://mill-56-rpi:11434",
)
DEFAULT_RPI_METRICS_URLS = (
    "http://192.168.1.173:9000",
    "http://mill-56-rpi.local:9000",
    "http://mill-56-rpi:9000",
)

# HTTP client: hard limit jak długo czekamy na pełny stream (sekundy).
# 7B GGUF on RPi can spend over a minute in prompt evaluation before the first
# token, so keep this configurable instead of treating 60s as universal.
REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "180"))
MODEL_LOAD_TIMEOUT = 240
OLLAMA_KEEP_ALIVE = "10m"

# Normalizacja **e2e** w funkcji celu — musi być > REQUEST_TIMEOUT × 1000 ms,
# inaczej każdy timeout wpada w ten sam „najgorszy'' punkt skali i TPE się myli.
E2E_NORM_CAP_MS = int(
    os.environ.get(
        "E2E_NORM_CAP_MS",
        str(max(90_000, REQUEST_TIMEOUT * 1000 + 30_000)),
    )
)

# Normalizacja TTFT (czas do pierwszego chunka treści przy streamingu).
TTFT_NORM_CAP_MS = int(
    os.environ.get(
        "TTFT_NORM_CAP_MS",
        str(max(45_000, REQUEST_TIMEOUT * 1000)),
    )
)

W_FAITHFULNESS = 0.40
W_RELEVANCY    = 0.30
W_CONCISENESS  = 0.20
W_LATENCY      = 0.10
W_LATENCY_E2E  = W_LATENCY / 2.0   # połowa wagi czasu na pełny odpowiedź
W_LATENCY_TTFT = W_LATENCY / 2.0   # połowa na „reaktywność'' pierwszego tokenu

# ── Helpers ───────────────────────────────────────────────────────────────────

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
        + W_RELEVANCY   * answer_relevancy
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


def get_rpi_metrics(metrics_urls: list[str]) -> dict:
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


def get_ollama_tags(ollama_urls: list[str]) -> tuple[str, dict]:
    last_error: Exception | None = None
    for ollama_url in ollama_urls:
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return ollama_url, resp.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Ollama niedostępna pod {ollama_urls}: {last_error}")


def check_model_available(ollama_urls: list[str], model: str) -> bool:
    try:
        _, tags = get_ollama_tags(ollama_urls)
        names = [m["name"] for m in tags.get("models", [])]
        # normalize: treat "phi3.5" and "phi3.5:latest" as equivalent
        def _base(name: str) -> str:
            return name[: -len(":latest")] if name.endswith(":latest") else name
        model_base = _base(model)
        return any(_base(n) == model_base for n in names)
    except Exception:
        return False


def get_model_capabilities_for_url(ollama_url: str, model: str) -> set[str]:
    try:
        resp = httpx.post(f"{ollama_url}/api/show", json={"model": model}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {str(cap) for cap in data.get("capabilities", []) if isinstance(cap, str)}
    except Exception:
        return set()


def uses_completion_api_from_capabilities(capabilities: set[str]) -> bool:
    return "completion" in capabilities and "chat" not in capabilities


def _model_to_safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def num_ctx_choices_for_model(model: str) -> list[int]:
    return MODEL_NUM_CTX_CHOICES.get(model, DEFAULT_NUM_CTX_CHOICES)


def benchmark_model_list(include_env_extras: bool = True) -> list[str]:
    """Domyślna lista modeli + opcjonalne dopiski z ENV (PL / dodatkowe)."""
    names: list[str] = list(MODELS)
    if not include_env_extras:
        return names

    extras = []
    polish = os.environ.get("POLISH_OLLAMA_MODEL", "").strip()
    if polish:
        extras.append(polish)

    bulk = os.environ.get("EXTRA_OLLAMA_MODELS", "").strip()
    if bulk:
        extras.extend(p.strip() for p in bulk.split(",") if p.strip())

    for m in extras:
        if m not in names:
            names.append(m)
    return names


def parse_model_trials(specs: list[str]) -> dict[str, int]:
    """Parsuje `--model-trials gemma3:4b=30` na mapę nazwa→N."""
    out: dict[str, int] = {}
    for raw in specs:
        if "=" not in raw:
            sys.exit(f"--model-trials wymaga postaci MODEL=N, dostałem: {raw!r}")
        model_part, sep, num_part = raw.rpartition("=")
        if not sep:
            sys.exit(f"--model-trials niepoprawne: {raw!r}")
        model_key = model_part.strip()
        try:
            n = int(num_part.strip())
        except ValueError as exc:
            raise SystemExit(f"--model-trials: nie liczba w {raw!r}") from exc
        if n < 1:
            sys.exit(f"--model-trials: N musi być ≥ 1, dostałem {n}")
        out[model_key] = n
    return out


def purge_optuna_studies_for_models(storage_uri: str, models: list[str]) -> list[str]:
    """Usuwa wszystkie studia których `study_name` zaczyna się od `{safe(model)}_`."""
    removed: list[str] = []
    prefixes = tuple(_model_to_safe(m) + "_" for m in models)
    for summary in optuna.study.get_all_study_summaries(storage=storage_uri):
        name = summary.study_name
        if any(name.startswith(p) for p in prefixes):
            optuna.delete_study(study_name=name, storage=storage_uri)
            removed.append(name)
    return sorted(removed)


# ── --test mode ───────────────────────────────────────────────────────────────

def run_test(models: list[str] | None = None) -> int:
    ollama_urls = rpi_ollama_urls()
    metrics_urls = rpi_metrics_urls()
    targets = models if models else benchmark_model_list()

    print("=" * 60)
    print("Wilga Benchmark — sanity check")
    print("=" * 60)

    # Connectivity
    print(f"\nSprawdzam połączenie z RPi ({', '.join(ollama_urls)})...")
    try:
        ollama_url, tags = get_ollama_tags(ollama_urls)
        available_names = [m["name"] for m in tags.get("models", [])]
        print(f"  ✓ Ollama odpowiada pod {ollama_url} — {len(available_names)} modeli na RPi")
    except Exception as exc:
        print(f"  ✗ Ollama niedostępna: {exc}")
        return 1

    print(f"\nSprawdzam metrics server ({', '.join(metrics_urls)})...")
    metrics = get_rpi_metrics(metrics_urls)
    if metrics["ram_mb"] or metrics["cpu_temp_c"]:
        print("  ✓ Metrics server odpowiada")
    else:
        print("  ✗ Metrics server niedostępny")
        print("  (benchmark zadziała, ale metryki RPi będą zerowe)")

    def _base(name: str) -> str:
        return name[: -len(":latest")] if name.endswith(":latest") else name
    available_bases = [_base(n) for n in available_names]

    # Model availability
    print(f"\nSprawdzam dostępność modeli:")
    all_ok = True
    first_available = None
    for model in targets:
        ok = _base(model) in available_bases
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {model}")
        if not ok:
            all_ok = False
        elif first_available is None:
            first_available = model

    # Ping test
    if first_available:
        print(f"\nPing test na {first_available}...")
        t0 = time.time()
        try:
            capabilities = get_model_capabilities_for_url(ollama_url, first_available)
            if uses_completion_api_from_capabilities(capabilities):
                resp = httpx.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": first_available,
                        "prompt": "Odpowiedz jednym słowem: test",
                        "stream": False,
                        "options": {"num_predict": 10},
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                answer_path = ("response",)
            else:
                resp = httpx.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": first_available,
                        "messages": [{"role": "user", "content": "Cześć"}],
                        "stream": False,
                        "options": {"num_predict": 10},
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                answer_path = ("message", "content")
            resp.raise_for_status()
            e2e_ms = (time.time() - t0) * 1000
            data = resp.json()
            if answer_path == ("response",):
                answer = data.get("response", "")
            else:
                answer = data.get("message", {}).get("content", "")
            print(f"  ✓ Odpowiedź w {e2e_ms:.0f}ms: {answer[:80]!r}")
        except Exception as exc:
            print(f"  ✗ Ping nieudany: {exc}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("OK — wszystkie modele gotowe. Możesz uruchomić pełny benchmark.")
        return 0
    else:
        missing = [m for m in targets if _base(m) not in available_bases]
        print(f"BRAK modeli na RPi: {missing}")
        print("Zainstaluj je przez: ollama pull <model>")
        return 1


# ── Benchmark runner ──────────────────────────────────────────────────────────

class BenchmarkRunner:
    def __init__(
        self,
        dry_run: bool = False,
        n_trials: int | None = None,
        run_id: str | None = None,
        model_trials: dict[str, int] | None = None,
    ):
        self.ollama_urls = rpi_ollama_urls()
        self.metrics_urls = rpi_metrics_urls()
        self.dry_run = dry_run
        self.default_n_trials = 2 if dry_run else (n_trials if n_trials is not None else N_TRIALS)
        self.model_trials = dict(model_trials or {})
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.storage = f"sqlite:///{Path(__file__).parent / 'optuna.db'}"
        self.golden = self._load_golden()
        if dry_run:
            self.golden = self.golden[:3]
        self.run_id = run_id if run_id else datetime.now(timezone.utc).strftime("%Y-%m-%d_benchmark_%H%M")
        self.force_non_stream_models: set[str] = set()
        self.model_capabilities: dict[str, set[str]] = {}

    def _load_golden(self) -> list[dict]:
        golden_path = Path(__file__).parent / "golden.jsonl"
        if not golden_path.exists():
            print("Błąd: golden.jsonl nie istnieje. Uruchom najpierw generate_golden.py")
            sys.exit(1)
        records = []
        for line in golden_path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _get_model_capabilities(self, model: str) -> set[str]:
        if model in self.model_capabilities:
            return self.model_capabilities[model]

        last_error: Exception | None = None
        for ollama_url in self.ollama_urls:
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(f"{ollama_url}/api/show", json={"model": model})
                    resp.raise_for_status()
                    data = resp.json()
                capabilities = {
                    str(cap) for cap in data.get("capabilities", []) if isinstance(cap, str)
                }
                self.model_capabilities[model] = capabilities
                return capabilities
            except Exception as exc:
                last_error = exc
                continue

        logging.warning("Could not read capabilities for %s: %s", model, last_error)
        self.model_capabilities[model] = set()
        return set()

    def _uses_completion_api(self, model: str) -> bool:
        capabilities = self._get_model_capabilities(model)
        return uses_completion_api_from_capabilities(capabilities)

    @staticmethod
    def _completion_prompt(system_prompt: str, question: str) -> str:
        return f"{system_prompt}\n\nPytanie: {question}\nOdpowiedź:"

    @staticmethod
    def _clean_model_answer(answer: str) -> str:
        cleaned = answer.strip()
        for token in ("<s>", "</s>"):
            while cleaned.startswith(token):
                cleaned = cleaned[len(token):].lstrip()
            while cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
        return cleaned

    def _call_ollama(
        self, model: str, system_prompt: str, question: str, options: dict
    ) -> tuple[str, float, float | None]:
        """Zwraca (odpowiedź, e2e_ms, ttft_ms). Streaming wymagany dla TTFT."""
        t0 = time.time()
        ttft_ms: float | None = None
        use_completion_api = self._uses_completion_api(model)
        if use_completion_api:
            endpoint = "/api/generate"
            payload_base = {
                "model": model,
                "prompt": self._completion_prompt(system_prompt, question),
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": options,
            }
        else:
            endpoint = "/api/chat"
            payload_base = {
                "model": model,
                "messages": [
                    {"role": "system", "content": f"/no_think\n{system_prompt}"},
                    {"role": "user", "content": question},
                ],
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": options,
            }
        timeout = httpx.Timeout(REQUEST_TIMEOUT)

        def extract_answer_text(data: dict) -> str:
            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            response = data.get("response")
            if isinstance(response, str) and response:
                return response
            return ""

        def write_debug(reason: str, raw_response) -> None:
            debug_path = self.results_dir / "debug_empty_responses.jsonl"
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "model": model,
                "reason": reason,
                "question": question,
                "options": options,
                "raw_response": raw_response,
            }
            with open(debug_path, "a", encoding="utf-8") as debug_file:
                debug_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        def call_non_stream(reason: str) -> tuple[str, float, float | None]:
            payload = dict(payload_base)
            payload["stream"] = False
            last_transport_error: httpx.TransportError | None = None
            for ollama_url in self.ollama_urls:
                try:
                    with httpx.Client(timeout=timeout) as client:
                        resp = client.post(f"{ollama_url}{endpoint}", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                    answer = self._clean_model_answer(extract_answer_text(data))
                    e2e_ms = (time.time() - t0) * 1000
                    if not answer.strip():
                        write_debug(reason, data)
                    return answer, e2e_ms, None
                except httpx.TimeoutException:
                    write_debug("non_stream_timeout", {"timeout_s": REQUEST_TIMEOUT})
                    return "", float(REQUEST_TIMEOUT * 1000), None
                except httpx.TransportError as exc:
                    last_transport_error = exc
                    continue

            if last_transport_error is not None:
                raise last_transport_error
            raise RuntimeError("brak skonfigurowanych URL Ollamy")

        if model in self.force_non_stream_models:
            return call_non_stream("forced_non_stream")

        try:
            parts: list[str] = []
            raw_chunks: list[dict] = []
            payload = dict(payload_base)
            payload["stream"] = True
            last_transport_error: httpx.TransportError | None = None
            for ollama_url in self.ollama_urls:
                try:
                    with httpx.Client(timeout=timeout) as client:
                        with client.stream(
                            "POST",
                            f"{ollama_url}{endpoint}",
                            json=payload,
                        ) as resp:
                            resp.raise_for_status()
                            for line in resp.iter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                err = data.get("error")
                                if err:
                                    raise RuntimeError(str(err))
                                raw_chunks.append(data)
                                delta = extract_answer_text(data)
                                if delta:
                                    if ttft_ms is None:
                                        ttft_ms = (time.time() - t0) * 1000
                                    parts.append(delta)
                                if data.get("done"):
                                    break
                    e2e_ms = (time.time() - t0) * 1000
                    answer = self._clean_model_answer("".join(parts))
                    if answer.strip():
                        return answer, e2e_ms, ttft_ms

                    write_debug("empty_stream_response_before_non_stream_fallback", raw_chunks)
                    fallback_answer, fallback_e2e_ms, _ = call_non_stream("empty_non_stream_response_after_stream_fallback")
                    if fallback_answer.strip():
                        self.force_non_stream_models.add(model)
                        logging.warning(
                            "%s returned empty streaming content; using non-stream mode for remaining calls",
                            model,
                        )
                    return fallback_answer, fallback_e2e_ms, None
                except httpx.TimeoutException:
                    write_debug(
                        "stream_timeout",
                        {
                            "timeout_s": REQUEST_TIMEOUT,
                            "chunks_seen": len(raw_chunks),
                            "last_chunks": raw_chunks[-5:],
                        },
                    )
                    return "", float(REQUEST_TIMEOUT * 1000), ttft_ms
                except httpx.TransportError as exc:
                    last_transport_error = exc
                    continue

            if last_transport_error is not None:
                raise last_transport_error

            raise RuntimeError("brak skonfigurowanych URL Ollamy")

        except httpx.TimeoutException:
            return "", float(REQUEST_TIMEOUT * 1000), ttft_ms
        except Exception as exc:
            logging.warning("Ollama error: %s", exc)
            return "", float(REQUEST_TIMEOUT * 1000), ttft_ms

    def _unload_ollama_model(self, model: str) -> None:
        payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        for ollama_url in self.ollama_urls:
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(f"{ollama_url}/api/generate", json=payload)
                    resp.raise_for_status()
                return
            except Exception as exc:
                logging.warning("Could not unload %s via %s: %s", model, ollama_url, exc)

    def _warm_ollama_model(self, model: str, num_ctx: int) -> None:
        load_payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": num_ctx, "num_predict": 1},
        }
        use_completion_api = self._uses_completion_api(model)
        if use_completion_api:
            prime_endpoint = "/api/generate"
            prime_payload = {
                "model": model,
                "prompt": "Odpowiedz jednym słowem: ok",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": num_ctx, "num_predict": 4, "temperature": 0.1},
            }
        else:
            prime_endpoint = "/api/chat"
            prime_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Odpowiedz jednym słowem: ok"}],
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": num_ctx, "num_predict": 4, "temperature": 0.1},
            }
        last_error: Exception | None = None
        for ollama_url in self.ollama_urls:
            try:
                with httpx.Client(timeout=MODEL_LOAD_TIMEOUT) as client:
                    resp = client.post(f"{ollama_url}/api/generate", json=load_payload)
                    resp.raise_for_status()
                    resp = client.post(f"{ollama_url}{prime_endpoint}", json=prime_payload)
                    resp.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                logging.warning("Could not warm %s via %s: %s", model, ollama_url, exc)
        if last_error is not None:
            raise RuntimeError(f"Could not warm {model}: {last_error}") from last_error

    def _run_trial(
        self,
        model: str,
        trial_num: int,
        trials_total_for_model: int,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        top_p: float,
        repeat_penalty: float,
        out_file,
    ) -> float:
        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        scores = []
        empty_answers = 0
        consecutive_empty_timeouts = 0
        pbar = tqdm(
            self.golden,
            desc=f"[{model}] Trial {trial_num}/{trials_total_for_model}",
            leave=False,
            ncols=80,
        )
        for case in pbar:
            timed_out = False
            answer, e2e_ms, ttft_ms = self._call_ollama(
                model, case["system_prompt"], case["question"], options
            )
            timeout_ms = float(REQUEST_TIMEOUT * 1000)
            if e2e_ms >= timeout_ms - 0.001:
                timed_out = True

            metrics = get_rpi_metrics(self.metrics_urls)

            if not answer.strip():
                empty_answers += 1

            if timed_out and not answer.strip():
                consecutive_empty_timeouts += 1
            else:
                consecutive_empty_timeouts = 0

            if timed_out or not answer.strip():
                eval_result = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "conciseness": 0.0,
                    "polish_quality": 0.0,
                    "hallucination_flag": True,
                    "haiku_raw": "",
                }
            else:
                eval_result = evaluate.evaluate_response(
                    system_prompt=case["system_prompt"],
                    question=case["question"],
                    golden_answer=case["golden_answer"],
                    model_answer=answer,
                    category=case["category"],
                )

            ttft_saved = None if ttft_ms is None else round(ttft_ms, 1)

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
                "model": model,
                "trial": trial_num,
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "golden_answer": case["golden_answer"],
                "model_answer": answer,
                "ttft_ms": ttft_saved,
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
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            scores.append(score)

            if consecutive_empty_timeouts >= MAX_CONSECUTIVE_EMPTY_TIMEOUTS:
                msg = (
                    f"Model {model} returned {consecutive_empty_timeouts} consecutive "
                    "empty timeouts; stopping this trial early"
                )
                logging.warning(msg)
                return 0.0

        if trial_num == 1 and empty_answers == len(self.golden):
            logging.warning("Model %s produced empty answers for the whole first trial", model)
            return 0.0

        return mean(scores) if scores else 0.0

    def run_model(self, model: str) -> None:
        if not check_model_available(self.ollama_urls, model):
            print(f"\n[SKIP] {model} — nie znaleziony na RPi")
            return

        n_for_model = self.model_trials.get(model, self.default_n_trials)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = _model_to_safe(model)
        out_path = self.results_dir / f"{safe}_{timestamp}.jsonl"

        print(f"\n{'='*60}")
        print(f"Model: {model}  |  Trials: {n_for_model}  |  Cases: {len(self.golden)}")
        print(f"num_ctx choices: {num_ctx_choices_for_model(model)}")
        print(f"Output: {out_path.name}")
        print(f"{'='*60}")

        study: optuna.Study | None = None

        with open(out_path, "w", encoding="utf-8") as out_file:
            def objective(trial: optuna.Trial) -> float:
                temperature    = trial.suggest_float("temperature", 0.1, 0.8)
                num_ctx        = trial.suggest_categorical("num_ctx", num_ctx_choices_for_model(model))
                num_predict    = trial.suggest_int("num_predict", 48, 160)
                top_p          = trial.suggest_float("top_p", 0.80, 0.99)
                repeat_penalty = trial.suggest_float("repeat_penalty", 1.0, 1.15)

                self._unload_ollama_model(model)
                self._warm_ollama_model(model, num_ctx)
                try:
                    return self._run_trial(
                        model=model,
                        trial_num=trial.number + 1,
                        trials_total_for_model=n_for_model,
                        temperature=temperature,
                        num_ctx=num_ctx,
                        num_predict=num_predict,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        out_file=out_file,
                    )
                finally:
                    self._unload_ollama_model(model)

            study = optuna.create_study(
                study_name=f"{_model_to_safe(model)}_{self.run_id}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=self.storage,
                load_if_exists=True,
            )
            study.optimize(objective, n_trials=n_for_model)

        assert study is not None
        best = study.best_trial
        print(f"\nBest trial #{best.number}: score={best.value:.4f}")
        print(f"  params: {best.params}")

    def run_all(self, models: list[str] | None = None) -> None:
        mode = "[DRY RUN] " if self.dry_run else ""
        targets = models if models else benchmark_model_list()
        trials_line = []
        for m in targets:
            trials_line.append(str(self.model_trials.get(m, self.default_n_trials)))
        print(f"\nWilga Benchmark {mode}— start {self.run_id}")
        print(f"Modele: {targets}")
        print(f"Trials per model: {trials_line}  (kolejność jak lista modeli)")
        print(f"Dashboard: optuna-dashboard {self.storage}  (odpal w osobnym terminalu)")
        for model in targets:
            self.run_model(model)
        report_path = Path(__file__).parent / "report.html"
        report.generate_report(str(self.results_dir), str(report_path))
        print(f"\nRaport: {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Wilga LLM Benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="2 trialy, 3 cases (wszystkie modele)")
    group.add_argument("--test", action="store_true", help="Sanity check RPi + modele")
    parser.add_argument(
        "--purge-models",
        nargs="+",
        metavar="MODEL",
        dest="purge_models_list",
        default=None,
        help="Usuń z benchmark/optuna.db wszystkie studia dla podanych modeli (wszystkie run_id); kończy program",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="MODEL1,MODEL2",
        help=(
            "Podzbiór modeli do uruchomienia (domyślnie: MODELS z kodu "
            "+ POLISH_OLLAMA_MODEL / EXTRA_OLLAMA_MODELS z ENV)"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help=f"Liczba triali Optuny dla modeli bez wpisu --model-trials (domyślnie: {N_TRIALS})",
    )
    parser.add_argument(
        "--model-trials",
        action="append",
        default=[],
        metavar="MODEL=N",
        help=(
            "N triali dla jednego modelu, powtarzalne, np. --model-trials gemma3:4b=30 "
            "--model-trials llama3.2:3b=10"
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="ID",
        help="Wspólny sufiks nazwy study Optuny (np. ten sam przy kilku wywołaniach na tej samej sesji eksperymentalnej)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    storage_uri = f"sqlite:///{base_dir / 'optuna.db'}"

    if args.purge_models_list is not None:
        purge_targets = args.purge_models_list
        deleted = purge_optuna_studies_for_models(storage_uri, purge_targets)
        if not deleted:
            print(f"Brak studiów do usunięcia dla prefiksów modeli {purge_targets}")
        else:
            print(f"Usunięto {len(deleted)} studiów:")
            for n in deleted:
                print(f"  - {n}")
        sys.exit(0)

    models_list = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else None
    )

    if args.test:
        sys.exit(run_test(models=models_list))

    model_tri_map = parse_model_trials(list(args.model_trials)) if args.model_trials else {}

    runner = BenchmarkRunner(
        dry_run=args.dry_run,
        n_trials=args.trials,
        run_id=args.run_id,
        model_trials=model_tri_map,
    )
    runner.run_all(models=models_list)


if __name__ == "__main__":
    main()
