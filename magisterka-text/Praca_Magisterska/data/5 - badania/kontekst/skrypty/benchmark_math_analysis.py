#!/usr/bin/env python3
"""Mathematical analysis of benchmark JSONL results.

The script intentionally uses only the Python standard library so it can be
run in the thesis workspace without installing pandas, numpy, matplotlib, etc.
It produces CSV tables, a Markdown report, and static SVG plots.
"""

from __future__ import annotations

import csv
import glob
import html
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis-output"

HPARAMS = ["temperature", "num_ctx", "num_predict", "top_p", "repeat_penalty"]
METRICS = ["composite_score", "faithfulness", "answer_relevancy", "conciseness", "polish_quality"]

MODEL_COLORS = {
    "gemma2:2b": "#1f77b4",
    "gemma3:4b": "#2ca02c",
    "llama3.2:3b": "#ff7f0e",
    "qwen2.5:3b": "#d62728",
    "phi3.5": "#9467bd",
    "batiai/gemma4-e2b:q4": "#17becf",
    "gemma4:e2b": "#7f7f7f",
}


def read_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(ROOT / "results-*/*.jsonl"))):
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_dataset"] = p.parent.name
                row["_file"] = p.name
                row["_path"] = str(p.relative_to(ROOT))
                row["_line"] = line_no
                rows.append(row)
    return rows


def mean(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[float | int | None]) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None)
    n = len(vals)
    if not n:
        return None
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def pctl(xs: Iterable[float | int | None], p: float) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    idx = round((len(vals) - 1) * p)
    return vals[max(0, min(len(vals) - 1, idx))]


def sd(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    if len(vals) > 1:
        return statistics.stdev(vals)
    if len(vals) == 1:
        return 0.0
    return None


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def pearson(xs: Iterable[float | int | None], ys: Iterable[float | int | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def rank(vals: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    j = 0
    while j < len(indexed):
        k = j + 1
        while k < len(indexed) and indexed[k][0] == indexed[j][0]:
            k += 1
        avg_rank = (j + k - 1) / 2 + 1
        for _, i in indexed[j:k]:
            out[i] = avg_rank
        j = k
    return out


def spearman(xs: Iterable[float | int | None], ys: Iterable[float | int | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    if len(set(xvals)) < 2 or len(set(yvals)) < 2:
        return None
    return pearson(rank(xvals), rank(yvals))


def simple_regression(xs: Iterable[float | int | None], ys: Iterable[float | int | None]) -> dict[str, float | None]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "intercept": None, "slope": None, "r2": None}
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    sxx = sum((x - mx) ** 2 for x in xvals)
    if sxx == 0:
        return {"n": len(pairs), "intercept": None, "slope": None, "r2": None}
    slope = sum((x - mx) * (y - my) for x, y in pairs) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in yvals)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pairs)
    r2 = None if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"n": len(pairs), "intercept": intercept, "slope": slope, "r2": r2}


def fmt(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


@dataclass
class TrialSummary:
    dataset: str
    file: str
    model: str
    trial: int
    n_cases: int
    temperature: float | None
    num_ctx: float | None
    num_predict: float | None
    top_p: float | None
    repeat_penalty: float | None
    score_mean: float | None
    hallucination_rate: float | None
    nonhall_score_mean: float | None
    nonhall_n: int
    e2e_median_ms: float | None
    e2e_p90_ms: float | None
    ttft_median_ms: float | None
    empty_rate: float | None

    def asdict(self) -> dict:
        return self.__dict__.copy()


def build_trial_summaries(rows: list[dict]) -> list[TrialSummary]:
    groups: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["_dataset"], row["_file"], row["model"], int(row["trial"]))].append(row)

    summaries: list[TrialSummary] = []
    for (dataset, file_name, model, trial), rs in sorted(groups.items()):
        first = rs[0]
        nonhall = [r for r in rs if not r.get("hallucination_flag")]
        summaries.append(
            TrialSummary(
                dataset=dataset,
                file=file_name,
                model=model,
                trial=trial,
                n_cases=len(rs),
                temperature=first.get("temperature"),
                num_ctx=first.get("num_ctx"),
                num_predict=first.get("num_predict"),
                top_p=first.get("top_p"),
                repeat_penalty=first.get("repeat_penalty"),
                score_mean=mean(r.get("composite_score") for r in rs),
                hallucination_rate=mean(1.0 if r.get("hallucination_flag") else 0.0 for r in rs),
                nonhall_score_mean=mean(r.get("composite_score") for r in nonhall),
                nonhall_n=len(nonhall),
                e2e_median_ms=median(r.get("e2e_ms") for r in rs),
                e2e_p90_ms=pctl((r.get("e2e_ms") for r in rs), 0.90),
                ttft_median_ms=median(r.get("ttft_ms") for r in rs),
                empty_rate=mean(1.0 if not (r.get("model_answer") or "").strip() else 0.0 for r in rs),
            )
        )
    return summaries


def model_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["_dataset"], row["model"])].append(row)

    out: list[dict] = []
    for (dataset, model), rs in sorted(groups.items()):
        nonhall = [r for r in rs if not r.get("hallucination_flag")]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "rows": len(rs),
                "trials": len({(r["_file"], r["trial"]) for r in rs}),
                "cases": len({r["case_id"] for r in rs}),
                "score_mean": mean(r.get("composite_score") for r in rs),
                "score_sd": sd(r.get("composite_score") for r in rs),
                "hallucination_rate": mean(1.0 if r.get("hallucination_flag") else 0.0 for r in rs),
                "nonhall_score_mean": mean(r.get("composite_score") for r in nonhall),
                "nonhall_score_sd": sd(r.get("composite_score") for r in nonhall),
                "e2e_median_ms": median(r.get("e2e_ms") for r in rs),
                "e2e_p90_ms": pctl((r.get("e2e_ms") for r in rs), 0.90),
                "ttft_median_ms": median(r.get("ttft_ms") for r in rs),
                "empty_rate": mean(1.0 if not (r.get("model_answer") or "").strip() else 0.0 for r in rs),
                "throttling_rate": mean(1.0 if r.get("throttling") else 0.0 for r in rs),
            }
        )
    return out


def decomposition_table(trials: list[TrialSummary]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[TrialSummary]] = defaultdict(list)
    for t in trials:
        groups[(t.dataset, t.file, t.model)].append(t)

    out: list[dict] = []
    all_reg = simple_regression(
        [t.hallucination_rate for t in trials],
        [t.score_mean for t in trials],
    )
    out.append(
        {
            "dataset": "__ALL__",
            "file": "__ALL__",
            "model": "__ALL__",
            "n_trials": len(trials),
            "score_mean": mean(t.score_mean for t in trials),
            "score_sd_between_trials": sd(t.score_mean for t in trials),
            "hallucination_rate_mean": mean(t.hallucination_rate for t in trials),
            "hallucination_rate_sd": sd(t.hallucination_rate for t in trials),
            "nonhall_score_mean": mean(t.nonhall_score_mean for t in trials),
            "nonhall_score_sd_between_trials": sd(t.nonhall_score_mean for t in trials),
            "corr_score_hallucination": pearson(
                [t.score_mean for t in trials],
                [t.hallucination_rate for t in trials],
            ),
            "r2_score_from_hallucination_rate": all_reg["r2"],
            "slope_score_from_hallucination_rate": all_reg["slope"],
            "nonhall_sd_to_score_sd_ratio": safe_div(
                sd(t.nonhall_score_mean for t in trials),
                sd(t.score_mean for t in trials),
            ),
        }
    )

    for (dataset, file_name, model), ts in sorted(groups.items()):
        reg = simple_regression(
            [t.hallucination_rate for t in ts],
            [t.score_mean for t in ts],
        )
        out.append(
            {
                "dataset": dataset,
                "file": file_name,
                "model": model,
                "n_trials": len(ts),
                "score_mean": mean(t.score_mean for t in ts),
                "score_sd_between_trials": sd(t.score_mean for t in ts),
                "hallucination_rate_mean": mean(t.hallucination_rate for t in ts),
                "hallucination_rate_sd": sd(t.hallucination_rate for t in ts),
                "nonhall_score_mean": mean(t.nonhall_score_mean for t in ts),
                "nonhall_score_sd_between_trials": sd(t.nonhall_score_mean for t in ts),
                "corr_score_hallucination": pearson(
                    [t.score_mean for t in ts],
                    [t.hallucination_rate for t in ts],
                ),
                "r2_score_from_hallucination_rate": reg["r2"],
                "slope_score_from_hallucination_rate": reg["slope"],
                "nonhall_sd_to_score_sd_ratio": safe_div(
                    sd(t.nonhall_score_mean for t in ts),
                    sd(t.score_mean for t in ts),
                ),
            }
        )
    return out


def hparam_effects(trials: list[TrialSummary]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[TrialSummary]] = defaultdict(list)
    for t in trials:
        groups[(t.dataset, t.file, t.model)].append(t)

    out: list[dict] = []
    targets = [
        ("score_mean", lambda t: t.score_mean),
        ("hallucination_rate", lambda t: t.hallucination_rate),
        ("nonhall_score_mean", lambda t: t.nonhall_score_mean),
        ("e2e_median_ms", lambda t: t.e2e_median_ms),
    ]
    for (dataset, file_name, model), ts in sorted(groups.items()):
        for target_name, get_y in targets:
            y = [get_y(t) for t in ts]
            for hp in HPARAMS:
                x = [getattr(t, hp) for t in ts]
                out.append(
                    {
                        "dataset": dataset,
                        "file": file_name,
                        "model": model,
                        "n_trials": len(ts),
                        "target": target_name,
                        "hparam": hp,
                        "pearson_r": pearson(x, y),
                        "spearman_r": spearman(x, y),
                    }
                )
    return out


def case_difficulty(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["_dataset"], row["case_id"], row["category"])].append(row)
    out: list[dict] = []
    for (dataset, case_id, category), rs in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "category": category,
                "n": len(rs),
                "score_mean": mean(r.get("composite_score") for r in rs),
                "hallucination_rate": mean(1.0 if r.get("hallucination_flag") else 0.0 for r in rs),
                "e2e_median_ms": median(r.get("e2e_ms") for r in rs),
                "empty_rate": mean(1.0 if not (r.get("model_answer") or "").strip() else 0.0 for r in rs),
            }
        )
    return out


def category_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["_dataset"], row["category"])].append(row)
    out: list[dict] = []
    for (dataset, category), rs in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "category": category,
                "n": len(rs),
                "score_mean": mean(r.get("composite_score") for r in rs),
                "hallucination_rate": mean(1.0 if r.get("hallucination_flag") else 0.0 for r in rs),
                "e2e_median_ms": median(r.get("e2e_ms") for r in rs),
                "empty_rate": mean(1.0 if not (r.get("model_answer") or "").strip() else 0.0 for r in rs),
            }
        )
    return out


def best_worst_trials(trials: list[TrialSummary]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[TrialSummary]] = defaultdict(list)
    for t in trials:
        groups[(t.dataset, t.file, t.model)].append(t)

    out: list[dict] = []
    for (dataset, file_name, model), ts in sorted(groups.items()):
        if len(ts) < 2:
            continue
        best = max(ts, key=lambda t: t.score_mean if t.score_mean is not None else -1)
        worst = min(ts, key=lambda t: t.score_mean if t.score_mean is not None else 999)
        out.append(
            {
                "dataset": dataset,
                "file": file_name,
                "model": model,
                "n_trials": len(ts),
                "best_trial": best.trial,
                "best_score": best.score_mean,
                "best_hallucination_rate": best.hallucination_rate,
                "best_nonhall_score": best.nonhall_score_mean,
                "worst_trial": worst.trial,
                "worst_score": worst.score_mean,
                "worst_hallucination_rate": worst.hallucination_rate,
                "worst_nonhall_score": worst.nonhall_score_mean,
                "score_gap": (best.score_mean or 0) - (worst.score_mean or 0),
                "nonhall_score_gap": None
                if best.nonhall_score_mean is None or worst.nonhall_score_mean is None
                else best.nonhall_score_mean - worst.nonhall_score_mean,
                "hallucination_rate_gap": None
                if best.hallucination_rate is None or worst.hallucination_rate is None
                else worst.hallucination_rate - best.hallucination_rate,
            }
        )
    return out


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202124}",
        ".axis{stroke:#5f6368;stroke-width:1}",
        ".grid{stroke:#e0e0e0;stroke-width:1}",
        ".label{font-size:12px}",
        ".title{font-size:18px;font-weight:700}",
        ".small{font-size:11px;fill:#5f6368}",
        "</style>",
    ]


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def draw_axes(
    parts: list[str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: int = 5,
    y_ticks: int = 5,
) -> None:
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}"/>')
    for i in range(x_ticks + 1):
        v = x_min + (x_max - x_min) * i / x_ticks
        x = scale(v, x_min, x_max, x0, x0 + w)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + h}"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{y0 + h + 18}" text-anchor="middle">{v:.2f}</text>')
    for i in range(y_ticks + 1):
        v = y_min + (y_max - y_min) * i / y_ticks
        y = scale(v, y_min, y_max, y0 + h, y0)
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    parts.append(f'<text class="label" x="{x0 + w / 2}" y="{y0 + h + 45}" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append(
        f'<text class="label" transform="translate({x0 - 50},{y0 + h / 2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )


def plot_score_vs_hallucination(trials: list[TrialSummary], path: Path, dataset_filter: str | None = None) -> None:
    points = [t for t in trials if t.score_mean is not None and t.hallucination_rate is not None]
    if dataset_filter:
        points = [t for t in points if t.dataset == dataset_filter]
    if not points:
        return

    xs = [t.hallucination_rate for t in points if t.hallucination_rate is not None]
    ys = [t.score_mean for t in points if t.score_mean is not None]
    x_min, x_max = 0.0, max(0.65, max(xs) * 1.05)
    y_min, y_max = 0.0, min(1.0, max(ys) * 1.08)
    reg = simple_regression(xs, ys)

    width, height = 860, 560
    x0, y0, w, h = 85, 70, 650, 390
    parts = svg_header(width, height)
    title = "Composite score vs hallucination rate"
    if dataset_filter:
        title += f" ({dataset_filter})"
    parts.append(f'<text class="title" x="35" y="34">{html.escape(title)}</text>')
    draw_axes(parts, x0, y0, w, h, "hallucination rate per trial", "mean composite score per trial", x_min, x_max, y_min, y_max)

    if reg["slope"] is not None and reg["intercept"] is not None:
        x_a, x_b = x_min, x_max
        y_a = reg["intercept"] + reg["slope"] * x_a
        y_b = reg["intercept"] + reg["slope"] * x_b
        parts.append(
            f'<line x1="{scale(x_a, x_min, x_max, x0, x0 + w):.1f}" y1="{scale(y_a, y_min, y_max, y0 + h, y0):.1f}" '
            f'x2="{scale(x_b, x_min, x_max, x0, x0 + w):.1f}" y2="{scale(y_b, y_min, y_max, y0 + h, y0):.1f}" '
            'stroke="#111" stroke-width="2" stroke-dasharray="6 4"/>'
        )
        parts.append(f'<text class="small" x="{x0 + 12}" y="{y0 + 20}">R²={reg["r2"]:.3f}, slope={reg["slope"]:.3f}</text>')

    for t in points:
        x = scale(t.hallucination_rate or 0, x_min, x_max, x0, x0 + w)
        y = scale(t.score_mean or 0, y_min, y_max, y0 + h, y0)
        color = MODEL_COLORS.get(t.model, "#333333")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.82">'
            f'<title>{html.escape(t.dataset)} | {html.escape(t.model)} | trial {t.trial}</title></circle>'
        )

    legend_x = 755
    legend_y = 85
    for i, model in enumerate(sorted({t.model for t in points})):
        y = legend_y + i * 22
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{MODEL_COLORS.get(model, "#333")}"/>')
        parts.append(f'<text class="small" x="{legend_x + 12}" y="{y + 4}">{html.escape(model)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_latency_pareto(model_rows: list[dict], path: Path, dataset: str = "results-selected-9.05") -> None:
    rows = [r for r in model_rows if r["dataset"] == dataset and r.get("e2e_median_ms") is not None]
    if not rows:
        return
    x_vals = [float(r["e2e_median_ms"]) / 1000 for r in rows]
    y_vals = [float(r["hallucination_rate"]) for r in rows]
    x_min, x_max = min(x_vals) * 0.90, max(x_vals) * 1.10
    y_min, y_max = 0.0, max(0.25, max(y_vals) * 1.25)

    width, height = 800, 520
    x0, y0, w, h = 85, 70, 560, 350
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="35" y="34">Latency vs hallucination rate ({html.escape(dataset)})</text>')
    draw_axes(parts, x0, y0, w, h, "median E2E latency [s]", "hallucination rate", x_min, x_max, y_min, y_max)

    for r in rows:
        x = scale(float(r["e2e_median_ms"]) / 1000, x_min, x_max, x0, x0 + w)
        y = scale(float(r["hallucination_rate"]), y_min, y_max, y0 + h, y0)
        color = MODEL_COLORS.get(r["model"], "#333333")
        radius = 7 + 11 * float(r.get("score_mean") or 0)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.75"/>')
        parts.append(f'<text class="small" x="{x + 12:.1f}" y="{y + 4:.1f}">{html.escape(r["model"])}</text>')
    parts.append('<text class="small" x="35" y="485">Lower-left is better. Point radius is proportional to mean composite score.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_nonhall_spread(trials: list[TrialSummary], path: Path, dataset: str = "results-selected-9.05") -> None:
    points = [t for t in trials if t.dataset == dataset and t.nonhall_score_mean is not None]
    if not points:
        return
    models = sorted({t.model for t in points})
    y_vals = [t.nonhall_score_mean for t in points if t.nonhall_score_mean is not None]
    y_min, y_max = max(0.75, min(y_vals) - 0.02), min(0.95, max(y_vals) + 0.02)

    width, height = 860, 520
    x0, y0, w, h = 80, 65, 670, 350
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="35" y="34">Conditional score after removing hallucinated answers ({html.escape(dataset)})</text>')
    draw_axes(parts, x0, y0, w, h, "model", "mean score | hallucination_flag=false", 0, len(models) - 1, y_min, y_max, x_ticks=max(1, len(models) - 1), y_ticks=5)

    for i, model in enumerate(models):
        model_points = [t for t in points if t.model == model]
        x_base = scale(i, 0, len(models) - 1, x0, x0 + w)
        vals = [t.nonhall_score_mean for t in model_points if t.nonhall_score_mean is not None]
        m = mean(vals)
        s = sd(vals)
        if m is not None:
            y_m = scale(m, y_min, y_max, y0 + h, y0)
            parts.append(f'<line x1="{x_base - 22:.1f}" y1="{y_m:.1f}" x2="{x_base + 22:.1f}" y2="{y_m:.1f}" stroke="#111" stroke-width="2"/>')
        if m is not None and s is not None:
            y1 = scale(m - s, y_min, y_max, y0 + h, y0)
            y2 = scale(m + s, y_min, y_max, y0 + h, y0)
            parts.append(f'<line x1="{x_base:.1f}" y1="{y1:.1f}" x2="{x_base:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="1"/>')
        for j, t in enumerate(model_points):
            jitter = ((j % 7) - 3) * 5
            x = x_base + jitter
            y = scale(t.nonhall_score_mean or 0, y_min, y_max, y0 + h, y0)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{MODEL_COLORS.get(model, "#333")}" opacity="0.72"/>')
        parts.append(
            f'<text class="small" transform="translate({x_base:.1f},{y0 + h + 32}) rotate(-22)" text-anchor="end">{html.escape(model)}</text>'
        )
    parts.append('<text class="small" x="35" y="485">Horizontal marker: mean per model. Vertical whisker: +/- 1 SD across trials.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_case_hallucinations(case_rows: list[dict], path: Path, dataset: str = "results-selected-9.05", top_n: int = 12) -> None:
    rows = [r for r in case_rows if r["dataset"] == dataset]
    rows = sorted(rows, key=lambda r: float(r["hallucination_rate"] or 0), reverse=True)[:top_n]
    if not rows:
        return
    width, height = 860, 520
    x0, y0, w, bar_h = 190, 70, 560, 26
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="35" y="34">Most hallucination-prone cases ({html.escape(dataset)})</text>')
    max_v = max(float(r["hallucination_rate"]) for r in rows) or 1.0
    for i, r in enumerate(rows):
        y = y0 + i * (bar_h + 8)
        val = float(r["hallucination_rate"])
        bw = scale(val, 0, max_v, 0, w)
        parts.append(f'<text class="small" x="{x0 - 10}" y="{y + 18}" text-anchor="end">{html.escape(r["case_id"])}</text>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="#d62728" opacity="0.78"/>')
        parts.append(f'<text class="small" x="{x0 + bw + 8:.1f}" y="{y + 18}">{val:.1%}</text>')
    parts.append('<text class="small" x="35" y="485">Bars show the share of rows with hallucination_flag=true.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_hparam_heatmap(effects: list[dict], path: Path, dataset: str = "results-selected-9.05") -> None:
    rows = [
        r
        for r in effects
        if r["dataset"] == dataset
        and r["target"] in {"score_mean", "hallucination_rate", "nonhall_score_mean", "e2e_median_ms"}
        and r.get("pearson_r") is not None
    ]
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    targets = ["score_mean", "hallucination_rate", "nonhall_score_mean", "e2e_median_ms"]
    cell_w, cell_h = 78, 35
    left, top = 210, 90
    width = left + len(HPARAMS) * cell_w + 260
    height = top + len(models) * len(targets) * cell_h + 90
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="35" y="34">Univariate Pearson correlations: hyperparameters vs trial metrics ({html.escape(dataset)})</text>')
    parts.append('<text class="small" x="35" y="56">Red: positive; blue: negative; color intensity is |r|. Correlations are descriptive, not causal.</text>')

    lookup = {(r["model"], r["target"], r["hparam"]): r["pearson_r"] for r in rows}
    for j, hp in enumerate(HPARAMS):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text class="small" x="{x:.1f}" y="{top - 12}" text-anchor="middle">{html.escape(hp)}</text>')

    row_idx = 0
    for model in models:
        parts.append(f'<text class="small" x="35" y="{top + row_idx * cell_h + 23}">{html.escape(model)}</text>')
        for target in targets:
            y = top + row_idx * cell_h
            parts.append(f'<text class="small" x="150" y="{y + 23}">{html.escape(target)}</text>')
            for j, hp in enumerate(HPARAMS):
                val = lookup.get((model, target, hp))
                x = left + j * cell_w
                if val is None:
                    fill = "#f1f3f4"
                    text = ""
                else:
                    intensity = int(235 - min(1.0, abs(float(val))) * 155)
                    if val >= 0:
                        fill = f"rgb(220,{intensity},{intensity})"
                    else:
                        fill = f"rgb({intensity},{intensity},220)"
                    text = f"{val:+.2f}"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" fill="{fill}" stroke="#fff"/>')
                parts.append(f'<text class="small" x="{x + cell_w / 2:.1f}" y="{y + 22}" text-anchor="middle">{text}</text>')
            row_idx += 1
        row_idx += 1
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows: list[dict], columns: list[str], digits: int = 4, limit: int | None = None) -> str:
    shown = rows[:limit] if limit is not None else rows
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(fmt(val, digits))
            else:
                vals.append(str(val) if val is not None else "")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report(
    rows: list[dict],
    model_rows: list[dict],
    decomp_rows: list[dict],
    case_rows: list[dict],
    category_rows: list[dict],
    best_worst_rows: list[dict],
) -> None:
    all_decomp = next(r for r in decomp_rows if r["dataset"] == "__ALL__")
    selected_models = [r for r in model_rows if r["dataset"] == "results-selected-9.05"]
    selected_cases = sorted(
        [r for r in case_rows if r["dataset"] == "results-selected-9.05"],
        key=lambda r: float(r["hallucination_rate"] or 0),
        reverse=True,
    )
    selected_best_worst = [r for r in best_worst_rows if r["dataset"] == "results-selected-9.05"]

    lines = [
        "# Analiza matematyczna wynikow benchmarkow",
        "",
        "## Cel",
        "",
        (
            "Celem analizy jest sprawdzenie, czy zmiany hiperparametrow dobieranych przez Optune "
            "istotnie wplywaja na jakosc modelu, czy tez obserwowane roznice wynikaja przede wszystkim "
            "z dwoch czynnikow: opoznienia odpowiedzi oraz binarnej flagi halucynacji."
        ),
        "",
        "## Dekompozycja wyniku",
        "",
        (
            "Dla triala t sredni wynik mozna zapisac jako: "
            "`mean_score_t = (1 - hallucination_rate_t) * mean_score_t_given_no_hallucination`, "
            "poniewaz w danych kazdy rekord z `hallucination_flag=true` ma `composite_score=0`."
        ),
        "",
        (
            f"W calym zbiorze regresja liniowa `mean_score_t ~ hallucination_rate_t` daje "
            f"`R^2={fmt(all_decomp['r2_score_from_hallucination_rate'], 4)}` oraz korelacje "
            f"`r={fmt(all_decomp['corr_score_hallucination'], 4)}`. Oznacza to, ze sama czestosc "
            "halucynacji tlumaczy niemal cala zmiennosc sredniego wyniku miedzy trialami."
        ),
        "",
        "## Modele w zestawie selected",
        "",
        markdown_table(
            selected_models,
            [
                "model",
                "rows",
                "trials",
                "score_mean",
                "hallucination_rate",
                "nonhall_score_mean",
                "nonhall_score_sd",
                "e2e_median_ms",
                "e2e_p90_ms",
            ],
            digits=4,
        ),
        "",
        "## Najtrudniejsze przypadki w zestawie selected",
        "",
        markdown_table(
            selected_cases,
            ["case_id", "category", "n", "score_mean", "hallucination_rate", "e2e_median_ms"],
            digits=4,
            limit=12,
        ),
        "",
        "## Najlepszy i najgorszy trial",
        "",
        (
            "Tabela porownuje najlepszy i najgorszy trial wedlug `composite_score`. "
            "Jesli `nonhall_score_gap` jest maly, a `hallucination_rate_gap` duzy, to roznica "
            "wyniku pochodzi glownie z liczby odpowiedzi oznaczonych jako halucynacje."
        ),
        "",
        markdown_table(
            selected_best_worst,
            [
                "model",
                "best_score",
                "best_hallucination_rate",
                "best_nonhall_score",
                "worst_score",
                "worst_hallucination_rate",
                "worst_nonhall_score",
                "score_gap",
                "nonhall_score_gap",
                "hallucination_rate_gap",
            ],
            digits=4,
        ),
        "",
        "## Wnioski robocze",
        "",
        "1. `composite_score` jest zdominowany przez `hallucination_flag`, a nie przez subtelne roznice ocen czastkowych.",
        "2. Po usunieciu odpowiedzi z flaga halucynacji sredni wynik warunkowy jest stabilny miedzy trialami.",
        "3. Dalsze testowanie modeli na tych samych benchmarkach ma ograniczona wartosc, jesli celem jest strojenie hiperparametrow jakosciowych.",
        "4. Sensowniejszym kryterium decyzyjnym jest Pareto: minimalizowac latency przy akceptowalnym poziomie halucynacji.",
        "5. Najwieksza wartosc diagnostyczna benchmarku lezy w kategoriach `missing_data` i `stale_data`, bo tam koncentruja sie flagi halucynacji.",
        "",
        "## Wygenerowane wykresy",
        "",
        "- `fig_score_vs_hallucination_all.svg`",
        "- `fig_score_vs_hallucination_selected.svg`",
        "- `fig_latency_vs_hallucination_selected.svg`",
        "- `fig_nonhall_score_spread_selected.svg`",
        "- `fig_case_hallucinations_selected.svg`",
        "- `fig_hparam_correlations_selected.svg`",
        "",
        "## Uwagi metodologiczne",
        "",
        (
            "Korelacje hiperparametrow nalezy traktowac opisowo, nie przyczynowo. W wielu plikach jest tylko "
            "5 triali, co nie wystarcza do stabilnej regresji wielowymiarowej. Dlatego glowny argument opiera sie "
            "na dekompozycji wyniku oraz porownaniu wariancji `mean_score` i `mean_score_given_no_hallucination`."
        ),
    ]
    (OUT / "thesis_math_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = read_rows()
    trials = build_trial_summaries(rows)
    model_rows = model_summary(rows)
    decomp_rows = decomposition_table(trials)
    effect_rows = hparam_effects(trials)
    case_rows = case_difficulty(rows)
    category_rows = category_summary(rows)
    best_worst_rows = best_worst_trials(trials)

    write_csv(
        OUT / "trial_summary.csv",
        [t.asdict() for t in trials],
        [
            "dataset",
            "file",
            "model",
            "trial",
            "n_cases",
            *HPARAMS,
            "score_mean",
            "hallucination_rate",
            "nonhall_score_mean",
            "nonhall_n",
            "e2e_median_ms",
            "e2e_p90_ms",
            "ttft_median_ms",
            "empty_rate",
        ],
    )
    write_csv(
        OUT / "model_summary.csv",
        model_rows,
        [
            "dataset",
            "model",
            "rows",
            "trials",
            "cases",
            "score_mean",
            "score_sd",
            "hallucination_rate",
            "nonhall_score_mean",
            "nonhall_score_sd",
            "e2e_median_ms",
            "e2e_p90_ms",
            "ttft_median_ms",
            "empty_rate",
            "throttling_rate",
        ],
    )
    write_csv(
        OUT / "score_decomposition.csv",
        decomp_rows,
        [
            "dataset",
            "file",
            "model",
            "n_trials",
            "score_mean",
            "score_sd_between_trials",
            "hallucination_rate_mean",
            "hallucination_rate_sd",
            "nonhall_score_mean",
            "nonhall_score_sd_between_trials",
            "corr_score_hallucination",
            "r2_score_from_hallucination_rate",
            "slope_score_from_hallucination_rate",
            "nonhall_sd_to_score_sd_ratio",
        ],
    )
    write_csv(
        OUT / "hparam_univariate_effects.csv",
        effect_rows,
        ["dataset", "file", "model", "n_trials", "target", "hparam", "pearson_r", "spearman_r"],
    )
    write_csv(
        OUT / "case_difficulty.csv",
        case_rows,
        ["dataset", "case_id", "category", "n", "score_mean", "hallucination_rate", "e2e_median_ms", "empty_rate"],
    )
    write_csv(
        OUT / "category_summary.csv",
        category_rows,
        ["dataset", "category", "n", "score_mean", "hallucination_rate", "e2e_median_ms", "empty_rate"],
    )
    write_csv(
        OUT / "best_worst_trials.csv",
        best_worst_rows,
        [
            "dataset",
            "file",
            "model",
            "n_trials",
            "best_trial",
            "best_score",
            "best_hallucination_rate",
            "best_nonhall_score",
            "worst_trial",
            "worst_score",
            "worst_hallucination_rate",
            "worst_nonhall_score",
            "score_gap",
            "nonhall_score_gap",
            "hallucination_rate_gap",
        ],
    )

    plot_score_vs_hallucination(trials, OUT / "fig_score_vs_hallucination_all.svg")
    plot_score_vs_hallucination(trials, OUT / "fig_score_vs_hallucination_selected.svg", "results-selected-9.05")
    plot_latency_pareto(model_rows, OUT / "fig_latency_vs_hallucination_selected.svg")
    plot_nonhall_spread(trials, OUT / "fig_nonhall_score_spread_selected.svg")
    plot_case_hallucinations(case_rows, OUT / "fig_case_hallucinations_selected.svg")
    plot_hparam_heatmap(effect_rows, OUT / "fig_hparam_correlations_selected.svg")

    generate_report(rows, model_rows, decomp_rows, case_rows, category_rows, best_worst_rows)

    all_decomp = next(r for r in decomp_rows if r["dataset"] == "__ALL__")
    print(f"Rows: {len(rows)}")
    print(f"Trials: {len(trials)}")
    print(f"Output directory: {OUT.relative_to(ROOT)}")
    print(f"R2(mean_score ~ hallucination_rate): {all_decomp['r2_score_from_hallucination_rate']:.4f}")
    print(f"corr(mean_score, hallucination_rate): {all_decomp['corr_score_hallucination']:.4f}")


if __name__ == "__main__":
    main()
