#!/usr/bin/env python3
"""Generate Ch.5 review figures: LN for warm/cold quant, Gauss for Optuna E2E.

Writes to benchmark/analysis/review-figures-ch5/ — does NOT touch thesis TeX.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ln_stats import fit_lognormal, ln_group_summary

OUT = SCRIPT_ROOT / "review-figures-ch5"
WARM_OUT = OUT / "warm"
COLD_OUT = OUT / "cold"
OPTUNA_OUT = OUT / "optuna"
FIT_OUT = OUT / "fit"
TABLE_OUT = OUT / "tables"

WARM_SRC = SCRIPT_ROOT.parent / "results" / "quant_latency_warm_n100.jsonl"
COLD_SRC = SCRIPT_ROOT.parent / "results" / "quant_latency_cold_n100.jsonl"

MODEL_ORDER = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
VARIANT_ORDER = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#262626",
    "muted": "#555555",
    "grid": "#CCCCCC",
    "axis": "#CCCCCC",
}

NEUTRAL = {"dark": "#464C55"}

COLORS = {
    "Q2_K": {"base": "#ff7f0e", "dark": "#8c4a0a"},
    "Q4_0": {"base": "#1f77b4", "dark": "#174f78"},
    "Q4_K_M": {"base": "#2ca02c", "dark": "#1b651b"},
    "Q8_0": {"base": "#9467bd", "dark": "#5f3f7f"},
}

MODEL_COLORS = {
    "gemma3-4b": {"base": "#2ca02c", "dark": "#1b651b"},
    "gemma4-e2b": {"base": "#1f77b4", "dark": "#174f78"},
    "bielik-4.5b": {"base": "#d62728", "dark": "#8f1c1d"},
}

LN_NOTE = "Model LN: wysokość słupka = mediana e^μ; przedział = e^(μ±σ)"

MODEL_PALETTE = {
    "llama3.2:3b": "#1f77b4",
    "gemma2:2b": "#d62728",
    "gemma3:4b": "#2ca02c",
    "qwen2.5:3b": "#9467bd",
    "phi3.5": "#ff7f0e",
    "batiai/gemma4-e2b:q4": "#8c564b",
    "gemma4:e2b": "#e377c2",
}

TARGET_LABELS = {
    "score_mean": "sredni wynik laczny",
    "nonhall_score_mean": "sredni wynik bez halucynacji",
    "hallucination_rate": "odsetek halucynacji",
    "e2e_median_ms": "mediana opoznienia E2E",
    "e2e_mean_ms": "srednie opoznienie E2E",
}

REGRESSION_LABELS = {
    "hallucination_only": "tylko halucynacje",
    "hparams_only": "tylko hiperparametry",
    "model_only": "tylko model",
    "model_plus_hparams": "model + hiperparametry",
    "model_plus_hallucination": "model + halucynacje",
    "full": "pelny model",
}


def use_chart_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
        }
    )


def save_fig(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_quant_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("gguf_size_gb", "ttft_ms", "e2e_ms", "generation_ms", "answer_chars", "question_idx"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "cache_hit" in df.columns:
        df["cache_hit"] = df["cache_hit"].fillna(False).astype(bool)
    else:
        df["cache_hit"] = False
    df["empty_answer"] = df["answer_chars"].fillna(0).eq(0)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["quant_variant"] = pd.Categorical(df["quant_variant"], categories=VARIANT_ORDER, ordered=True)
    df["generation_s"] = df["generation_ms"] / 1000.0
    df["ttft_s"] = df["ttft_ms"] / 1000.0
    df["e2e_s"] = df["e2e_ms"] / 1000.0
    df["chars_per_s"] = np.where(
        (df["generation_ms"] > 0) & (df["answer_chars"] > 0),
        df["answer_chars"] / (df["generation_ms"] / 1000.0),
        np.nan,
    )
    return df.sort_values(["model", "quant_variant", "question_idx"]).reset_index(drop=True)


def build_ln_summary(df: pd.DataFrame) -> pd.DataFrame:
    gen = ln_group_summary(df, ["model", "quant_variant"], "generation_ms", "gen")
    ttft = ln_group_summary(df, ["model", "quant_variant"], "ttft_ms", "ttft")
    e2e = ln_group_summary(df, ["model", "quant_variant"], "e2e_ms", "e2e")
    cps = ln_group_summary(df, ["model", "quant_variant"], "chars_per_s", "cps")
    size = (
        df.groupby(["model", "quant_variant"], observed=True)
        .agg(gguf_size_gb=("gguf_size_gb", "median"), n_rows=("case_id", "count"))
        .reset_index()
    )
    out = size.merge(gen, on=["model", "quant_variant"]).merge(ttft, on=["model", "quant_variant"])
    out = out.merge(e2e, on=["model", "quant_variant"]).merge(cps, on=["model", "quant_variant"])

    q8 = out[out["quant_variant"].astype(str).eq("Q8_0")][["model", "gen_center"]].rename(
        columns={"gen_center": "q8_gen_center"}
    )
    out = out.merge(q8, on="model", how="left")
    out["speedup_vs_q8"] = out["q8_gen_center"] / out["gen_center"]
    out["gen_share"] = out["gen_center"] / out["e2e_center"]
    return out.sort_values(["model", "quant_variant"]).reset_index(drop=True)


def _err_from_center(center: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return asymmetric errorbar array shape (2, n) in same units as center."""
    yerr = np.vstack([np.maximum(center - lo, 0.0), np.maximum(hi - center, 0.0)])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    return yerr


def grouped_bar_ln(
    summary: pd.DataFrame,
    center_col: str,
    lo_col: str,
    hi_col: str,
    out_dir: Path,
    name: str,
    title: str,
    ylabel: str,
    phase_label: str,
    scale: float = 1.0,
    label_fmt: str = "{:.1f}",
    y_formatter: mticker.Formatter | None = None,
) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    for i, variant in enumerate(VARIANT_ORDER):
        part = plot[plot["quant_variant"].astype(str).eq(variant)].set_index("model").reindex(MODEL_ORDER)
        centers = part[center_col].to_numpy(dtype=float) / scale
        los = part[lo_col].to_numpy(dtype=float) / scale
        his = part[hi_col].to_numpy(dtype=float) / scale
        yerr = _err_from_center(centers, los, his)
        offset = (i - (len(VARIANT_ORDER) - 1) / 2) * width
        color = COLORS[variant]
        bars = ax.bar(
            x + offset,
            centers,
            width=width,
            label=variant,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
            yerr=yerr,
            error_kw={"ecolor": color["dark"], "capsize": 2.5, "elinewidth": 0.9},
        )
        finite = centers[np.isfinite(centers)]
        pad = max(finite.max() * 0.02, 0.05) if len(finite) else 0.05
        for bar, val in zip(bars, centers):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    label_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=TOKENS["ink"],
                    fontfamily=MONO_FONT_FAMILY[0],
                )
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel(ylabel)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    ax.legend(title="Kwantyzacja", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xlabel("")
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. {LN_NOTE}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, out_dir, name)


def plot_grouped_by_variant_ln(
    summary: pd.DataFrame,
    center_col: str,
    lo_col: str,
    hi_col: str,
    out_dir: Path,
    name: str,
    title: str,
    ylabel: str,
    phase_label: str,
    scale: float = 1.0,
    label_fmt: str = "{:.0f}",
    y_formatter: mticker.Formatter | None = None,
) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(VARIANT_ORDER))
    width = 0.22
    for i, model in enumerate(MODEL_ORDER):
        part = plot[plot["model"].astype(str).eq(model)].set_index("quant_variant").reindex(VARIANT_ORDER)
        centers = part[center_col].to_numpy(dtype=float) / scale
        los = part[lo_col].to_numpy(dtype=float) / scale
        his = part[hi_col].to_numpy(dtype=float) / scale
        yerr = _err_from_center(centers, los, his)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        color = MODEL_COLORS[model]
        bars = ax.bar(
            x + offset,
            centers,
            width=width,
            label=model,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
            yerr=yerr,
            error_kw={"ecolor": color["dark"], "capsize": 2.5, "elinewidth": 0.9},
        )
        finite = centers[np.isfinite(centers)]
        pad = max(finite.max() * 0.015, 0.08) if len(finite) else 0.08
        for bar, val in zip(bars, centers):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    label_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=TOKENS["ink"],
                    fontfamily=MONO_FONT_FAMILY[0],
                )
    ax.set_xticks(x, VARIANT_ORDER)
    ax.set_ylabel(ylabel)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xlabel("Wariant kwantyzacji")
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. {LN_NOTE}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, out_dir, name)


def plot_e2e_decomposition_ln(summary: pd.DataFrame, out_dir: Path, name: str, phase_label: str) -> None:
    """Three panels (one per model) so x-labels are only quant variants — no overlap."""
    from matplotlib.patches import Patch

    plot = summary.copy()
    plot["ttft_s"] = plot["ttft_center"] / 1000.0
    plot["generation_s"] = plot["gen_center"] / 1000.0
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(12.5, 5.6), sharey=True)
    if len(MODEL_ORDER) == 1:
        axes = [axes]
    for ax, model in zip(axes, MODEL_ORDER):
        part = plot[plot["model"].astype(str).eq(model)].set_index("quant_variant").reindex(VARIANT_ORDER)
        x = np.arange(len(VARIANT_ORDER))
        ttft = part["ttft_s"].to_numpy(dtype=float)
        gen = part["generation_s"].to_numpy(dtype=float)
        ax.bar(x, ttft, color="#1f77b4", edgecolor="#174f78", linewidth=1.0)
        ax.bar(
            x,
            gen,
            bottom=ttft,
            color="#ff7f0e",
            edgecolor="#8c4a0a",
            linewidth=1.0,
        )
        ax.set_xticks(x, VARIANT_ORDER)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.set_title(model, fontsize=11, color=TOKENS["ink"])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("")
    axes[0].set_ylabel("Czas [s]")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    fig.suptitle(
        "E2E jako suma median modelu LN: TTFT + generacja",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        y=1.02,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#1f77b4", edgecolor="#174f78", label="TTFT (mediana LN)"),
            Patch(facecolor="#ff7f0e", edgecolor="#8c4a0a", label="Generacja (mediana LN)"),
        ],
        title="Składnik E2E",
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. Słupki = mediana modelu LN (e^μ) per składnik.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.05, 0.88, 0.96))
    save_fig(fig, out_dir, name)


def plot_speedup_ln(summary: pd.DataFrame, out_dir: Path, name: str, phase_label: str) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    variants = ["Q2_K", "Q4_0", "Q4_K_M"]
    for i, variant in enumerate(variants):
        part = plot[plot["quant_variant"].astype(str).eq(variant)].set_index("model").reindex(MODEL_ORDER)
        values = part["speedup_vs_q8"].to_numpy(dtype=float)
        offset = (i - (len(variants) - 1) / 2) * width
        color = COLORS[variant]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"{variant} względem Q8_0",
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
        )
        for bar, val in zip(bars, values):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.04,
                    f"{val:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=TOKENS["ink"],
                    fontfamily=MONO_FONT_FAMILY[0],
                )
    ax.axhline(1, color=NEUTRAL["dark"], linewidth=1.0, linestyle="--")
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel("Przyspieszenie względem Q8_0")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}x"))
    ax.legend(title="Wariant", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title("Przyspieszenie generacji względem Q8_0 (mediany modelu LN)")
    ax.set_xlabel("")
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. Speedup = mediana_LN(Q8) / mediana_LN(wariant) dla generation_ms.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, out_dir, name)


def plot_generation_vs_answer_length(df: pd.DataFrame, out_dir: Path, name: str, phase_label: str) -> None:
    plot = df[df["generation_ms"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        part = plot[plot["model"].astype(str).eq(model)]
        color = MODEL_COLORS[model]
        ax.scatter(
            part["answer_chars"],
            part["generation_ms"] / 1000.0,
            s=36,
            alpha=0.75,
            label=model,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=0.7,
        )
    ax.set_xlabel("Długość odpowiedzi [znaki]")
    ax.set_ylabel("Czas generacji [s]")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Długość odpowiedzi a czas generacji")
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. Punkty = pojedyncze pomiary (n≈100); proxy znaków, bez tokenów.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, out_dir, name)


def plot_size_latency_ln(summary: pd.DataFrame, out_dir: Path, name: str, phase_label: str) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        part = plot[plot["model"].astype(str).eq(model)]
        color = MODEL_COLORS[model]
        ax.scatter(
            part["gguf_size_gb"],
            part["gen_center"] / 1000.0,
            s=92,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
            label=model,
        )
        for _, row in part.iterrows():
            ax.errorbar(
                row["gguf_size_gb"],
                row["gen_center"] / 1000.0,
                yerr=[[(row["gen_center"] - row["gen_lo"]) / 1000.0], [(row["gen_hi"] - row["gen_center"]) / 1000.0]],
                fmt="none",
                ecolor=color["dark"],
                elinewidth=0.9,
                capsize=3,
            )
            ax.annotate(
                str(row["quant_variant"]),
                (row["gguf_size_gb"], row["gen_center"] / 1000.0),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                color=TOKENS["ink"],
            )
    ax.set_xlabel("Rozmiar pliku GGUF [GB]")
    ax.set_ylabel("Mediana LN czasu generacji [s]")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Rozmiar GGUF a mediana modelu LN czasu generacji")
    fig.text(
        0.5,
        0.01,
        f"{phase_label}. {LN_NOTE}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, out_dir, name)


def plot_ln_fit_matrix(df: pd.DataFrame, out_path: Path, phase_label: str) -> None:
    """3×4 grid: hist of generation [s] + fitted log-normal curve."""
    use_chart_theme()
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(VARIANT_ORDER),
        figsize=(11.5, 8.2),
        sharex=False,
        sharey=False,
    )
    for i, model in enumerate(MODEL_ORDER):
        for j, variant in enumerate(VARIANT_ORDER):
            ax = axes[i, j]
            part = df[(df["model"].astype(str) == model) & (df["quant_variant"].astype(str) == variant)]
            vals = part["generation_ms"].dropna().to_numpy(dtype=float)
            vals = vals[vals > 0] / 1000.0
            n = len(vals)
            if n == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "brak danych", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            else:
                ax.hist(vals, bins=min(20, max(8, n // 5)), density=True, color="#c5cad3", edgecolor="#7a828f", linewidth=0.6)
                shape, loc, scale = stats.lognorm.fit(vals, floc=0)
                x_max = float(np.percentile(vals, 99.5)) * 1.15
                xg = np.linspace(max(vals.min() * 0.5, 1e-6), x_max, 200)
                ax.plot(xg, stats.lognorm.pdf(xg, shape, loc=loc, scale=scale), color="#1f77b4", lw=1.6)
                ax.text(0.97, 0.95, f"n={n}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=TOKENS["muted"])
                ax.set_xlim(0, x_max)
            if i == 0:
                ax.set_title(variant, fontsize=10)
            if j == 0:
                ax.set_ylabel(model, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(False)
    fig.suptitle(
        f"Dopasowanie log-normalne: generation [s] — {phase_label}",
        fontsize=12,
        fontweight="semibold",
        color=TOKENS["ink"],
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Histogram generation [s] + dopasowany rozkład log-normalny.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gauss_fit_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """1×4 grid: hist of E2E [s] + fitted normal curve (Optuna selected)."""
    use_chart_theme()
    models = sorted(df["model"].dropna().unique().tolist())
    # Prefer a stable display order if known
    preferred = ["gemma3:4b", "gemma2:2b", "llama3.2:3b", "qwen2.5:3b"]
    models = [m for m in preferred if m in models] + [m for m in models if m not in preferred]

    fig, axes = plt.subplots(1, len(models), figsize=(12.5, 3.6), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        vals = df.loc[df["model"] == model, "e2e_ms"].dropna().to_numpy(dtype=float)
        vals = vals[vals > 0] / 1000.0
        n = len(vals)
        if n == 0:
            ax.set_axis_off()
            continue
        ax.hist(vals, bins=min(25, max(10, n // 30)), density=True, color="#c5cad3", edgecolor="#7a828f", linewidth=0.6)
        mu = float(np.mean(vals))
        sig = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        if sig > 0:
            x_lo = max(0.0, mu - 3.5 * sig)
            x_hi = mu + 3.5 * sig
            xg = np.linspace(x_lo, x_hi, 200)
            ax.plot(xg, stats.norm.pdf(xg, mu, sig), color="#d62728", lw=1.6)
            ax.set_xlim(x_lo, x_hi)
        ax.set_title(model, fontsize=10)
        ax.text(0.97, 0.95, f"n={n}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=TOKENS["muted"])
        ax.tick_params(labelsize=7)
        ax.grid(False)
        ax.set_xlabel("E2E [s]", fontsize=8)
    axes[0].set_ylabel("gęstość", fontsize=8)
    fig.suptitle(
        "Dopasowanie normalne: E2E [s] — Optuna selected",
        fontsize=12,
        fontweight="semibold",
        color=TOKENS["ink"],
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Histogram E2E [s] + dopasowany rozkład normalny. Optuna selected.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_quant_phase(df: pd.DataFrame, out_dir: Path, phase: str, name_prefix: str = "") -> pd.DataFrame:
    """Generate LN figures for warm or cold. name_prefix='' for warm (TeX stems), 'cold_' for cold."""
    use_chart_theme()
    summary = build_ln_summary(df)
    phase_label = "Warm cache, n≈100" if phase == "warm" else "Cold cache, n≈100"
    p = name_prefix

    grouped_bar_ln(
        summary,
        "gen_center",
        "gen_lo",
        "gen_hi",
        out_dir,
        f"{p}01_generation_wedlug_wariantu",
        "Czas generacji — mediana modelu LN według wariantu",
        "Czas generacji [s]",
        phase_label,
        scale=1000.0,
        label_fmt="{:.1f}",
        y_formatter=mticker.StrMethodFormatter("{x:.0f}s"),
    )
    grouped_bar_ln(
        summary,
        "ttft_center",
        "ttft_lo",
        "ttft_hi",
        out_dir,
        f"{p}02_ttft_wedlug_wariantu",
        "TTFT — mediana modelu LN według wariantu",
        "TTFT [s]",
        phase_label,
        scale=1000.0,
        label_fmt="{:.1f}",
        y_formatter=mticker.StrMethodFormatter("{x:.0f}s"),
    )
    plot_e2e_decomposition_ln(summary, out_dir, f"{p}03_dekompozycja_e2e", phase_label)
    plot_speedup_ln(summary, out_dir, f"{p}04_przyspieszenie_wzgledem_q8", phase_label)
    plot_generation_vs_answer_length(df, out_dir, f"{p}05_generacja_a_dlugosc_odpowiedzi", phase_label)
    plot_size_latency_ln(summary, out_dir, f"{p}06_rozmiar_gguf_a_generacja", phase_label)
    plot_grouped_by_variant_ln(
        summary,
        "gen_center",
        "gen_lo",
        "gen_hi",
        out_dir,
        f"{p}07_generation_ms_wedlug_wariantu",
        "Czas generacji [ms] — mediana modelu LN, bez prefill",
        "Czas generacji [ms]",
        phase_label,
        scale=1.0,
        label_fmt="{:.0f}",
        y_formatter=mticker.StrMethodFormatter("{x:.0f}"),
    )
    plot_grouped_by_variant_ln(
        summary,
        "cps_center",
        "cps_lo",
        "cps_hi",
        out_dir,
        f"{p}08_szybkosc_generacji_tok_s_wedlug_wariantu",
        "Szybkość generacji [znaki/s] — mediana modelu LN",
        "Szybkość [znaki/s] (proxy tok/s)",
        phase_label,
        scale=1.0,
        label_fmt="{:.1f}",
        y_formatter=mticker.StrMethodFormatter("{x:.1f}"),
    )
    return summary


def generate_optuna_figures() -> None:
    """Quality figures unchanged; Pareto uses Gauss mean±σ E2E."""
    import thesis_statistical_analysis as tsa

    use_chart_theme()
    OPTUNA_OUT.mkdir(parents=True, exist_ok=True)

    # Restrict loader to results-selected + all results-* under analysis/
    df = tsa.load_rows()
    ts = tsa.trial_summary(df)
    model_tbl = tsa.model_summary(df)
    # Enrich model_tbl with Gauss E2E stats for selected
    gauss_rows = []
    for (dataset, model), g in df.groupby(["_dataset", "model"]):
        e2e = g["e2e_ms"].dropna().to_numpy(dtype=float)
        gauss_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "e2e_mean_ms": float(np.mean(e2e)) if len(e2e) else float("nan"),
                "e2e_std_ms": float(np.std(e2e, ddof=1)) if len(e2e) > 1 else float("nan"),
            }
        )
    gauss_df = pd.DataFrame(gauss_rows)
    model_tbl = model_tbl.merge(gauss_df, on=["dataset", "model"], how="left")

    case_tbl = (
        df.groupby(["_dataset", "case_id", "category"])
        .agg(
            n=("case_id", "count"),
            score_mean=("composite_score", "mean"),
            hallucination_rate=("hallucination", "mean"),
            e2e_median_ms=("e2e_ms", "median"),
        )
        .reset_index()
    )
    reg_tbl = tsa.regression_comparison(ts)

    def _savefig(name: str) -> None:
        for ext in ("png", "pdf"):
            plt.savefig(OPTUNA_OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
        plt.close()

    plot_df = ts.dropna(subset=["score_mean", "hallucination_rate"])
    plt.figure(figsize=(8.2, 5.4))
    sns.scatterplot(
        data=plot_df,
        x="hallucination_rate",
        y="score_mean",
        hue="model",
        palette=MODEL_PALETTE,
        marker="o",
        s=70,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
    )
    sns.regplot(data=plot_df, x="hallucination_rate", y="score_mean", scatter=False, color="black", line_kws={"lw": 1.7})
    plt.title("Średni wynik łączny jest zdominowany przez odsetek halucynacji")
    plt.xlabel("Odsetek halucynacji w trialu")
    plt.ylabel("Średni wynik łączny triala")
    plt.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _savefig("01_score_vs_hallucination_all")

    selected = ts[ts["_dataset"] == "results-selected-9.05"].copy()
    plt.figure(figsize=(8.2, 5.2))
    sns.scatterplot(
        data=selected,
        x="hallucination_rate",
        y="score_mean",
        hue="model",
        palette=MODEL_PALETTE,
        marker="o",
        s=80,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
    )
    sns.regplot(data=selected, x="hallucination_rate", y="score_mean", scatter=False, color="black", line_kws={"lw": 1.7})
    plt.title("Wybrane runy: wynik triala podąża za flagą halucynacji")
    plt.xlabel("Odsetek halucynacji w trialu")
    plt.ylabel("Średni wynik łączny triala")
    plt.legend(title="Model", loc="best", frameon=False)
    _savefig("02_score_vs_hallucination_selected")

    plt.figure(figsize=(8.2, 5.2))
    sns.stripplot(
        data=selected, x="model", y="nonhall_score_mean", hue="model", palette=MODEL_PALETTE, jitter=0.18, size=6, legend=False
    )
    sns.pointplot(data=selected, x="model", y="nonhall_score_mean", color="black", errorbar="sd", markers="_", linestyles="none")
    plt.title("Jakość warunkowa po usunięciu odpowiedzi z halucynacją")
    plt.xlabel("")
    plt.ylabel("Średni wynik łączny przy hallucination_flag=false")
    plt.xticks(rotation=20, ha="right")
    _savefig("03_nonhall_score_spread_selected")

    # Pareto: Gauss mean ± σ
    selected_models = model_tbl[model_tbl["dataset"] == "results-selected-9.05"].copy()
    plt.figure(figsize=(7.8, 5.4))
    ax = plt.gca()
    for _, row in selected_models.iterrows():
        color = MODEL_PALETTE.get(row["model"], "#333333")
        x = row["e2e_mean_ms"] / 1000.0
        y = row["hallucination_rate"]
        xerr = (row["e2e_std_ms"] / 1000.0) if math.isfinite(row["e2e_std_ms"]) else 0.0
        size = 80 + 400 * (row["score_mean"] - selected_models["score_mean"].min()) / max(
            selected_models["score_mean"].max() - selected_models["score_mean"].min(), 1e-9
        )
        ax.errorbar(x, y, xerr=xerr, fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=4, markersize=0)
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(row["model"], (x, y), xytext=(7, 3), textcoords="offset points", fontsize=9)
    ax.set_title("Kompromis Pareto: średnie E2E (±σ) a halucynacje")
    ax.set_xlabel("Średnia E2E [s] (±σ, model Gaussa)")
    ax.set_ylabel("Odsetek halucynacji")
    fig = plt.gcf()
    fig.text(
        0.5,
        0.01,
        "Optuna results-selected-9.05. Oś X: mean±σ (Gauss), nie mediana. Rozmiar punktu ~ score_mean.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _savefig("04_latency_hallucination_pareto_selected")

    selected_cases = case_tbl[case_tbl["_dataset"] == "results-selected-9.05"].nlargest(12, "hallucination_rate")
    plt.figure(figsize=(8.5, 5.2))
    sns.barplot(data=selected_cases, y="case_id", x="hallucination_rate", hue="category", dodge=False)
    plt.title("Przypadki testowe najczęściej oznaczane jako halucynacje")
    plt.xlabel("Odsetek halucynacji")
    plt.ylabel("")
    plt.legend(title="Kategoria", loc="lower right", frameon=False)
    _savefig("05_case_hallucination_rates_selected")

    selected_corr = tsa.hparam_correlations(selected).query(
        "target in ['score_mean', 'hallucination_rate', 'nonhall_score_mean', 'e2e_median_ms']"
    )
    selected_corr = selected_corr.copy()
    selected_corr["target"] = selected_corr["target"].map(TARGET_LABELS)
    selected_corr = selected_corr.pivot_table(index=["model", "target"], columns="hparam", values="pearson_r")
    plt.figure(figsize=(8.2, 7.2))
    sns.heatmap(selected_corr, cmap="vlag", center=0, annot=True, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Jednowymiarowe korelacje hiperparametrów w wybranych runach")
    plt.xlabel("")
    plt.ylabel("")
    _savefig("06_hparam_correlation_heatmap_selected")

    r2_plot = reg_tbl[
        (reg_tbl["dataset"] == "results-selected-9.05") & (reg_tbl["target"].isin(["score_mean", "nonhall_score_mean", "e2e_median_ms"]))
    ].copy()
    r2_plot["target"] = r2_plot["target"].map(TARGET_LABELS)
    r2_plot["regression"] = r2_plot["regression"].map(REGRESSION_LABELS)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=r2_plot, x="target", y="r2", hue="regression")
    plt.title("Siła wyjaśniająca modeli regresyjnych w wybranych runach")
    plt.xlabel("")
    plt.ylabel("R²")
    plt.xticks(rotation=10, ha="right")
    plt.legend(title="Regresja", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _savefig("07_regression_r2_comparison_selected")

    # Export Gauss E2E table for review
    selected_models.to_csv(TABLE_OUT / "optuna_selected_gauss_e2e.csv", index=False)

    # Fit matrix: Gauss on E2E
    sel = df[df["_dataset"] == "results-selected-9.05"].copy()
    plot_gauss_fit_matrix(sel, FIT_OUT / "optuna_e2e_gauss_fit_matrix")


def write_readme() -> None:
    text = textwrap.dedent(
        f"""\
        # Review figures — rozdział 5 (LN / Gauss)

        Wygenerowane przez `generate_review_figures_ch5.py`. **Nie są jeszcze wpięte w TeX.**

        ## Polityka

        | Folder | Estymator |
        |---|---|
        | `warm/`, `cold/` | **Log-normal**: wysokość słupka = mediana modelu LN (e^μ); przedział = e^(μ±σ) |
        | `optuna/` jakość | bez zmian (mean / rate) |
        | `optuna/04_latency_hallucination_pareto_selected` | **Gauss**: mean E2E ± σ |

        Nomenklatura: **nie** „centrum” / moda PDF — wysokość to **mediana modelu LN** (e^μ).

        ## Matryce dopasowania (`fit/`)

        - `warm_generation_ln_fit_matrix` — hist generation + krzywa **LN** (3×4)
        - `cold_generation_ln_fit_matrix` — to samo, cold
        - `optuna_e2e_gauss_fit_matrix` — hist E2E + krzywa **Normal** (4 modele Optuny)

        ## Dane

        - Warm: `{WARM_SRC}`
        - Cold: `{COLD_SRC}`
        - Optuna: `benchmark/analysis/results-*/` (głównie `results-selected-9.05`)

        ## Mapa: stary plik w TeXu → nowy

        | Stary stem (wykresy/) | Nowy (warm/ lub optuna/) | Estymator |
        |---|---|---|
        | `01_mediana_generacji_wedlug_wariantu` | `warm/01_generation_wedlug_wariantu` | LN |
        | `02_mediana_ttft_wedlug_wariantu` | `warm/02_ttft_wedlug_wariantu` | LN |
        | `03_dekompozycja_e2e` | `warm/03_dekompozycja_e2e` | LN |
        | `04_przyspieszenie_wzgledem_q8` | `warm/04_przyspieszenie_wzgledem_q8` | LN speedup |
        | `05_generacja_a_dlugosc_odpowiedzi` | `warm/05_generacja_a_dlugosc_odpowiedzi` | scatter |
        | `06_rozmiar_gguf_a_generacja` | `warm/06_rozmiar_gguf_a_generacja` | LN |
        | `07_mediana_generacji_ms_wedlug_wariantu` | `warm/07_generation_ms_wedlug_wariantu` | LN |
        | `08_szybkosc_generacji_tok_s_wedlug_wariantu` | `warm/08_szybkosc_generacji_tok_s_wedlug_wariantu` | LN |
        | `02_score_vs_hallucination_selected` | `optuna/02_...` | jakość |
        | `03_nonhall_score_spread_selected` | `optuna/03_...` | jakość |
        | `04_latency_hallucination_pareto_selected` | `optuna/04_...` | **Gauss mean±σ** |
        | `05_case_hallucination_rates_selected` | `optuna/05_...` | jakość |
        | `06_hparam_correlation_heatmap_selected` | `optuna/06_...` | jakość |
        | `07_regression_r2_comparison_selected` | `optuna/07_...` | jakość |

        Cold: te same stemy z prefixem `cold_` w `cold/`.

        ## Tabele

        - `tables/warm_ln_summary.csv`, `tables/cold_ln_summary.csv` — μ, σ, mediana LN e^μ, e^(μ±σ), speedup
        - `tables/optuna_selected_gauss_e2e.csv` — mean/σ E2E per model

        ## Podpisy

        Propozycje LaTeX: `CAPTIONS.tex` (nie includowane w `main.tex`).
        """
    )
    (OUT / "README.md").write_text(text, encoding="utf-8")


def write_captions() -> None:
    text = textwrap.dedent(
        r"""
        % Propozycje podpisów do rozdz. 5 — NIE includowane w main.tex.
        % Po akceptacji review: podmienić \caption w tex/5-badania.tex.

        % --- Optuna / jakość (bez zmian modelu) ---
        % fig:score_vs_hall
        \caption{Zależność \emph{composite score} od wskaźnika halucynacji
                 (\texttt{results-selected-9.05}). Flaga halucynacji determinuje
                 score w~ponad 99\% przypadków.}

        % fig:regression_r2
        \caption{Porównanie $R^2$ modeli regresji wyjaśniających wariancję score
                 (\texttt{results-selected-9.05}).}

        % fig:nonhall_spread
        \caption{Rozpiętość \emph{nonhall\_score} w~porównaniu z~\emph{composite score}
                 (\texttt{results-selected-9.05}). Hiperparametry nie różnicują
                 jakości odpowiedzi bez halucynacji.}

        % fig:case_hall_rates
        \caption{Wskaźnik halucynacji per przypadek testowy
                 (\texttt{results-selected-9.05}).}

        % fig:hparam_heatmap
        \caption{Mapa korelacji hiperparametrów z~metrykami wynikowymi
                 (\texttt{results-selected-9.05}). Brak silnych korelacji z~jakością.}

        % fig:pareto_latency_hall  — Gauss mean±σ (nie mediana)
        \caption{Kompromis Pareto: średnie opóźnienie E2E ($\pm\sigma$, model Gaussa)
                 vs wskaźnik halucynacji (\texttt{results-selected-9.05}).
                 Rozmiar punktu odpowiada średniemu \emph{composite score}.}

        % fit: optuna_e2e_gauss_fit_matrix
        \caption{Dopasowanie rozkładu normalnego do opóźnienia E2E
                 (\texttt{results-selected-9.05}). Histogram oraz gęstość
                 $\mathcal{N}(\hat\mu,\hat\sigma^2)$ per model.}

        % fit: warm/cold generation LN
        \caption{Dopasowanie rozkładu log-normalnego do \emph{generation\_ms}
                 (warm/cold cache, $n\approx 100$). Histogram oraz gęstość LN
                 per model i wariant kwantyzacji.}

        % --- Warm kwantyzacja (log-normal) ---
        % fig:dekompozycja_e2e
        \caption{Dekompozycja E2E na TTFT i~\emph{generation\_ms}.
                 Wysokości słupków to mediany modelu log-normalnego $e^{\hat\mu}$
                 (warm cache, $n\approx 100$). Kwantyzacja wpływa na czas generacji,
                 nie na prefill.}

        % fig:speedup_q8
        \caption{Przyspieszenie \emph{generation\_ms} względem Q8\_0
                 w~eksperymencie warm-cache.
                 Speedup $= e^{\hat\mu_{\mathrm{Q8}}}/e^{\hat\mu_{\mathrm{wariant}}}$
                 (stosunek median modelu LN).}

        % fig:gen_ms_variant
        \caption{Czas \emph{generation\_ms} według wariantu kwantyzacji
                 (mediana modelu LN $e^{\hat\mu}$, przedział $e^{\hat\mu\pm\hat\sigma}$;
                 warm cache, $n\approx 100$).}

        % fig:tok_s_variant
        \caption{Szybkość generacji [znaki/s] (proxy tok/s) według wariantu kwantyzacji.
                 Mediana modelu LN na per-rekord $\mathrm{answer\_chars}/(\mathrm{generation\_ms}/1000)$.}

        % fig:gen_vs_length
        \caption{Zależność czasu generacji od długości odpowiedzi
                 (anomalia Q2\_K; warm cache, $n\approx 100$).}

        % fig:ttft_warm
        \caption{TTFT według wariantu kwantyzacji (warm cache):
                 mediana modelu LN $e^{\hat\mu}$ z~przedziałem $e^{\hat\mu\pm\hat\sigma}$.}

        % --- Cold (jeszcze nie w TeXu; do wstawienia później) ---
        % cold_03_dekompozycja_e2e
        \caption{Dekompozycja E2E na TTFT i~\emph{generation\_ms} (cold cache).
                 Mediany modelu LN $e^{\hat\mu}$, $n\approx 100$.}

        % cold_02_ttft / cold_07_generation / cold_04_speedup — analogicznie jak warm.
        """
    ).lstrip()
    (OUT / "CAPTIONS.tex").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    WARM_OUT.mkdir(parents=True, exist_ok=True)
    COLD_OUT.mkdir(parents=True, exist_ok=True)
    FIT_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading warm…")
    warm_df = load_quant_jsonl(WARM_SRC)
    print("Loading cold…")
    cold_df = load_quant_jsonl(COLD_SRC)

    print("Warm LN figures…")
    warm_summary = generate_quant_phase(warm_df, WARM_OUT, "warm", name_prefix="")
    warm_summary.to_csv(TABLE_OUT / "warm_ln_summary.csv", index=False)

    print("Cold LN figures…")
    cold_summary = generate_quant_phase(cold_df, COLD_OUT, "cold", name_prefix="cold_")
    cold_summary.to_csv(TABLE_OUT / "cold_ln_summary.csv", index=False)

    print("LN fit matrices…")
    plot_ln_fit_matrix(warm_df, FIT_OUT / "warm_generation_ln_fit_matrix", "Warm cache, n≈100")
    plot_ln_fit_matrix(cold_df, FIT_OUT / "cold_generation_ln_fit_matrix", "Cold cache, n≈100")

    print("Optuna figures…")
    generate_optuna_figures()

    write_readme()
    write_captions()
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
