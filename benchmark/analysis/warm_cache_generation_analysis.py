#!/usr/bin/env python3
"""Analiza benchmarku kwantyzacji przy ciepłym cache.

Skrypt czyta pełny plik JSONL z benchmarku przy ciepłym cache i zapisuje artefakty
do wykorzystania w pracy:
- raport Markdown,
- tabele CSV,
- wykresy PNG, SVG i PDF.

Analiza rozdziela TTFT od czasu generacji, żeby wpływ kwantyzacji nie był
przykrywany przez koszt prefilla i trafienia/chybienia cache.
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


ROOT = SCRIPT_ROOT / "quant-warm-cache"
SOURCE = ROOT / "warm_cache_generation_full_20260618_20260621.jsonl"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
REPORT = ROOT / "report.md"

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

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

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


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=82, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=118, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    fig.subplots_adjust(top=max(0.68, 0.86 - 0.04 * (title_lines - 1) - 0.03 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        left,
        0.93 - 0.04 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_s(ms: float | int | None) -> str:
    if pd.isna(ms):
        return "NA"
    return f"{float(ms) / 1000:.2f}s"


def fmt_pct(value: float | int | None) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.0f}%"


def fmt_float(value: float | int | None, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_rows() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    rows: list[dict] = []
    with SOURCE.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [
        "gguf_size_gb",
        "ttft_ms",
        "e2e_ms",
        "generation_ms",
        "answer_chars",
        "question_idx",
        "tokens_per_s",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cache_hit"] = df["cache_hit"].fillna(False).astype(bool)
    df["valid_generation"] = df["generation_ms"].notna()
    df["empty_answer"] = df["answer_chars"].fillna(0).eq(0)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["quant_variant"] = pd.Categorical(df["quant_variant"], categories=VARIANT_ORDER, ordered=True)
    df["generation_s"] = df["generation_ms"] / 1000
    df["ttft_s"] = df["ttft_ms"] / 1000
    df["e2e_s"] = df["e2e_ms"] / 1000
    df["ms_per_char"] = np.where(df["answer_chars"] > 0, df["generation_ms"] / df["answer_chars"], np.nan)
    if "tokens_per_s" in df.columns and df["tokens_per_s"].notna().any():
        df["tokens_per_s_effective"] = df["tokens_per_s"]
    else:
        df["tokens_per_s_effective"] = np.where(
            df["generation_ms"] > 0,
            df["answer_chars"] / (df["generation_ms"] / 1000),
            np.nan,
        )
    return df.sort_values(["model", "quant_variant", "question_idx"]).reset_index(drop=True)


def q90(series: pd.Series) -> float:
    return series.dropna().quantile(0.90)


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["model", "quant_variant"]
    summary = (
        df.groupby(group_cols, observed=True)
        .agg(
            rows=("case_id", "count"),
            valid_generation_rows=("valid_generation", "sum"),
            null_generation_rows=("generation_ms", lambda s: int(s.isna().sum())),
            cache_hit_rate=("cache_hit", "mean"),
            empty_answer_rate=("empty_answer", "mean"),
            gguf_size_gb=("gguf_size_gb", "median"),
            median_ttft_ms=("ttft_ms", "median"),
            p90_ttft_ms=("ttft_ms", q90),
            median_generation_ms=("generation_ms", "median"),
            p90_generation_ms=("generation_ms", q90),
            median_e2e_ms=("e2e_ms", "median"),
            p90_e2e_ms=("e2e_ms", q90),
            median_answer_chars=("answer_chars", "median"),
            median_ms_per_char=("ms_per_char", "median"),
            median_tokens_per_s=("tokens_per_s_effective", "median"),
        )
        .reset_index()
    )

    q8 = summary[summary["quant_variant"].astype(str).eq("Q8_0")][
        ["model", "median_generation_ms", "median_ms_per_char"]
    ].rename(
        columns={
            "median_generation_ms": "q8_median_generation_ms",
            "median_ms_per_char": "q8_median_ms_per_char",
        }
    )
    summary = summary.merge(q8, on="model", how="left")
    summary["speedup_vs_q8_generation"] = summary["q8_median_generation_ms"] / summary["median_generation_ms"]
    summary["speedup_vs_q8_ms_per_char"] = summary["q8_median_ms_per_char"] / summary["median_ms_per_char"]
    summary["generation_share_of_e2e"] = summary["median_generation_ms"] / summary["median_e2e_ms"]
    summary = summary.sort_values(["model", "quant_variant"]).reset_index(drop=True)

    speedup = summary[
        [
            "model",
            "quant_variant",
            "gguf_size_gb",
            "median_generation_ms",
            "q8_median_generation_ms",
            "speedup_vs_q8_generation",
            "median_ms_per_char",
            "speedup_vs_q8_ms_per_char",
        ]
    ].copy()

    quality = (
        df.groupby(group_cols, observed=True)
        .agg(
            rows=("case_id", "count"),
            valid_generation_rows=("valid_generation", "sum"),
            null_ttft_rows=("ttft_ms", lambda s: int(s.isna().sum())),
            null_generation_rows=("generation_ms", lambda s: int(s.isna().sum())),
            empty_answer_rows=("empty_answer", "sum"),
            cache_hit_rows=("cache_hit", "sum"),
            cache_hit_rate=("cache_hit", "mean"),
            min_answer_chars=("answer_chars", "min"),
            median_answer_chars=("answer_chars", "median"),
            max_answer_chars=("answer_chars", "max"),
        )
        .reset_index()
        .sort_values(["model", "quant_variant"])
    )

    question = (
        df.groupby(["question_idx", "case_id"], observed=True)
        .agg(
            rows=("case_id", "count"),
            median_generation_ms=("generation_ms", "median"),
            median_ttft_ms=("ttft_ms", "median"),
            median_answer_chars=("answer_chars", "median"),
            null_generation_rows=("generation_ms", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .sort_values("question_idx")
    )

    common_pl = {
        "model": "model",
        "quant_variant": "wariant_kwantyzacji",
        "rows": "liczba_rekordow",
        "valid_generation_rows": "poprawne_generacje",
        "null_generation_rows": "brak_generation_ms",
        "cache_hit_rate": "odsetek_trafien_cache",
        "empty_answer_rate": "odsetek_pustych_odpowiedzi",
        "gguf_size_gb": "rozmiar_gguf_gb",
        "median_ttft_ms": "mediana_ttft_ms",
        "p90_ttft_ms": "p90_ttft_ms",
        "median_generation_ms": "mediana_generacji_ms",
        "p90_generation_ms": "p90_generacji_ms",
        "median_e2e_ms": "mediana_e2e_ms",
        "p90_e2e_ms": "p90_e2e_ms",
        "median_answer_chars": "mediana_znakow_odpowiedzi",
        "median_ms_per_char": "mediana_ms_na_znak",
        "median_tokens_per_s": "mediana_tokenow_na_s",
        "q8_median_generation_ms": "q8_mediana_generacji_ms",
        "q8_median_ms_per_char": "q8_mediana_ms_na_znak",
        "speedup_vs_q8_generation": "przyspieszenie_generacji_wzgledem_q8",
        "speedup_vs_q8_ms_per_char": "przyspieszenie_ms_na_znak_wzgledem_q8",
        "generation_share_of_e2e": "udzial_generacji_w_e2e",
    }
    quality_pl = {
        **common_pl,
        "null_ttft_rows": "brak_ttft",
        "empty_answer_rows": "puste_odpowiedzi",
        "cache_hit_rows": "trafienia_cache",
        "min_answer_chars": "min_znakow_odpowiedzi",
        "max_answer_chars": "max_znakow_odpowiedzi",
    }
    question_pl = {
        "question_idx": "numer_pytania",
        "case_id": "id_przypadku",
        "rows": "liczba_rekordow",
        "median_generation_ms": "mediana_generacji_ms",
        "median_ttft_ms": "mediana_ttft_ms",
        "median_answer_chars": "mediana_znakow_odpowiedzi",
        "null_generation_rows": "brak_generation_ms",
    }

    summary.rename(columns=common_pl).to_csv(TABLE / "podsumowanie_model_wariant.csv", index=False)
    speedup.rename(columns=common_pl).to_csv(TABLE / "przyspieszenie_wzgledem_q8.csv", index=False)
    quality.rename(columns=quality_pl).to_csv(TABLE / "jakosc_danych.csv", index=False)
    question.rename(columns=question_pl).to_csv(TABLE / "podsumowanie_pytan.csv", index=False)
    return summary, speedup, quality, question


def grouped_bar(summary: pd.DataFrame, value_col: str, name: str, title: str, subtitle: str, ylabel: str) -> None:
    plot = summary.copy()
    plot[value_col] = plot[value_col] / 1000
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    for i, variant in enumerate(VARIANT_ORDER):
        part = plot[plot["quant_variant"].astype(str).eq(variant)].set_index("model").reindex(MODEL_ORDER)
        values = part[value_col].to_numpy(dtype=float)
        offset = (i - (len(VARIANT_ORDER) - 1) / 2) * width
        color = COLORS[variant]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=variant,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
        )
        for bar, val in zip(bars, values):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(np.nanmax(values) * 0.015, 0.08),
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=TOKENS["ink"],
                    fontfamily=MONO_FONT_FAMILY[0],
                )
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Kwantyzacja", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xlabel("")
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9, color=TOKENS["muted"])
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, name)


def plot_e2e_decomposition(summary: pd.DataFrame) -> None:
    plot = summary.copy()
    plot["label"] = plot["model"].astype(str) + "\n" + plot["quant_variant"].astype(str)
    plot["ttft_s"] = plot["median_ttft_ms"] / 1000
    plot["generation_s"] = plot["median_generation_ms"] / 1000
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot))
    ax.bar(x, plot["ttft_s"], color="#1f77b4", edgecolor="#174f78", linewidth=1.0, label="TTFT")
    ax.bar(
        x,
        plot["generation_s"],
        bottom=plot["ttft_s"],
        color="#ff7f0e",
        edgecolor="#8c4a0a",
        linewidth=1.0,
        label="Generacja",
    )
    ax.set_xticks(x, plot["label"])
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.set_ylabel("Czas [s]")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Składnik E2E", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title("Mediana E2E jako suma TTFT i czasu generacji")
    ax.set_xlabel("")
    fig.text(
        0.5,
        0.01,
        "Run z ciepłym cache; słupki pokazują medianę w sekundach dla każdej kombinacji modelu i kwantyzacji.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, "03_dekompozycja_e2e")


def plot_speedup(speedup: pd.DataFrame) -> None:
    plot = speedup.copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    variants = ["Q2_K", "Q4_0", "Q4_K_M"]
    for i, variant in enumerate(variants):
        part = plot[plot["quant_variant"].astype(str).eq(variant)].set_index("model").reindex(MODEL_ORDER)
        values = part["speedup_vs_q8_generation"].to_numpy(dtype=float)
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
    ax.set_title("Przyspieszenie generacji względem wariantu Q8_0")
    ax.set_xlabel("")
    fig.text(
        0.5,
        0.01,
        "Wartość powyżej 1,0x oznacza krótszą medianę czasu generacji niż w Q8_0 dla tej samej rodziny modelu.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, "04_przyspieszenie_wzgledem_q8")


def plot_grouped_by_variant(
    summary: pd.DataFrame,
    value_col: str,
    name: str,
    title: str,
    subtitle: str,
    ylabel: str,
    label_format: str,
    y_formatter: mticker.Formatter,
) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(VARIANT_ORDER))
    width = 0.22
    for i, model in enumerate(MODEL_ORDER):
        part = plot[plot["model"].astype(str).eq(model)].set_index("quant_variant").reindex(VARIANT_ORDER)
        values = part[value_col].to_numpy(dtype=float)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        color = MODEL_COLORS[model]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
        )
        finite = values[np.isfinite(values)]
        label_offset = max(finite.max() * 0.015, 0.08) if len(finite) else 0.08
        for bar, val in zip(bars, values):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + label_offset,
                    label_format.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=TOKENS["ink"],
                    fontfamily=MONO_FONT_FAMILY[0],
                )
    ax.set_xticks(x, VARIANT_ORDER)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xlabel("Wariant kwantyzacji")
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9, color=TOKENS["muted"])
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, name)


def plot_generation_vs_answer_length(df: pd.DataFrame) -> None:
    plot = df[df["generation_ms"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        part = plot[plot["model"].astype(str).eq(model)]
        color = MODEL_COLORS[model]
        ax.scatter(
            part["answer_chars"],
            part["generation_ms"] / 1000,
            s=44,
            alpha=0.82,
            label=model,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=0.8,
        )
    ax.set_xlabel("Długość odpowiedzi [znaki]")
    ax.set_ylabel("Czas generacji [s]")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Długość odpowiedzi a czas generacji")
    fig.text(
        0.5,
        0.01,
        "Każdy punkt to jedno żądanie z poprawnym generation_ms; liczba znaków jest tylko przybliżeniem, bo benchmark nie zapisywał liczby tokenów.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, "05_generacja_a_dlugosc_odpowiedzi")


def plot_size_latency(summary: pd.DataFrame) -> None:
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        part = plot[plot["model"].astype(str).eq(model)]
        color = MODEL_COLORS[model]
        ax.scatter(
            part["gguf_size_gb"],
            part["median_generation_ms"] / 1000,
            s=92,
            color=color["base"],
            edgecolor=color["dark"],
            linewidth=1.0,
            label=model,
        )
        for _, row in part.iterrows():
            ax.annotate(
                str(row["quant_variant"]),
                (row["gguf_size_gb"], row["median_generation_ms"] / 1000),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                color=TOKENS["ink"],
            )
    ax.set_xlabel("Rozmiar pliku GGUF [GB]")
    ax.set_ylabel("Mediana czasu generacji [s]")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}s"))
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Rozmiar GGUF a mediana czasu generacji")
    fig.text(
        0.5,
        0.01,
        "Etykiety punktów pokazują wariant kwantyzacji; oś Y używa mediany generation_ms z żądań z ciepłym cache.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, "06_rozmiar_gguf_a_generacja")


def build_figures(df: pd.DataFrame, summary: pd.DataFrame, speedup: pd.DataFrame) -> None:
    grouped_bar(
        summary,
        "median_generation_ms",
        "01_mediana_generacji_wedlug_wariantu",
        "Mediana czasu generacji ujawnia wpływ kwantyzacji",
        "Mediana w sekundach dla 20 pytań ze stałym kontekstem; niżej oznacza szybciej.",
        "Mediana czasu generacji [s]",
    )
    grouped_bar(
        summary,
        "median_ttft_ms",
        "02_mediana_ttft_wedlug_wariantu",
        "Mediana TTFT przy ciepłym cache",
        "Czas do pierwszego tokenu dla tego samego system promptu; niższe wartości wskazują skuteczniejsze wykorzystanie prefix cache.",
        "Mediana TTFT [s]",
    )
    plot_e2e_decomposition(summary)
    plot_speedup(speedup)
    plot_grouped_by_variant(
        summary,
        "median_generation_ms",
        "07_mediana_generacji_ms_wedlug_wariantu",
        "Mediana czasu generacji [ms] — warm cache, bez prefill",
        "Mediana generation_ms = e2e_ms - ttft_ms; oś X pokazuje warianty kwantyzacji, a słupki modele.",
        "Mediana czasu generacji [ms]",
        "{:.0f}",
        mticker.StrMethodFormatter("{x:.0f}"),
    )
    plot_grouped_by_variant(
        summary,
        "median_tokens_per_s",
        "08_szybkosc_generacji_tok_s_wedlug_wariantu",
        "Szybkość generacji [tok/s] — warm cache, prefill wyizolowany",
        "W danych nie ma pola tokens_per_s, więc wartości policzono jako answer_chars / (generation_ms / 1000), a następnie zmedianowano.",
        "Szybkość generacji [tok/s]",
        "{:.1f}",
        mticker.StrMethodFormatter("{x:.1f}"),
    )
    plot_generation_vs_answer_length(df)
    plot_size_latency(summary)


def markdown_table(df: pd.DataFrame, cols: list[str], rename: dict[str, str] | None = None) -> str:
    out = df[cols].copy()
    if rename:
        out = out.rename(columns=rename)
    return out.to_markdown(index=False)


def report(summary: pd.DataFrame, speedup: pd.DataFrame, quality: pd.DataFrame, df: pd.DataFrame) -> str:
    total_rows = len(df)
    combos = summary.shape[0]
    null_generation = int(df["generation_ms"].isna().sum())
    cache_hit_rate = df["cache_hit"].mean()

    best_by_model = (
        summary.sort_values(["model", "median_generation_ms"])
        .groupby("model", observed=True)
        .head(1)[["model", "quant_variant", "median_generation_ms", "median_ttft_ms", "speedup_vs_q8_generation"]]
        .reset_index(drop=True)
    )

    q4_rows = speedup[speedup["quant_variant"].astype(str).isin(["Q4_0", "Q4_K_M"])].copy()
    q4_focus = q4_rows[["model", "quant_variant", "speedup_vs_q8_generation", "median_generation_ms"]].copy()
    q4_focus["generacja"] = q4_focus["median_generation_ms"].map(fmt_s)
    q4_focus["przyspieszenie"] = q4_focus["speedup_vs_q8_generation"].map(lambda x: fmt_float(x, 2) + "x")

    summary_view = summary.copy()
    summary_view["med_ttft"] = summary_view["median_ttft_ms"].map(fmt_s)
    summary_view["med_gen"] = summary_view["median_generation_ms"].map(fmt_s)
    summary_view["med_e2e"] = summary_view["median_e2e_ms"].map(fmt_s)
    summary_view["trafienia_cache"] = summary_view["cache_hit_rate"].map(fmt_pct)
    summary_view["przyspieszenie_q8"] = summary_view["speedup_vs_q8_generation"].map(lambda x: fmt_float(x, 2) + "x")
    summary_view["znaki"] = summary_view["median_answer_chars"].map(lambda x: fmt_float(x, 0))

    best_view = best_by_model.copy()
    best_view["med_gen"] = best_view["median_generation_ms"].map(fmt_s)
    best_view["med_ttft"] = best_view["median_ttft_ms"].map(fmt_s)
    best_view["przyspieszenie_q8"] = best_view["speedup_vs_q8_generation"].map(lambda x: fmt_float(x, 2) + "x")

    quality_view = quality.copy()
    quality_view["trafienia_cache"] = quality_view["cache_hit_rate"].map(fmt_pct)

    gemma3_q4km = summary[
        summary["model"].astype(str).eq("gemma3-4b") & summary["quant_variant"].astype(str).eq("Q4_K_M")
    ].iloc[0]
    gemma3_q8 = summary[
        summary["model"].astype(str).eq("gemma3-4b") & summary["quant_variant"].astype(str).eq("Q8_0")
    ].iloc[0]
    gemma4_q4km = summary[
        summary["model"].astype(str).eq("gemma4-e2b") & summary["quant_variant"].astype(str).eq("Q4_K_M")
    ].iloc[0]
    gemma4_q8 = summary[
        summary["model"].astype(str).eq("gemma4-e2b") & summary["quant_variant"].astype(str).eq("Q8_0")
    ].iloc[0]

    text = f"""# Ciepły cache ujawnia realny koszt generacji po kwantyzacji

## Podsumowanie

- **Ten przebieg benchmarku mierzy właściwą rzecz dla Raspberry Pi:** rozdziela `TTFT` od `generation_ms = E2E - TTFT`, więc długi prefill nie przykrywa szybkości dekodowania.
- **Dane są kompletne dla siatki testowej:** {total_rows} rekordów, {combos} kombinacji model/wariant i 20 pytań dla każdej kombinacji. Globalny odsetek trafień cache wynosi {fmt_pct(cache_hit_rate)}.
- **Warianty Q4 pokazują oczekiwany efekt kwantyzacji dla Gemm.** `gemma3-4b/Q4_K_M` ma medianę generacji {fmt_s(gemma3_q4km["median_generation_ms"])} wobec {fmt_s(gemma3_q8["median_generation_ms"])} dla `Q8_0`. `gemma4-e2b/Q4_K_M` ma {fmt_s(gemma4_q4km["median_generation_ms"])} wobec {fmt_s(gemma4_q8["median_generation_ms"])} dla `Q8_0`.
- **Q2 nie jest automatycznie najlepszym wyborem.** W danych widać długie odpowiedzi albo puste odpowiedzi, szczególnie dla `gemma4-e2b/Q2_K`, dlatego interpretacja Q2 wymaga kontroli długości odpowiedzi i jakości.

## Zbiór danych i metodologia

Źródło danych: `analysis/quant-warm-cache/warm_cache_generation_full_20260618_20260621.jsonl`.

Każdy rekord pochodzi z żądania do Ollamy w trybie streaming. `ttft_ms` jest mierzony do pierwszego niepustego tokenu, `e2e_ms` do końca odpowiedzi, a `generation_ms` to różnica między nimi. Test używał stałego promptu systemowego i `keep_alive=30m`, żeby utrzymać model oraz prefix cache w stanie ciepłym.

W raporcie używam median, bo odpowiedzi mają różną długość i w danych są długie ogony. Nie mamy liczby tokenów, więc `answer_chars` i `ms_per_char` są tylko przybliżeniem, a nie pełnym odpowiednikiem liczby tokenów generowanych na sekundę.

## Najlepszy wariant per model

{markdown_table(best_view, ["model", "quant_variant", "med_gen", "med_ttft", "przyspieszenie_q8"], {"quant_variant": "wariant", "med_gen": "mediana generacji", "med_ttft": "mediana TTFT", "przyspieszenie_q8": "przyspieszenie vs Q8"})}

![Mediana czasu generacji według wariantu](figures/01_mediana_generacji_wedlug_wariantu.png)

**Najważniejszy wykres to mediana generacji, nie E2E.** Po odjęciu TTFT widać, że kwantyzacja wpływa na właściwy koszt dekodowania. Dla Gemm warianty Q4 są wyraźnie lepsze od Q8, natomiast Q2 potrafi produkować dłuższe odpowiedzi i przez to nie zawsze daje najniższy czas końcowy.

## TTFT i ciepły cache

![Mediana TTFT według wariantu](figures/02_mediana_ttft_wedlug_wariantu.png)

**Ciepły cache stabilizuje TTFT, ale nie kasuje różnic między rodzinami modeli.** Gemmy mają medianę TTFT około kilku sekund, natomiast Bielik pozostaje wyraźnie wyżej. To potwierdza praktyczną konsekwencję dla systemu: jeżeli prompt będzie często zmieniany, benchmark wróci do mierzenia prefilla zamiast szybkości generacji.

![Dekompozycja E2E](figures/03_dekompozycja_e2e.png)

**E2E nadal jest sumą dwóch zjawisk.** Ten wykres pokazuje, dlaczego poprzednie pomiary mogły mylić: gdy TTFT/prefill dominuje, różnice między kwantyzacjami są łatwe do przykrycia. Przy ciepłym cache część generacyjna staje się głównym miejscem porównania.

## Przyspieszenie względem Q8

{markdown_table(q4_focus, ["model", "quant_variant", "generacja", "przyspieszenie"], {"quant_variant": "wariant"})}

![Przyspieszenie względem Q8](figures/04_przyspieszenie_wzgledem_q8.png)

**Porównanie do Q8 jest najbardziej czytelne dla tezy o kwantyzacji.** Wartość powyżej 1.0x oznacza, że wariant jest szybszy od Q8 w tej samej rodzinie modeli. To jest dokładnie sytuacja, której nie było dobrze widać w runach z zimnym lub zmiennym prefillem.

![Mediana czasu czystej generacji](figures/07_mediana_generacji_ms_wedlug_wariantu.png)

**Ten wykres pokazuje tę samą metrykę w jednostkach bezwzględnych.** Oś X przechodzi po wariantach kwantyzacji, a słupki porównują modele. Wartość to mediana `generation_ms`, czyli sam czas generacji po odjęciu TTFT/prefilla.

![Szybkość generacji po wyizolowaniu prefilla](figures/08_szybkosc_generacji_tok_s_wedlug_wariantu.png)

**Szybkość generacji jest liczona per rekord, a potem medianowana.** W danych nie ma pola `tokens_per_s`, dlatego skrypt używa przybliżenia `answer_chars / (generation_ms / 1000)`. Ten wykres należy traktować jako porównanie praktycznej szybkości odpowiedzi, a nie dokładny pomiar tokenizerowych tokenów na sekundę.

## Kontrola długości odpowiedzi

![Czas generacji względem długości odpowiedzi](figures/05_generacja_a_dlugosc_odpowiedzi.png)

**Część długich czasów generacji wynika z długości odpowiedzi.** To szczególnie ważne dla Q2: krótszy model w pamięci nie pomaga, jeśli wariant zaczyna odpowiadać znacznie bardziej rozwlekle albo zwraca puste odpowiedzi. Dlatego w pracy warto pokazać zarówno medianę `generation_ms`, jak i zastrzeżenie o braku liczby tokenów.

![Rozmiar modelu a opóźnienie](figures/06_rozmiar_gguf_a_generacja.png)

**Rozmiar GGUF zwykle pomaga, ale nie wystarcza jako jedyna metryka.** Mniejsze pliki zmniejszają presję na RAM i często poprawiają generację, lecz zachowanie zależy od rodziny modelu i stabilności odpowiedzi.

## Pełna tabela wyników

{markdown_table(summary_view, ["model", "quant_variant", "gguf_size_gb", "med_ttft", "med_gen", "med_e2e", "trafienia_cache", "znaki", "przyspieszenie_q8"], {"quant_variant": "wariant", "gguf_size_gb": "GGUF [GB]", "med_ttft": "mediana TTFT", "med_gen": "mediana generacji", "med_e2e": "mediana E2E", "trafienia_cache": "trafienia cache", "znaki": "mediana znaków", "przyspieszenie_q8": "przyspieszenie vs Q8"})}

## Jakość danych i anomalie

{markdown_table(quality_view, ["model", "quant_variant", "rows", "valid_generation_rows", "null_ttft_rows", "null_generation_rows", "empty_answer_rows", "trafienia_cache"], {"quant_variant": "wariant", "rows": "rekordy", "valid_generation_rows": "poprawne generacje", "null_ttft_rows": "brak TTFT", "null_generation_rows": "brak generation_ms", "empty_answer_rows": "puste odpowiedzi", "trafienia_cache": "trafienia cache"})}

W całym pliku jest {null_generation} rekordów bez `generation_ms`. Wszystkie takie rekordy należy traktować jako sygnał niestabilności wariantu, a nie jako szybkie odpowiedzi. Najważniejszy przypadek to `gemma4-e2b/Q2_K`, gdzie puste odpowiedzi zaniżają możliwość bezpośredniego porównania.

## Wniosek do pracy

Wyniki wspierają tezę, że na Raspberry Pi interpretacja benchmarków LLM musi rozdzielać koszt prefilla od kosztu generacji. Jeżeli prompt lub kontekst zmienia się między żądaniami, TTFT może zdominować E2E i ukryć wpływ kwantyzacji. Po utrzymaniu stałego promptu i ciepłego cache różnice w `generation_ms` stają się widoczne: warianty Q4 dla modeli Gemma są istotnie szybsze od Q8, a wybór wariantu powinien uwzględniać zarówno szybkość generacji, jak i stabilność oraz długość odpowiedzi.

## Artefakty

- `tables/podsumowanie_model_wariant.csv`
- `tables/przyspieszenie_wzgledem_q8.csv`
- `tables/jakosc_danych.csv`
- `tables/podsumowanie_pytan.csv`
- `figures/01_mediana_generacji_wedlug_wariantu.*`
- `figures/02_mediana_ttft_wedlug_wariantu.*`
- `figures/03_dekompozycja_e2e.*`
- `figures/04_przyspieszenie_wzgledem_q8.*`
- `figures/07_mediana_generacji_ms_wedlug_wariantu.*`
- `figures/08_szybkosc_generacji_tok_s_wedlug_wariantu.*`
- `figures/05_generacja_a_dlugosc_odpowiedzi.*`
- `figures/06_rozmiar_gguf_a_generacja.*`
"""
    return text


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    use_chart_theme()
    df = load_rows()
    summary, speedup, quality, question = build_tables(df)
    build_figures(df, summary, speedup)
    REPORT.write_text(report(summary, speedup, quality, df), encoding="utf-8")
    print(f"rekordy={len(df)}")
    print(f"kombinacje={summary.shape[0]}")
    print(f"raport={REPORT}")
    print(f"wykresy={FIG}")
    print(f"tabele={TABLE}")


if __name__ == "__main__":
    main()
