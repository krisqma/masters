#!/usr/bin/env python3
"""Draw cold-cache TTFT + generation decomposition charts."""

from __future__ import annotations

import argparse
import json
import os
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


ROOT = SCRIPT_ROOT / "quant-cold-cache"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
DEFAULT_RESULTS_DIR = SCRIPT_ROOT.parent / "results"

MODEL_ORDER = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
VARIANT_ORDER = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#262626",
    "muted": "#555555",
    "axis": "#CCCCCC",
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


def newest_source(results_dir: Path) -> Path:
    matches = sorted(results_dir.glob("cold_cache_generation_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"No cold_cache_generation_*.jsonl files found in {results_dir}")
    return matches[-1]


def load_rows(source: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with source.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    if not rows:
        raise SystemExit(f"Source file is empty: {source}")

    df = pd.DataFrame(rows)
    for col in ("gguf_size_gb", "ttft_ms", "e2e_ms", "generation_ms", "answer_chars", "question_idx"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"model", "quant_variant", "ttft_ms", "e2e_ms", "generation_ms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{source} is missing required columns: {', '.join(missing)}")

    df = df[df["model"].isin(MODEL_ORDER) & df["quant_variant"].isin(VARIANT_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["quant_variant"] = pd.Categorical(df["quant_variant"], categories=VARIANT_ORDER, ordered=True)
    df["valid_decomposition"] = df["ttft_ms"].notna() & df["generation_ms"].notna()
    return df.sort_values(["model", "quant_variant", "question_idx"]).reset_index(drop=True)


def q90(series: pd.Series) -> float:
    return series.dropna().quantile(0.90)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "quant_variant"], observed=True)
        .agg(
            rows=("case_id", "count"),
            valid_decomposition_rows=("valid_decomposition", "sum"),
            null_ttft_rows=("ttft_ms", lambda s: int(s.isna().sum())),
            null_generation_rows=("generation_ms", lambda s: int(s.isna().sum())),
            gguf_size_gb=("gguf_size_gb", "median"),
            median_ttft_ms=("ttft_ms", "median"),
            p90_ttft_ms=("ttft_ms", q90),
            median_generation_ms=("generation_ms", "median"),
            p90_generation_ms=("generation_ms", q90),
            median_e2e_ms=("e2e_ms", "median"),
            p90_e2e_ms=("e2e_ms", q90),
            median_answer_chars=("answer_chars", "median"),
        )
        .reset_index()
        .sort_values(["model", "quant_variant"])
    )
    summary["generation_share_of_e2e"] = summary["median_generation_ms"] / summary["median_e2e_ms"]
    TABLE.mkdir(parents=True, exist_ok=True)
    summary.rename(
        columns={
            "quant_variant": "wariant_kwantyzacji",
            "rows": "liczba_rekordow",
            "valid_decomposition_rows": "poprawna_dekompozycja",
            "null_ttft_rows": "brak_ttft",
            "null_generation_rows": "brak_generation_ms",
            "gguf_size_gb": "rozmiar_gguf_gb",
            "median_ttft_ms": "mediana_ttft_ms",
            "p90_ttft_ms": "p90_ttft_ms",
            "median_generation_ms": "mediana_generacji_ms",
            "p90_generation_ms": "p90_generacji_ms",
            "median_e2e_ms": "mediana_e2e_ms",
            "p90_e2e_ms": "p90_e2e_ms",
            "median_answer_chars": "mediana_znakow_odpowiedzi",
            "generation_share_of_e2e": "udzial_generacji_w_e2e",
        }
    ).to_csv(TABLE / "podsumowanie_model_wariant.csv", index=False)
    return summary


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_e2e_decomposition(summary: pd.DataFrame) -> None:
    plot = (
        summary.set_index(["model", "quant_variant"])
        .reindex(pd.MultiIndex.from_product([MODEL_ORDER, VARIANT_ORDER], names=["model", "quant_variant"]))
        .reset_index()
    )
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
        "Run z zimnym cache; słupki pokazują medianę w sekundach dla każdej kombinacji modelu i kwantyzacji.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    save_fig(fig, "03_dekompozycja_e2e_cold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cold-cache generation benchmark JSONL")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    source = args.source or newest_source(args.results_dir)
    use_chart_theme()
    df = load_rows(source)
    summary = build_summary(df)
    plot_e2e_decomposition(summary)

    complete = int(summary["valid_decomposition_rows"].sum())
    total = int(summary["rows"].sum())
    print(f"source={source}")
    print(f"rows={total} valid_decomposition={complete}")
    print(f"figure={FIG / '03_dekompozycja_e2e_cold.png'}")
    print(f"table={TABLE / 'podsumowanie_model_wariant.csv'}")


if __name__ == "__main__":
    main()
