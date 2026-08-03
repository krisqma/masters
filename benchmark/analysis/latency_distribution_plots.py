#!/usr/bin/env python3
"""Wykresy diagnostyczne: czy latencja jest Gaussem?

Dla każdej jednorodnej grupy (eksperyment × model [× wariant] × metryka)
rysuje 3 panele z polskimi podpisami:
  A) surowe punkty (X=numer pytania, Y=czas)
  B) histogram: X=czas, Y=liczność + krzywe Gauss / log-normal / KDE
  C) mean±σ vs percentyle (test ujemnego dolnego zakresu)

Bez zmian w TeXu — tylko artefakty do obejrzenia.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

OUT = SCRIPT_ROOT / "stat-model-diagnostics" / "latency-dist"
GROUPS_DIR = OUT / "groups"
SHEETS_DIR = OUT / "sheets"
REPORT = OUT / "LATENCY_DIST_VERDICT.md"

THESIS_RAW = (
    SCRIPT_ROOT.parent.parent
    / "magisterka-text"
    / "Praca_Magisterska"
    / "data"
    / "5 - badania"
    / "surowe-wyniki"
)

SELECTED_DIRS = [
    SCRIPT_ROOT / "results-selected-9.05",
    THESIS_RAW / "selected-9.05",
]
WARM_SOURCES = [
    # n=100 (extend); pełna diagnostyka warm-n100 → run_warm_n100_diagnostics.py
    SCRIPT_ROOT.parent / "results" / "quant_latency_warm_n100.jsonl",
    # legacy n≈20
    SCRIPT_ROOT / "quant-warm-cache" / "warm_cache_generation_full_20260618_20260621.jsonl",
    THESIS_RAW / "warm_cache_generation_full_20260618_20260621.jsonl",
]
WARM_N20_LEGACY = (
    SCRIPT_ROOT / "quant-warm-cache" / "warm_cache_generation_full_20260618_20260621.jsonl"
)

# Nazwa robocza dla folderu results-selected-9.05 (pełny run Optuny, bez warm prefix-cache).
OPTUNA_FULL_LABEL = "Optuna Optimization Full (cold start)"
OPTUNA_FULL_SLUG = "optuna_optimization_full"

METRIC_LABELS = {
    "generation_ms": "generation_ms (czas generacji)",
    "ttft_ms": "TTFT (czas do pierwszego tokenu)",
    "e2e_ms": "E2E (czas end-to-end)",
}


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_jsonl_dir(directory: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_jsonl_file(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def sample_skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def group_stats(vals_s: np.ndarray) -> dict:
    """vals_s w sekundach."""
    vals = np.sort(vals_s.astype(float))
    n = len(vals)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(vals)),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "p95": float(np.quantile(vals, 0.95)),
        "skew": sample_skew(vals),
        "mean_minus_1sigma": mean - std,
        "mean_minus_2sigma": mean - 2 * std,
        "mean_plus_1sigma": mean + std,
        "mean_plus_2sigma": mean + 2 * std,
        "mean_over_median": mean / float(np.median(vals)) if np.median(vals) != 0 else float("nan"),
    }


def verdict_for(st: dict) -> str:
    if st["mean_minus_2sigma"] < 0:
        return "GAUSS_FAIL"
    if abs(st["skew"]) > 1.0 or (not math.isnan(st["mean_over_median"]) and st["mean_over_median"] > 1.15):
        return "GAUSS_FAIL"
    return "GAUSS_OK"


def safe_name(*parts: str) -> str:
    raw = "__".join(str(p) for p in parts if p is not None and str(p) != "")
    return (
        raw.replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def plot_group(
    vals_ms: np.ndarray,
    *,
    experiment: str,
    model: str,
    variant: str | None,
    metric: str,
    x_index: np.ndarray | None,
    out_path: Path,
) -> dict:
    vals_s = vals_ms.astype(float) / 1000.0
    st = group_stats(vals_s)
    verd = verdict_for(st)
    st["verdict"] = verd
    st["experiment"] = experiment
    st["model"] = model
    st["variant"] = variant or "—"
    st["metric"] = metric

    metric_pl = METRIC_LABELS.get(metric, metric)
    title_bits = [experiment, model]
    if variant:
        title_bits.append(variant)
    title_bits.append(f"{metric} (n={st['n']})")
    title = " | ".join(title_bits)

    if x_index is None:
        x_index = np.arange(1, len(vals_s) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # ----- Panel A: surowe punkty -----
    ax = axes[0]
    ax.scatter(x_index, vals_s, s=28, alpha=0.75, color="#1f77b4", zorder=3, label="pojedynczy request")
    ax.axhline(st["mean"], color="#d62728", lw=1.8, label=f"średnia = {st['mean']:.2f} s")
    ax.axhline(st["p50"], color="#2ca02c", lw=1.8, ls="--", label=f"p50 (mediana) = {st['p50']:.2f} s")
    ax.axhline(st["p90"], color="#ff7f0e", lw=1.5, ls=":", label=f"p90 = {st['p90']:.2f} s")
    ax.set_xlabel("Numer pytania / indeks próbki")
    ax.set_ylabel("Czas [s]")
    ax.set_title("A) Surowe pomiary")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.02,
        -0.18,
        "Każdy punkt = jeden request w tych samych warunkach",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )

    # ----- Panel B: histogram X=czas, Y=liczność -----
    ax = axes[1]
    n = st["n"]
    bins = max(5, min(12, n // 2)) if n >= 8 else max(3, n)
    counts, bin_edges, _ = ax.hist(
        vals_s,
        bins=bins,
        color="#aec7e8",
        edgecolor="#1f77b4",
        alpha=0.85,
        label="histogram (liczność)",
        zorder=1,
    )
    bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
    x_grid = np.linspace(max(vals_s.min() * 0.8, 0), vals_s.max() * 1.15, 300)

    # Gauss scaled to counts: density * n * bin_width
    if st["std"] > 0:
        gauss_pdf = stats.norm.pdf(x_grid, loc=st["mean"], scale=st["std"])
        ax.plot(x_grid, gauss_pdf * n * bin_width, color="#1f77b4", lw=2.2, label="Gauss N(średnia, σ²)", zorder=3)
    else:
        ax.axvline(st["mean"], color="#1f77b4", lw=2.2, label="Gauss (σ=0)")

    # Log-normal
    pos = vals_s[vals_s > 0]
    if len(pos) >= 3:
        shape, loc, scale = stats.lognorm.fit(pos, floc=0)
        logn_pdf = stats.lognorm.pdf(x_grid, shape, loc=loc, scale=scale)
        ax.plot(x_grid, logn_pdf * n * bin_width, color="#2ca02c", lw=2.0, label="Log-normal (fit)", zorder=3)

    # KDE
    if n >= 4 and st["std"] > 0:
        try:
            kde = stats.gaussian_kde(vals_s)
            ax.plot(x_grid, kde(x_grid) * n * bin_width, color="#7f7f7f", lw=1.6, ls="--", label="KDE (kształt danych)", zorder=2)
        except Exception:
            pass

    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Liczność w przedziale (bin)")
    ax.set_title("B) Histogram + krzywe modelu")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.02,
        -0.18,
        "Jeśli niebieska krzywa nie pokrywa słupków → Gauss słabo pasuje",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )

    # ----- Panel C: mean±σ vs percentyle -----
    ax = axes[2]
    ax.set_title("C) Mean±σ vs percentyle")
    ax.set_xlabel("Czas [s]")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["średnia ± 1σ", "średnia ± 2σ", "percentyle"])
    ax.set_ylim(-0.7, 2.7)

    # ±1σ
    lo1, hi1 = st["mean_minus_1sigma"], st["mean_plus_1sigma"]
    ax.plot([max(lo1, 0), hi1], [0, 0], color="#d62728", lw=6, solid_capstyle="butt", label="pasmo ±1σ")
    ax.plot(st["mean"], 0, "o", color="#d62728", ms=8)
    if lo1 < 0:
        ax.plot([lo1, 0], [0, 0], color="#d62728", lw=6, alpha=0.35, solid_capstyle="butt")
        ax.axvline(0, color="black", lw=0.8)

    # ±2σ
    lo2, hi2 = st["mean_minus_2sigma"], st["mean_plus_2sigma"]
    ax.plot([max(lo2, 0), hi2], [1, 1], color="#9467bd", lw=6, solid_capstyle="butt", label="pasmo ±2σ")
    ax.plot(st["mean"], 1, "o", color="#9467bd", ms=8)
    if lo2 < 0:
        ax.plot([lo2, 0], [1, 1], color="#9467bd", lw=6, alpha=0.35, solid_capstyle="butt")

    # percentyle
    ax.plot(st["p50"], 2, "s", color="#2ca02c", ms=9, label="p50")
    ax.plot(st["p90"], 2, "D", color="#ff7f0e", ms=8, label="p90")
    ax.plot(st["p95"], 2, "^", color="#8c564b", ms=8, label="p95")
    ax.plot([st["p50"], st["p95"]], [2, 2], color="#aaaaaa", lw=1.2, zorder=0)

    x_left = min(0.0, lo2, vals_s.min()) * 1.05 if lo2 < 0 else max(0.0, min(vals_s.min(), lo2) * 0.9)
    x_right = max(hi2, vals_s.max(), st["p95"]) * 1.08
    ax.set_xlim(x_left, x_right)
    ax.grid(True, axis="x", alpha=0.3)

    # Baner werdyktu
    if verd == "GAUSS_FAIL":
        reason = []
        if st["mean_minus_2sigma"] < 0:
            reason.append(f"średnia−2σ = {st['mean_minus_2sigma']:.2f} s < 0 (ujemny czas!)")
        if abs(st["skew"]) > 1.0:
            reason.append(f"silny skos = {st['skew']:.2f}")
        if not math.isnan(st["mean_over_median"]) and st["mean_over_median"] > 1.15:
            reason.append(f"mean/median = {st['mean_over_median']:.2f}")
        banner = "GAUSS_FAIL — " + "; ".join(reason)
        color = "#b00020"
    else:
        banner = (
            f"GAUSS_OK — średnia−2σ = {st['mean_minus_2sigma']:.2f} s ≥ 0, "
            f"skos = {st['skew']:.2f}"
        )
        color = "#1b5e20"

    ax.text(
        0.5,
        -0.22,
        banner,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        wrap=True,
    )

    # Legenda percentyli
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", markersize=8, label="p50"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff7f0e", markersize=7, label="p90"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#8c564b", markersize=7, label="p95"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.text(
        0.5,
        -0.02,
        f"Metryka: {metric_pl}   |   σ = {st['std']:.2f} s   |   mean/median = {st['mean_over_median']:.3f}",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return st


def plot_sheet_warm_gemma3(warm: pd.DataFrame, out_path: Path) -> None:
    """Arkusz: gemma3-4b × 4 warianty, generation_ms — histogramy obok siebie."""
    variants = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Arkusz zbiorczy: warm-cache | gemma3-4b | generation_ms\n"
        "X = czas [s], Y = liczność   |   niebieski = Gauss, zielony = log-normal",
        fontsize=12,
        fontweight="bold",
    )
    for ax, var in zip(axes.ravel(), variants):
        g = warm[(warm["model"] == "gemma3-4b") & (warm["quant_variant"] == var)]
        vals_s = g["generation_ms"].dropna().to_numpy(float) / 1000.0
        st = group_stats(vals_s)
        verd = verdict_for(st)
        n = st["n"]
        bins = max(5, min(10, n // 2))
        counts, edges, _ = ax.hist(vals_s, bins=bins, color="#aec7e8", edgecolor="#1f77b4", alpha=0.85)
        bw = edges[1] - edges[0]
        xg = np.linspace(0, vals_s.max() * 1.15, 200)
        if st["std"] > 0:
            ax.plot(xg, stats.norm.pdf(xg, st["mean"], st["std"]) * n * bw, color="#1f77b4", lw=2)
        pos = vals_s[vals_s > 0]
        if len(pos) >= 3:
            shape, loc, scale = stats.lognorm.fit(pos, floc=0)
            ax.plot(xg, stats.lognorm.pdf(xg, shape, loc=loc, scale=scale) * n * bw, color="#2ca02c", lw=1.8)
        ax.axvline(st["mean"], color="#d62728", lw=1.4, label="średnia")
        ax.axvline(st["p50"], color="#2ca02c", lw=1.4, ls="--", label="p50")
        flag = "FAIL" if verd == "GAUSS_FAIL" else "OK"
        ax.set_title(
            f"{var}  [{flag}]  mean−2σ={st['mean_minus_2sigma']:.2f}s  skew={st['skew']:.2f}",
            fontsize=10,
            color="#b00020" if verd == "GAUSS_FAIL" else "#1b5e20",
        )
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Liczność")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sheet_optuna_full_e2e(selected: pd.DataFrame, out_path: Path) -> None:
    models = sorted(selected["model"].dropna().unique())
    n_models = len(models)
    cols = 2
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"Arkusz zbiorczy: {OPTUNA_FULL_LABEL} | E2E\n"
        "Pełny run Optuny (folder results-selected-9.05) — bez warm prefix-cache\n"
        "X = czas [s], Y = liczność   |   niebieski = Gauss, zielony = log-normal",
        fontsize=12,
        fontweight="bold",
    )
    for ax, model in zip(axes_flat, models):
        vals_s = selected.loc[selected["model"] == model, "e2e_ms"].dropna().to_numpy(float) / 1000.0
        st = group_stats(vals_s)
        verd = verdict_for(st)
        n = st["n"]
        bins = 25 if n > 100 else max(8, n // 8)
        _, edges, _ = ax.hist(vals_s, bins=bins, color="#aec7e8", edgecolor="#1f77b4", alpha=0.85)
        bw = edges[1] - edges[0]
        xg = np.linspace(max(0, vals_s.min() * 0.9), vals_s.max() * 1.05, 300)
        if st["std"] > 0:
            ax.plot(xg, stats.norm.pdf(xg, st["mean"], st["std"]) * n * bw, color="#1f77b4", lw=2)
        pos = vals_s[vals_s > 0]
        if len(pos) >= 3:
            shape, loc, scale = stats.lognorm.fit(pos, floc=0)
            ax.plot(xg, stats.lognorm.pdf(xg, shape, loc=loc, scale=scale) * n * bw, color="#2ca02c", lw=1.6)
        ax.axvline(st["mean"], color="#d62728", lw=1.3)
        ax.axvline(st["p50"], color="#2ca02c", lw=1.3, ls="--")
        flag = "FAIL" if verd == "GAUSS_FAIL" else "OK"
        ax.set_title(
            f"{model}  [{flag}]  n={n}  mean−2σ={st['mean_minus_2sigma']:.2f}s  skew={st['skew']:.2f}",
            fontsize=10,
            color="#b00020" if verd == "GAUSS_FAIL" else "#1b5e20",
        )
        ax.set_xlabel("Czas E2E [s]")
        ax.set_ylabel("Liczność")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[n_models:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_verdict(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values(["experiment", "metric", "model", "variant"])
    df.to_csv(OUT / "latency_dist_summary.csv", index=False)

    lines = [
        "# Werdykt wizualny: czy latencja to Gauss?",
        "",
        "Wygenerowane przez `benchmark/analysis/latency_distribution_plots.py`.",
        "",
        "## Legenda osi",
        "",
        "- **Panel A:** X = numer pytania, Y = czas [s] (surowe punkty).",
        "- **Panel B:** X = czas [s], Y = liczność w binie (histogram) + krzywe Gauss / log-normal / KDE.",
        "- **Panel C:** porównanie pasm mean±σ z percentylami p50/p90/p95.",
        "",
        "## Reguła werdyktu",
        "",
        "- `GAUSS_FAIL` — gdy `średnia − 2σ < 0` **lub** `|skew| > 1` **lub** `mean/median > 1.15`.",
        "- `GAUSS_OK` — w przeciwnym razie (Gauss jako model roboczy do mean±σ).",
        "",
        "## Tabela",
        "",
        "| eksperyment | model | wariant | metryka | n | mean [s] | σ [s] | mean−2σ [s] | p50 | p90 | skew | mean/med | werdykt |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['experiment']} | `{r['model']}` | {r['variant']} | `{r['metric']}` | "
            f"{int(r['n'])} | {r['mean']:.2f} | {r['std']:.2f} | {r['mean_minus_2sigma']:.2f} | "
            f"{r['p50']:.2f} | {r['p90']:.2f} | {r['skew']:.2f} | {r['mean_over_median']:.3f} | "
            f"**{r['verdict']}** |"
        )

    n_ok = int((df["verdict"] == "GAUSS_OK").sum())
    n_fail = int((df["verdict"] == "GAUSS_FAIL").sum())
    lines += [
        "",
        f"**Podsumowanie:** GAUSS_OK = {n_ok}, GAUSS_FAIL = {n_fail} (z {len(df)} grup).",
        "",
        "## Gdzie patrzeć w pierwszej kolejności",
        "",
        f"- Arkusz warm generation (najczęściej FAIL): `{SHEETS_DIR / 'sheet_warm_gemma3_generation.png'}`",
        f"- Arkusz {OPTUNA_FULL_LABEL} E2E: `{SHEETS_DIR / f'sheet_{OPTUNA_FULL_SLUG}_e2e.png'}`",
        f"- Wszystkie grupy 3-panelowe: `{GROUPS_DIR}/`",
        "",
        "## Rekomendacja do pracy",
        "",
    ]
    fail_gen = df[(df["metric"] == "generation_ms") & (df["verdict"] == "GAUSS_FAIL")]
    optuna_e2e = df[(df["metric"] == "e2e_ms") & (df["experiment"] == OPTUNA_FULL_LABEL)]
    ok_e2e = optuna_e2e[optuna_e2e["verdict"] == "GAUSS_OK"]
    if len(fail_gen):
        lines.append(
            f"- Warm-cache `generation_ms`: **{len(fail_gen)}/{len(df[df.metric=='generation_ms'])}** grup = GAUSS_FAIL "
            "→ do raportowania lepiej **p50 / p90** (albo mean z zastrzeżeniem o ogonie), nie mean±2σ jako „95% pomiarów”."
        )
    if len(ok_e2e):
        lines.append(
            f"- {OPTUNA_FULL_LABEL} `e2e_ms`: **{len(ok_e2e)}/{len(optuna_e2e)}** "
            "modeli = GAUSS_OK → mean±σ jest obronne jako model roboczy."
        )
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    warm_path = first_existing(WARM_SOURCES)
    selected_dir = first_existing(SELECTED_DIRS)
    if warm_path is None:
        raise FileNotFoundError(WARM_SOURCES)
    if selected_dir is None:
        raise FileNotFoundError(SELECTED_DIRS)

    print(f"warm:     {warm_path}")
    print(f"selected: {selected_dir}")

    warm = load_jsonl_file(warm_path)
    selected = load_jsonl_dir(selected_dir)

    rows: list[dict] = []

    # Warm-cache groups
    for (model, variant), g in warm.groupby(["model", "quant_variant"], sort=True):
        for metric in ("generation_ms", "ttft_ms", "e2e_ms"):
            vals = g[metric].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            if "question_idx" in g.columns:
                x_idx = g.loc[g[metric].notna(), "question_idx"].to_numpy()
            else:
                x_idx = None
            path = GROUPS_DIR / f"warm__{safe_name(model, variant, metric)}.png"
            st = plot_group(
                vals,
                experiment="warm-cache",
                model=str(model),
                variant=str(variant),
                metric=metric,
                x_index=x_idx,
                out_path=path,
            )
            rows.append(st)
            print(f"  wrote {path.name}  [{st['verdict']}]")

    # Optuna Optimization Full (cold start) — folder results-selected-9.05
    for model, g in selected.groupby("model", sort=True):
        vals = g["e2e_ms"].dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        # Dla dużego n panel A: użyj indeksu 1..n (nie case_id — czytelniej)
        path = GROUPS_DIR / f"{OPTUNA_FULL_SLUG}__{safe_name(model, 'e2e_ms')}.png"
        st = plot_group(
            vals,
            experiment=OPTUNA_FULL_LABEL,
            model=str(model),
            variant=None,
            metric="e2e_ms",
            x_index=None,
            out_path=path,
        )
        rows.append(st)
        print(f"  wrote {path.name}  [{st['verdict']}]")

    plot_sheet_warm_gemma3(warm, SHEETS_DIR / "sheet_warm_gemma3_generation.png")
    print("  wrote sheet_warm_gemma3_generation.png")
    sheet_optuna = SHEETS_DIR / f"sheet_{OPTUNA_FULL_SLUG}_e2e.png"
    plot_sheet_optuna_full_e2e(selected, sheet_optuna)
    print(f"  wrote {sheet_optuna.name}")

    write_verdict(rows)
    print(f"Wrote verdict: {REPORT}")
    print(f"Groups: {GROUPS_DIR}")
    print(f"Sheets: {SHEETS_DIR}")


if __name__ == "__main__":
    main()
