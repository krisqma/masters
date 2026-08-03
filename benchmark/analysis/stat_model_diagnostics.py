#!/usr/bin/env python3
"""Diagnostyka modelu statystycznego dla rozdziału 5 pracy.

Cel (zgodnie z planem):
- zamknąć mapę figura/tabela → JSONL/CSV → skrypt produkujący liczby,
- policzyć mean / median / σ / MAD / skew / SE / pasma ±1σ i ±2σ,
- spisać werdykt modelowy (Gauss / binomial / mediana+MAD),
- ujednolicić język σ vs SE vs ±1.96·SE oraz bootstrap 5000 vs 10 000.

NIE regeneruje wykresów do TeXu — tylko artefakty diagnostyczne.
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
from scipy import stats

OUT = SCRIPT_ROOT / "stat-model-diagnostics"
FIG = OUT / "figures"
TABLE = OUT / "tables"
REPORT = OUT / "MODEL_VERDICT.md"

THESIS_RAW = (
    SCRIPT_ROOT.parent.parent
    / "magisterka-text"
    / "Praca_Magisterska"
    / "data"
    / "5 - badania"
    / "surowe-wyniki"
)

# Prefer live analysis copies; fall back to thesis copies.
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
THROTTLING_DIRS = [
    SCRIPT_ROOT / "results-throttling",
    THESIS_RAW / "throttling",
]
NOTHROTTLING_DIRS = [
    SCRIPT_ROOT / "results-5-trials-nothrottling",
    THESIS_RAW / "nothrottling",
]

# Bootstrap w thesis_statistical_analysis.bootstrap_ci — faktyczna wartość w kodzie.
BOOTSTRAP_N_IN_CODE = 5000
BOOTSTRAP_N_IN_TEX = 10_000  # 5-badania.tex przy Tab. 5.8 — rozbieżność do uzgodnienia

Z_1SIGMA = 1.0
Z_2SIGMA = 2.0
Z_95 = 1.96  # ≈ 95% przy N(0,1); pokrycie przy ±2σ ≈ 95.4%

MAD_TO_SIGMA = 1.4826  # spójność z N(μ,σ²): σ ≈ 1.4826 · MAD

# ---------------------------------------------------------------------------
# Mapa: artefakt w TeXu → dane → skrypt / funkcja → estymator dziś
# ---------------------------------------------------------------------------
INVENTORY = [
    {
        "artifact": "tab:ranking_run1 (Tab. 5.4)",
        "data": "results-throttling/*.jsonl",
        "producer": "thesis_statistical_analysis.model_summary",
        "estimator_today": "score=mean; e2e=median; temp=median w tekście",
    },
    {
        "artifact": "tab:ranking_run2 (Tab. 5.5)",
        "data": "results-5-trials-nothrottling/*.jsonl",
        "producer": "thesis_statistical_analysis.model_summary",
        "estimator_today": "score=mean; e2e=median",
    },
    {
        "artifact": "fig:score_vs_hall (Rys. 5.2)",
        "data": "results-selected-9.05/*.jsonl",
        "producer": "thesis_statistical_analysis.make_figures / trial_summary",
        "estimator_today": "punkt = mean score i hall_rate per trial Optuny",
    },
    {
        "artifact": "fig:regression_r2 (Rys. 5.3)",
        "data": "results-selected-9.05/*.jsonl",
        "producer": "thesis_statistical_analysis.regression_comparison",
        "estimator_today": "R² z OLS / linregress na agregatach trial",
    },
    {
        "artifact": "tab:kwantyzacja_wyniki (Tab. 5.7)",
        "data": "warm_cache_generation_full_*.jsonl",
        "producer": "warm_cache_generation_analysis.build_tables",
        "estimator_today": "TTFT/gen/E2E = median; speedup na medianach",
    },
    {
        "artifact": "fig:dekompozycja_e2e (Rys. 5.7)",
        "data": "warm_cache_generation_full_*.jsonl",
        "producer": "warm_cache_generation_analysis.plot_e2e_decomposition",
        "estimator_today": "słupki z median TTFT + median generation",
    },
    {
        "artifact": "fig:speedup_q8 (Rys. 5.8)",
        "data": "warm_cache_generation_full_*.jsonl",
        "producer": "warm_cache_generation_analysis.plot_speedup",
        "estimator_today": "speedup = median_gen(Q8) / median_gen(wariant)",
    },
    {
        "artifact": "fig:tok_s_variant (Rys. 5.10)",
        "data": "warm_cache_generation_full_*.jsonl",
        "producer": "warm_cache_generation_analysis (chars/s proxy, potem median)",
        "estimator_today": "median(answer_chars / (generation_ms/1000))",
    },
    {
        "artifact": "fig:ttft_warm (Rys. 5.12)",
        "data": "warm_cache_generation_full_*.jsonl",
        "producer": "warm_cache_generation_analysis.grouped_bar TTFT",
        "estimator_today": "median TTFT",
    },
    {
        "artifact": "tab:wyniki_finalne (Tab. 5.8)",
        "data": "results-selected-9.05 → selected_model_confidence_intervals.csv",
        "producer": "thesis_statistical_analysis.selected_model_ci / bootstrap_ci",
        "estimator_today": (
            f"score/hall = mean + bootstrap CI α=0.05 "
            f"(kod n_boot={BOOTSTRAP_N_IN_CODE}; TeX pisze {BOOTSTRAP_N_IN_TEX}); "
            "E2E = median bez CI"
        ),
    },
]


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_jsonl_dir(directory: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_file"] = path.name
                row["_line"] = line_no
                rows.append(row)
    return pd.DataFrame(rows)


def load_jsonl_file(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_file"] = path.name
            row["_line"] = line_no
            rows.append(row)
    return pd.DataFrame(rows)


def sample_skew(x: np.ndarray) -> float:
    x = x.astype(float)
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def mad_raw(x: np.ndarray) -> float:
    """MAD = median(|x - median(x)|) — bez skalowania do σ."""
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def continuous_summary(values: pd.Series, group_cols: dict) -> dict:
    vals = values.dropna().to_numpy(dtype=float)
    n = len(vals)
    out = {
        **group_cols,
        "n": n,
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "mad": np.nan,
        "mad_sigma_equiv": np.nan,
        "skew": np.nan,
        "mean_over_median": np.nan,
        "se_mean": np.nan,
        "mean_minus_1sigma": np.nan,
        "mean_plus_1sigma": np.nan,
        "mean_minus_2sigma": np.nan,
        "mean_plus_2sigma": np.nan,
        "mean_minus_1.96_se": np.nan,
        "mean_plus_1.96_se": np.nan,
        "shapiro_w": np.nan,
        "shapiro_p": np.nan,
        "log_skew": np.nan,
        "log_shapiro_p": np.nan,
        "recommended_location": "",
        "recommended_scale": "",
        "model_family": "",
    }
    if n == 0:
        return out

    mean = float(vals.mean())
    med = float(np.median(vals))
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    mad = mad_raw(vals)
    mad_sigma = mad * MAD_TO_SIGMA
    skew = sample_skew(vals)
    mean_over_med = mean / med if med != 0 else float("nan")
    se = std / math.sqrt(n) if n > 0 else float("nan")

    out.update(
        {
            "mean": mean,
            "median": med,
            "std": std,
            "mad": mad,
            "mad_sigma_equiv": mad_sigma,
            "skew": skew,
            "mean_over_median": mean_over_med,
            "se_mean": se,
            "mean_minus_1sigma": mean - Z_1SIGMA * std,
            "mean_plus_1sigma": mean + Z_1SIGMA * std,
            "mean_minus_2sigma": mean - Z_2SIGMA * std,
            "mean_plus_2sigma": mean + Z_2SIGMA * std,
            "mean_minus_1.96_se": mean - Z_95 * se,
            "mean_plus_1.96_se": mean + Z_95 * se,
        }
    )

    # Shapiro-Wilk tylko dla małych próbek (warm-cache n≈20); przy n>50 bywa zbyt czuły.
    if 3 <= n <= 50:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, p = stats.shapiro(vals)
        out["shapiro_w"] = float(w)
        out["shapiro_p"] = float(p)

    positive = vals[vals > 0]
    if len(positive) >= 3:
        logv = np.log(positive)
        out["log_skew"] = sample_skew(logv)
        if 3 <= len(logv) <= 50:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, lp = stats.shapiro(logv)
            out["log_shapiro_p"] = float(lp)

    # Reguła decyzyjna zgodna z planem.
    nearly_symmetric = abs(skew) < 0.5 and (math.isnan(mean_over_med) or abs(mean_over_med - 1.0) < 0.08)
    if nearly_symmetric:
        out["recommended_location"] = "mean"
        out["recommended_scale"] = "std (σ)"
        out["model_family"] = "gaussian_raw"
    else:
        out["recommended_location"] = "median"
        out["recommended_scale"] = "MAD (→ σ_equiv = 1.4826·MAD)"
        out["model_family"] = "skewed_robust"
        if not math.isnan(out["log_skew"]) and abs(out["log_skew"]) < abs(skew):
            out["model_family"] = "skewed_consider_lognormal"

    return out


def binomial_summary(flags: pd.Series, group_cols: dict) -> dict:
    vals = flags.dropna().astype(int).to_numpy()
    n = len(vals)
    k = int(vals.sum()) if n else 0
    p = k / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n and not math.isnan(p) else float("nan")
    return {
        **group_cols,
        "n": n,
        "k_hallucinations": k,
        "p_hat": p,
        "se_binomial": se,
        "p_minus_1sigma_se": p - Z_1SIGMA * se if n else float("nan"),
        "p_plus_1sigma_se": p + Z_1SIGMA * se if n else float("nan"),
        "p_minus_1.96_se": p - Z_95 * se if n else float("nan"),
        "p_plus_1.96_se": p + Z_95 * se if n else float("nan"),
        "model_family": "binomial",
        "recommended_location": "p_hat = k/n",
        "recommended_scale": "SE = sqrt(p(1-p)/n)",
    }


def bootstrap_mean_ci(values: pd.Series, n_boot: int = BOOTSTRAP_N_IN_CODE, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    """Kopia logiki thesis_statistical_analysis.bootstrap_ci — do porównania z ±1.96·SE."""
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def diagnose_selected(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["hallucination"] = df["hallucination_flag"].astype(int)

    lat_rows = []
    score_rows = []
    hall_rows = []

    for model, g in df.groupby("model", sort=True):
        base = {"dataset": "results-selected-9.05", "model": model, "metric": "e2e_ms"}
        lat_rows.append(continuous_summary(g["e2e_ms"], base))

        # TTFT bywa NaN przy timeoutach — diagnostyka tylko na obecnych wartościach.
        ttft = continuous_summary(g["ttft_ms"], {**base, "metric": "ttft_ms"})
        lat_rows.append(ttft)

        s = continuous_summary(g["composite_score"], {"dataset": "results-selected-9.05", "model": model, "metric": "composite_score"})
        # Score jest ograniczony [0,1] z atomem w 0 przy hall — CLT na średniej i tak działa przy dużym n.
        s["model_family"] = "bounded_mean_clt"
        s["recommended_location"] = "mean"
        s["recommended_scale"] = "SE = σ/√n (rozrzut populacyjny = σ)"
        lo, hi = bootstrap_mean_ci(g["composite_score"])
        s["bootstrap_ci_low"] = lo
        s["bootstrap_ci_high"] = hi
        s["bootstrap_n"] = BOOTSTRAP_N_IN_CODE
        score_rows.append(s)

        h = binomial_summary(g["hallucination"], {"dataset": "results-selected-9.05", "model": model, "metric": "hallucination"})
        blo, bhi = bootstrap_mean_ci(g["hallucination"])
        h["bootstrap_ci_low"] = blo
        h["bootstrap_ci_high"] = bhi
        h["bootstrap_n"] = BOOTSTRAP_N_IN_CODE
        hall_rows.append(h)

    return pd.DataFrame(lat_rows), pd.DataFrame(score_rows), pd.DataFrame(hall_rows)


def diagnose_warm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant), g in df.groupby(["model", "quant_variant"], sort=True):
        for metric in ("ttft_ms", "generation_ms", "e2e_ms"):
            rows.append(
                continuous_summary(
                    g[metric],
                    {
                        "dataset": "warm_cache",
                        "model": model,
                        "quant_variant": variant,
                        "metric": metric,
                    },
                )
            )
    return pd.DataFrame(rows)


def diagnose_run_latency(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model", sort=True):
        rows.append(continuous_summary(g["e2e_ms"], {"dataset": dataset, "model": model, "metric": "e2e_ms"}))
    return pd.DataFrame(rows)


def qq_plot(vals: np.ndarray, title: str, path: Path) -> None:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    stats.probplot(vals, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_qq_selected(df: pd.DataFrame) -> None:
    for model, g in df.groupby("model"):
        safe = str(model).replace(":", "_").replace("/", "_")
        qq_plot(g["e2e_ms"].dropna().to_numpy(float), f"QQ E2E — {model} (selected-9.05)", FIG / f"qq_selected_e2e_{safe}.png")
        qq_plot(
            g["composite_score"].dropna().to_numpy(float),
            f"QQ score — {model} (selected-9.05)",
            FIG / f"qq_selected_score_{safe}.png",
        )


def write_qq_warm(df: pd.DataFrame) -> None:
    for (model, variant), g in df.groupby(["model", "quant_variant"]):
        safe = f"{model}_{variant}".replace(":", "_").replace("/", "_")
        qq_plot(
            g["generation_ms"].dropna().to_numpy(float),
            f"QQ generation_ms — {model}/{variant}",
            FIG / f"qq_warm_gen_{safe}.png",
        )
        pos = g["generation_ms"].dropna()
        pos = pos[pos > 0]
        if len(pos) >= 3:
            qq_plot(
                np.log(pos.to_numpy(float)),
                f"QQ log(generation_ms) — {model}/{variant}",
                FIG / f"qq_warm_loggen_{safe}.png",
            )
        qq_plot(
            g["ttft_ms"].dropna().to_numpy(float),
            f"QQ TTFT — {model}/{variant}",
            FIG / f"qq_warm_ttft_{safe}.png",
        )


def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def write_verdict(
    inventory_df: pd.DataFrame,
    selected_lat: pd.DataFrame,
    selected_score: pd.DataFrame,
    selected_hall: pd.DataFrame,
    warm: pd.DataFrame,
    run1: pd.DataFrame | None,
    run2: pd.DataFrame | None,
) -> None:
    lines: list[str] = []
    lines.append("# Werdykt modelu statystycznego — rozdział 5")
    lines.append("")
    lines.append("Wygenerowane przez `benchmark/analysis/stat_model_diagnostics.py`.")
    lines.append("Bez regeneracji wykresów TeX — tylko diagnostyka i decyzja modelowa.")
    lines.append("")

    lines.append("## 1. Mapa figura/tabela → dane → skrypt")
    lines.append("")
    lines.append("| Artefakt | Dane | Producer | Estymator dziś |")
    lines.append("|---|---|---|---|")
    for row in INVENTORY:
        lines.append(
            f"| `{row['artifact']}` | `{row['data']}` | `{row['producer']}` | {row['estimator_today']} |"
        )
    lines.append("")

    lines.append("## 2. Język σ / SE / „95%” (ujednolicenie)")
    lines.append("")
    lines.append("| Pasmo | Pokrycie przy \(N(\\mu,\\sigma^2)\) | Co oznacza w pracy |")
    lines.append("|---|---|---|")
    lines.append("| ±1σ | ≈ 68,3% | rozrzut **pojedynczych** pomiarów wokół średniej |")
    lines.append("| ±2σ | ≈ 95,4% | szeroki pas rozrzutu pomiarów (nie CI średniej) |")
    lines.append("| ±3σ | ≈ 99,7% | prawie cały rozkład przy założeniu Gaussa |")
    lines.append("| ±1,96·SE | ≈ 95% dla **estymatora** średniej/proporcji | to jest odpowiednik obecnego „95% CI” |")
    lines.append("")
    lines.append("**Ważne rozróżnienie:**")
    lines.append("")
    lines.append("- **σ** = odchylenie standardowe próbki (rozrzut pojedynczych rekordów).")
    lines.append("- **SE = σ/√n** = błąd standardowy **średniej** (niepewność estymatora).")
    lines.append("- Obecne „95% CI” z bootstrapa ≈ percentyle 2,5%/97,5% rozkładu **średnich bootstrapowych** ≈ ±1,96·SE przy CLT.")
    lines.append("- To **nie** jest „3σ ≈ 96%”. Przy Gaussie ~95% to **~2σ**, a 3σ to ~99,7%.")
    lines.append("")
    lines.append("### Bootstrap: kod vs TeX")
    lines.append("")
    lines.append(f"- W kodzie (`thesis_statistical_analysis.bootstrap_ci`): **n_boot = {BOOTSTRAP_N_IN_CODE}**.")
    lines.append(f"- W TeXu przy Tab. 5.8: napisane **{BOOTSTRAP_N_IN_TEX}** prób.")
    lines.append(f"- **Decyzja metodyczna:** w tekście podawać **{BOOTSTRAP_N_IN_CODE}** (zgodne z kodem),")
    lines.append("  albo podnieść `n_boot` do 10 000 i przebudować CSV — jedna wartość, bez rozjazdu.")
    lines.append("  Na razie w diagnostyce porównujemy bootstrap przy n_boot = 5000 z pasmem ±1,96·SE.")
    lines.append("")

    lines.append("## 3. Run 3 (`selected-9.05`) — E2E / score / hall")
    lines.append("")
    lines.append("### 3.1 Latencja E2E")
    lines.append("")
    lines.append("| model | n | mean | median | σ | mean/med | skew | model_family | rekomendacja |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    e2e = selected_lat[selected_lat["metric"] == "e2e_ms"]
    for _, r in e2e.iterrows():
        lines.append(
            f"| `{r['model']}` | {int(r['n'])} | {fmt(r['mean'],1)} | {fmt(r['median'],1)} | "
            f"{fmt(r['std'],1)} | {fmt(r['mean_over_median'],3)} | {fmt(r['skew'],2)} | "
            f"{r['model_family']} | {r['recommended_location']} ± {r['recommended_scale']} |"
        )
    lines.append("")
    lines.append(
        "**Werdykt E2E Run 3:** rozkład prawie symetryczny (mean/med ≈ 0.97–0.98, |skew| małe). "
        "Mediana nie jest konieczna — przechodzimy na **mean ± σ** (rozrzut) oraz **mean ± 1,96·SE** (niepewność średniej)."
    )
    lines.append("")

    lines.append("### 3.2 Composite score")
    lines.append("")
    lines.append("| model | n | mean | σ | SE | ±1.96·SE | bootstrap CI (5000) |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for _, r in selected_score.iterrows():
        lines.append(
            f"| `{r['model']}` | {int(r['n'])} | {fmt(r['mean'],4)} | {fmt(r['std'],4)} | "
            f"{fmt(r['se_mean'],4)} | [{fmt(r['mean_minus_1.96_se'],4)}, {fmt(r['mean_plus_1.96_se'],4)}] | "
            f"[{fmt(r['bootstrap_ci_low'],4)}, {fmt(r['bootstrap_ci_high'],4)}] |"
        )
    lines.append("")
    lines.append(
        "**Werdykt score:** model **średniej z CLT** (bounded [0,1], atom w 0 przy hall). "
        "Raportować mean ± SE; „95%” = ±1,96·SE lub bootstrap mean (powinny być blisko)."
    )
    lines.append("")

    lines.append("### 3.3 Halucynacje (binomial)")
    lines.append("")
    lines.append("| model | n | k | p̂ | SE_bin | ±1.96·SE | bootstrap CI (5000) |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for _, r in selected_hall.iterrows():
        lines.append(
            f"| `{r['model']}` | {int(r['n'])} | {int(r['k_hallucinations'])} | {fmt(r['p_hat'],4)} | "
            f"{fmt(r['se_binomial'],4)} | [{fmt(r['p_minus_1.96_se'],4)}, {fmt(r['p_plus_1.96_se'],4)}] | "
            f"[{fmt(r['bootstrap_ci_low'],4)}, {fmt(r['bootstrap_ci_high'],4)}] |"
        )
    lines.append("")
    lines.append(
        "**Werdykt hall:** model **dwumianowy** \(X\\sim\\mathrm{Bin}(n,p)\). "
        "Dla gemma3:4b p jest małe (~1%) — SE binominalne i bootstrap zostają; "
        "nie mówić o „σ score’u” w kontekście odsetka halucynacji."
    )
    lines.append("")

    lines.append("### 3.4 Jak przepisać Tab. 5.8 (język metodyczny)")
    lines.append("")
    lines.append("Docelowy opis (bez wdrażania w TeX na tym etapie):")
    lines.append("")
    lines.append("> Wyniki Run 3: średni composite score ± 1,96·SE (CLT / równoważnie bootstrap średniej, ")
    lines.append(f"> n_boot={BOOTSTRAP_N_IN_CODE}); odsetek halucynacji z SE binominalnym ")
    lines.append("> \(\\mathrm{{SE}}=\\sqrt{{\\hat p(1-\\hat p)/n}}\); E2E jako średnia ± σ ")
    lines.append("> (rozkład prawie symetryczny; mediana ≈ średnia).")
    lines.append("")
    lines.append("Kolumna „E2E med.” → „E2E mean ± σ” (lub mean z SE, jeśli chodzi o niepewność średniej).")
    lines.append("Nie mieszać w jednym wierszu CI średniej score z medianą E2E bez etykiety.")
    lines.append("")

    lines.append("## 4. Warm-cache (kwantyzacja) — TTFT vs generation_ms")
    lines.append("")
    lines.append("### 4.1 TTFT")
    lines.append("")
    ttft = warm[warm["metric"] == "ttft_ms"].copy()
    lines.append("| model/wariant | n | mean | median | σ | mean/med | skew | family | rekomendacja |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for _, r in ttft.iterrows():
        lines.append(
            f"| `{r['model']}/{r['quant_variant']}` | {int(r['n'])} | {fmt(r['mean'],1)} | {fmt(r['median'],1)} | "
            f"{fmt(r['std'],1)} | {fmt(r['mean_over_median'],3)} | {fmt(r['skew'],2)} | "
            f"{r['model_family']} | {r['recommended_location']} |"
        )
    lines.append("")
    lines.append(
        "**Werdykt TTFT warm (Rys. 5.12):** niemal zawsze mean≈median, |skew| umiarkowany. "
        "Przejść na **mean ± σ**; mediana może zostać w tekście jako kontrola odporności, nie jako główny estymator."
    )
    lines.append("")

    lines.append("### 4.2 generation_ms")
    lines.append("")
    gen = warm[warm["metric"] == "generation_ms"].copy()
    lines.append("| model/wariant | n | mean | median | σ | MAD·1.4826 | mean/med | skew | log_skew | family | rekomendacja |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for _, r in gen.iterrows():
        lines.append(
            f"| `{r['model']}/{r['quant_variant']}` | {int(r['n'])} | {fmt(r['mean'],1)} | {fmt(r['median'],1)} | "
            f"{fmt(r['std'],1)} | {fmt(r['mad_sigma_equiv'],1)} | {fmt(r['mean_over_median'],3)} | "
            f"{fmt(r['skew'],2)} | {fmt(r['log_skew'],2)} | {r['model_family']} | "
            f"{r['recommended_location']} + {r['recommended_scale']} |"
        )
    lines.append("")
    lines.append(
        "**Werdykt generation_ms (Rys. 5.7/5.8/5.10, Tab. 5.7):** dla Gemm Q4/Q8 silny prawy ogon "
        "(mean/med często > 1.2, skew > 1). **Zostawić medianę + MAD** jako skalę odporną. "
        "Gdzie `log_skew` << `skew`, diagnostycznie rozważyć log-normal (QQ w `figures/qq_warm_loggen_*.png`) "
        "— bez zmiany PNG TeXu na tym etapie."
    )
    lines.append("")

    lines.append("## 5. Run 1 / Run 2 (throttling) — E2E")
    lines.append("")
    if run1 is not None and len(run1):
        lines.append("### Run 1 (throttling)")
        lines.append("")
        lines.append("| model | n | mean | median | σ | mean/med | skew | family |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for _, r in run1.iterrows():
            lines.append(
                f"| `{r['model']}` | {int(r['n'])} | {fmt(r['mean'],1)} | {fmt(r['median'],1)} | "
                f"{fmt(r['std'],1)} | {fmt(r['mean_over_median'],3)} | {fmt(r['skew'],2)} | {r['model_family']} |"
            )
        lines.append("")
    if run2 is not None and len(run2):
        lines.append("### Run 2 (no throttling)")
        lines.append("")
        lines.append("| model | n | mean | median | σ | mean/med | skew | family |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for _, r in run2.iterrows():
            lines.append(
                f"| `{r['model']}` | {int(r['n'])} | {fmt(r['mean'],1)} | {fmt(r['median'],1)} | "
                f"{fmt(r['std'],1)} | {fmt(r['mean_over_median'],3)} | {fmt(r['skew'],2)} | {r['model_family']} |"
            )
        lines.append("")
    lines.append(
        "Przy capie timeoutów / pustych odpowiedziach mediana bywa odporniejsza; "
        "po stabilizacji termicznej (Run 2/3) preferować mean±σ jeśli skew mały."
    )
    lines.append("")

    lines.append("## 6. Podsumowanie decyzji (co zostaje / co zmienić później w TeXu)")
    lines.append("")
    lines.append("| Miejsce | Estymator dziś | Decyzja modelowa |")
    lines.append("|---|---|---|")
    lines.append("| Tab. 5.8 score / hall | mean + „95% bootstrap” | mean / p̂ + **±1,96·SE** (σ języka: SE); bootstrap = weryfikacja; n_boot uzgodnić na **5000** |")
    lines.append("| Tab. 5.8 E2E | median | **mean ± σ** (symetria); nie CI mediany |")
    lines.append("| Rys. 5.12 TTFT warm | median | **mean ± σ** |")
    lines.append("| Tab. 5.7 / Rys. 5.7–5.8–5.10 generation | median | **median + MAD** (skośność); opcjonalnie log-normal diagnostycznie |")
    lines.append("| Rys. 5.2 / 5.3 | mean per trial / R² | bez zmian modelu; to agregaty jakości, nie latencja |")
    lines.append("| Tab. 5.4 / 5.5 E2E | median | po diagnostyce Run1/2: mean±σ jeśli symetryczne, inaczej median+MAD |")
    lines.append("")
    lines.append("## 7. Pliki wyjściowe")
    lines.append("")
    lines.append("- `tables/inventory.csv`")
    lines.append("- `tables/selected_latency.csv`")
    lines.append("- `tables/selected_score.csv`")
    lines.append("- `tables/selected_hallucination.csv`")
    lines.append("- `tables/warm_latency.csv`")
    lines.append("- `tables/run1_throttling_e2e.csv` / `tables/run2_nothrottling_e2e.csv` (jeśli źródła dostępne)")
    lines.append("- `figures/qq_*.png` — diagnostyka normalności (nie do TeXu)")
    lines.append("")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    inventory_df = pd.DataFrame(INVENTORY)
    inventory_df.to_csv(TABLE / "inventory.csv", index=False)

    selected_dir = first_existing(SELECTED_DIRS)
    warm_path = first_existing(WARM_SOURCES)
    if selected_dir is None:
        raise FileNotFoundError(f"Brak selected-9.05 w {SELECTED_DIRS}")
    if warm_path is None:
        raise FileNotFoundError(f"Brak warm-cache JSONL w {WARM_SOURCES}")

    print(f"selected: {selected_dir}")
    print(f"warm:     {warm_path}")

    selected = load_jsonl_dir(selected_dir)
    warm = load_jsonl_file(warm_path)

    selected_lat, selected_score, selected_hall = diagnose_selected(selected)
    warm_lat = diagnose_warm(warm)

    selected_lat.to_csv(TABLE / "selected_latency.csv", index=False)
    selected_score.to_csv(TABLE / "selected_score.csv", index=False)
    selected_hall.to_csv(TABLE / "selected_hallucination.csv", index=False)
    warm_lat.to_csv(TABLE / "warm_latency.csv", index=False)

    run1_df = run2_df = None
    th_dir = first_existing(THROTTLING_DIRS)
    nth_dir = first_existing(NOTHROTTLING_DIRS)
    if th_dir is not None:
        print(f"run1:     {th_dir}")
        run1_df = diagnose_run_latency(load_jsonl_dir(th_dir), "results-throttling")
        run1_df.to_csv(TABLE / "run1_throttling_e2e.csv", index=False)
    if nth_dir is not None:
        print(f"run2:     {nth_dir}")
        run2_df = diagnose_run_latency(load_jsonl_dir(nth_dir), "results-5-trials-nothrottling")
        run2_df.to_csv(TABLE / "run2_nothrottling_e2e.csv", index=False)

    write_qq_selected(selected)
    write_qq_warm(warm)

    write_verdict(
        inventory_df,
        selected_lat,
        selected_score,
        selected_hall,
        warm_lat,
        run1_df,
        run2_df,
    )

    print(f"Wrote report: {REPORT}")
    print(f"Tables:       {TABLE}")
    print(f"QQ figures:   {FIG}")


if __name__ == "__main__":
    main()
