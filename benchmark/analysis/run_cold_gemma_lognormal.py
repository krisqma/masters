#!/usr/bin/env python3
"""Cold-cache log-normal diagnostyka dla Gemm (bez Bielika).

Używa aktualnego quant_latency_cold_n100.jsonl — pełne komórki dostają fit + wykresy,
niepełne (n_valid < 30) dostają placeholdery z n, żeby widać układ i tendencję.

  python3 run_cold_gemma_lognormal.py
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

COLD_PATH = SCRIPT_ROOT.parent / "results" / "quant_latency_cold_n100.jsonl"
OUT = SCRIPT_ROOT / "stat-model-diagnostics" / "cold-gemma-lognormal"
GROUPS_DIR = OUT / "latency-dist" / "groups"
SHEETS_DIR = OUT / "latency-dist" / "sheets"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
VERDICT = OUT / "COLD_GEMMA_LOGNORMAL_VERDICT.md"

MODELS = ["gemma3-4b", "gemma4-e2b"]
VARIANTS = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]
METRICS = ("generation_ms", "ttft_ms", "e2e_ms")
MIN_N_FULL = 30
TARGET_N = 100


def load_cold(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def valid_vals(g: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in g.columns:
        return np.array([], dtype=float)
    s = g[metric]
    mask = s.notna()
    if "error" in g.columns:
        err = g["error"]
        bad = err.notna() & (err.astype(str).str.strip() != "") & (err.astype(str).str.lower() != "nan")
        mask = mask & ~bad
    vals = s.loc[mask].to_numpy(dtype=float)
    return vals[vals > 0]


def fit_normal_seconds(vals_ms: np.ndarray) -> dict:
    x = vals_ms / 1000.0
    n = len(x)
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1)) if n > 1 else 0.0
    # AIC for Normal: 2k - 2lnL, k=2
    if sig > 0 and n >= 2:
        ll = float(np.sum(stats.norm.logpdf(x, mu, sig)))
        aic = 2 * 2 - 2 * ll
    else:
        ll, aic = float("nan"), float("nan")
    return {
        "n": n,
        "mean_s": mu,
        "std_s": sig,
        "median_s": float(np.median(x)),
        "lo_2sig_s": mu - 2 * sig,
        "hi_2sig_s": mu + 2 * sig,
        "aic_norm": aic,
        "ll_norm": ll,
    }


def fit_lognormal_seconds(vals_ms: np.ndarray) -> dict:
    x = vals_ms / 1000.0
    n = len(x)
    logx = np.log(x)
    mu = float(np.mean(logx))
    sig = float(np.std(logx, ddof=1)) if n > 1 else 0.0
    med = float(math.exp(mu))
    lo = float(math.exp(mu - 2 * sig)) if sig >= 0 else float("nan")
    hi = float(math.exp(mu + 2 * sig)) if sig >= 0 else float("nan")
    # Lognormal with floc=0: shape=sig, scale=exp(mu); scipy lognorm.pdf(x, s, scale=exp(mu))
    if sig > 0 and n >= 2:
        ll = float(np.sum(stats.lognorm.logpdf(x, s=sig, scale=math.exp(mu))))
        aic = 2 * 2 - 2 * ll
    else:
        ll, aic = float("nan"), float("nan")
    return {
        "n": n,
        "mu_log": mu,
        "sigma_log": sig,
        "model_median_s": med,
        "lo_exp_mu_minus_2sig_s": lo,
        "hi_exp_mu_plus_2sig_s": hi,
        "aic_lognorm": aic,
        "ll_lognorm": ll,
    }


def cell_status(n_total: int, n_valid: int) -> str:
    if n_valid <= 0:
        return "empty"
    if n_valid >= MIN_N_FULL and n_total >= TARGET_N and n_valid >= int(0.9 * TARGET_N):
        return "ok"
    if n_valid >= MIN_N_FULL:
        return "partial"
    return "partial" if n_valid > 0 else "empty"


def preferred_model(aic_norm: float, aic_lognorm: float) -> str:
    if math.isnan(aic_norm) or math.isnan(aic_lognorm):
        return "—"
    if aic_lognorm < aic_norm - 2:
        return "lognormal"
    if aic_norm < aic_lognorm - 2:
        return "normal"
    return "tie"


def plot_placeholder(ax, title: str, note: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color="#666666")
    ax.text(0.5, 0.55, note, ha="center", va="center", fontsize=12, color="#888888", wrap=True)
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, ls="--", ec="#bbbbbb", lw=1.5))


def plot_hist_panel(ax, vals_ms: np.ndarray, title: str) -> dict:
    x = vals_ms / 1000.0
    nfit = fit_normal_seconds(vals_ms)
    lfit = fit_lognormal_seconds(vals_ms)
    n = len(x)
    bins = max(8, min(25, n // 4))
    _, edges, _ = ax.hist(x, bins=bins, color="#aec7e8", edgecolor="#1f77b4", alpha=0.85)
    bw = edges[1] - edges[0]
    xg = np.linspace(max(1e-6, x.min() * 0.85), x.max() * 1.12, 300)
    if nfit["std_s"] > 0:
        ax.plot(xg, stats.norm.pdf(xg, nfit["mean_s"], nfit["std_s"]) * n * bw, color="#1f77b4", lw=2, label="Normal")
    if lfit["sigma_log"] > 0:
        ax.plot(
            xg,
            stats.lognorm.pdf(xg, s=lfit["sigma_log"], scale=math.exp(lfit["mu_log"])) * n * bw,
            color="#2ca02c",
            lw=2,
            label="LogNormal",
        )
    ax.axvline(nfit["mean_s"], color="#d62728", lw=1.2, ls="-", alpha=0.8)
    ax.axvline(lfit["model_median_s"], color="#2ca02c", lw=1.2, ls="--", alpha=0.9)
    pref = preferred_model(nfit["aic_norm"], lfit["aic_lognorm"])
    ax.set_title(
        f"{title}\nn={n}  prefer={pref}  AIC_n={nfit['aic_norm']:.0f}  AIC_ln={lfit['aic_lognorm']:.0f}",
        fontsize=9,
        color="#1b5e20" if pref == "lognormal" else ("#b00020" if pref == "normal" else "#333333"),
    )
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Liczność")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return {"norm": nfit, "lognorm": lfit, "pref": pref}


def plot_sheet_generation(df: pd.DataFrame, model: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Cold-cache | {model} | generation_ms | LogNormal vs Normal\n"
        "zielony = LogNormal, niebieski = Normal  |  partial → placeholder",
        fontsize=12,
        fontweight="bold",
    )
    for ax, var in zip(axes.ravel(), VARIANTS):
        g = df[(df["model"] == model) & (df["quant_variant"] == var)]
        vals = valid_vals(g, "generation_ms")
        n_total = len(g)
        if len(vals) < MIN_N_FULL:
            plot_placeholder(
                ax,
                f"{var}",
                f"partial / brak\nn_valid={len(vals)} / n_rows={n_total}\n(próg pełnego wykresu: {MIN_N_FULL})",
            )
        else:
            plot_hist_panel(ax, vals, var)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_group_or_placeholder(
    vals_ms: np.ndarray,
    *,
    model: str,
    variant: str,
    metric: str,
    n_total: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(vals_ms) < MIN_N_FULL:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_placeholder(
            ax,
            f"cold | {model} | {variant} | {metric}",
            f"PLACEHOLDER — za mało danych\nn_valid={len(vals_ms)} / n_rows={n_total}\n"
            f"Tendencja: dograj do n≥{MIN_N_FULL} (cel {TARGET_N})",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    x = vals_ms / 1000.0
    nfit = fit_normal_seconds(vals_ms)
    lfit = fit_lognormal_seconds(vals_ms)
    pref = preferred_model(nfit["aic_norm"], lfit["aic_lognorm"])
    n = len(x)
    idx = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.suptitle(
        f"cold-cache | {model} | {variant} | {metric} (n={n}) | prefer={pref}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    ax = axes[0]
    ax.scatter(idx, x, s=22, alpha=0.75, color="#1f77b4")
    ax.axhline(nfit["mean_s"], color="#d62728", lw=1.5, label=f"mean={nfit['mean_s']:.2f}s")
    ax.axhline(lfit["model_median_s"], color="#2ca02c", lw=1.5, ls="--", label=f"e^μ={lfit['model_median_s']:.2f}s")
    ax.set_xlabel("Indeks próbki")
    ax.set_ylabel("Czas [s]")
    ax.set_title("A) Surowe punkty")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    plot_hist_panel(ax, vals_ms, "B) Histogram")
    ax.set_title("B) Histogram + Normal / LogNormal")

    ax = axes[2]
    ax.set_title("C) Pasma modelu")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Normal ±2σ", "LogNormal e^{μ±2σ}", "empir. p50/p90"])
    ax.set_ylim(-0.6, 2.6)
    # Normal band
    lo_n, hi_n = nfit["lo_2sig_s"], nfit["hi_2sig_s"]
    ax.plot([max(lo_n, 0), hi_n], [0, 0], color="#1f77b4", lw=7, solid_capstyle="butt")
    if lo_n < 0:
        ax.plot([lo_n, 0], [0, 0], color="#1f77b4", lw=7, alpha=0.35)
        ax.axvline(0, color="black", lw=0.8)
    ax.plot(nfit["mean_s"], 0, "o", color="#1f77b4", ms=8)
    # Lognormal band
    lo_l, hi_l = lfit["lo_exp_mu_minus_2sig_s"], lfit["hi_exp_mu_plus_2sig_s"]
    ax.plot([lo_l, hi_l], [1, 1], color="#2ca02c", lw=7, solid_capstyle="butt")
    ax.plot(lfit["model_median_s"], 1, "o", color="#2ca02c", ms=8)
    # empiryczne
    p50, p90 = float(np.quantile(x, 0.5)), float(np.quantile(x, 0.9))
    ax.plot(p50, 2, "s", color="#ff7f0e", ms=8)
    ax.plot(p90, 2, "D", color="#9467bd", ms=8)
    ax.plot([p50, p90], [2, 2], color="#aaaaaa", lw=1)
    xmin = min(0.0, lo_n, lo_l, x.min()) * 1.05 if lo_n < 0 else max(0.0, min(lo_l, x.min()) * 0.9)
    xmax = max(hi_n, hi_l, x.max(), p90) * 1.08
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Czas [s]")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", lw=4, label="Normal ±2σ"),
            Line2D([0], [0], color="#2ca02c", lw=4, label="LogNormal e^{μ±2σ}"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=8, label="p50"),
        ],
        fontsize=8,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def qq_log(vals_ms: np.ndarray, title: str, path: Path) -> None:
    x = vals_ms / 1000.0
    if len(x) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    stats.probplot(np.log(x), dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def safe_name(*parts: str) -> str:
    return "__".join(str(p) for p in parts).replace(".", "_").replace("/", "_").replace(":", "_")


def write_verdict(fit_df: pd.DataFrame, status_df: pd.DataFrame, cold_path: Path) -> None:
    gen = fit_df[fit_df["metric"] == "generation_ms"].copy()
    pref_counts = gen["preferred"].value_counts().to_dict()
    lines = [
        "# Cold Gemmy — model LogNormal (na dostępnych danych)",
        "",
        f"Źródło: `{cold_path}`",
        f"Artefakty: `{OUT}`",
        "",
        "Bielik pominięty. Placeholdery gdy `n_valid < 30`.",
        "",
        "## Status komórek",
        "",
        "| model | wariant | metryka | n_rows | n_valid | status |",
        "|---|---|---|---:|---:|---|",
    ]
    for _, r in status_df.sort_values(["model", "variant", "metric"]).iterrows():
        lines.append(
            f"| `{r['model']}` | {r['variant']} | `{r['metric']}` | "
            f"{int(r['n_total'])} | {int(r['n_valid'])} | {r['status']} |"
        )

    lines += [
        "",
        "## Tendencja: Normal vs LogNormal (`generation_ms`, n_valid≥30)",
        "",
        f"Preferencje AIC: {pref_counts}",
        "",
        "| model | wariant | n | μ_log | σ_log | e^μ [s] | e^{μ±2σ} [s] | AIC_N | AIC_LN | prefer |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for _, r in gen[gen["n"] >= MIN_N_FULL].sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| `{r['model']}` | {r['variant']} | {int(r['n'])} | {r['mu_log']:.3f} | {r['sigma_log']:.3f} | "
            f"{r['model_median_s']:.2f} | [{r['lo_exp_mu_minus_2sig_s']:.2f}, {r['hi_exp_mu_plus_2sig_s']:.2f}] | "
            f"{r['aic_norm']:.0f} | {r['aic_lognorm']:.0f} | **{r['preferred']}** |"
        )

    n_ln = int((gen["preferred"] == "lognormal").sum())
    n_n = int((gen["preferred"] == "normal").sum())
    n_tie = int((gen["preferred"] == "tie").sum())
    n_fit = int((gen["n"] >= MIN_N_FULL).sum())

    lines += [
        "",
        "## Wniosek (tendencja)",
        "",
        f"Spośród {n_fit} komórek generation z pełnym fittem: "
        f"LogNormal wygrywa AIC w **{n_ln}**, Normal w **{n_n}**, remis **{n_tie}**.",
        "",
        "- Cold generation Gemm: trzymaj **LogNormal** jako model roboczy (pas \(e^{\\hat\\mu\\pm 2\\hat\\sigma}\)).",
        "- Normal na sekundach często daje ujemne \(\\bar x-2s\) albo gorszy AIC.",
        "- Gemma4 Q8_0 / częściowe komórki: placeholdery — odpal skrypt ponownie po dograniu.",
        "",
        "## Gdzie patrzeć",
        "",
        f"- Arkusze: `{SHEETS_DIR}/`",
        f"- Grupy: `{GROUPS_DIR}/`",
        f"- QQ log: `{FIG_DIR}/`",
        f"- CSV: `{TABLE_DIR}/`",
        "",
    ]
    VERDICT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for d in (OUT, GROUPS_DIR, SHEETS_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not COLD_PATH.exists():
        raise FileNotFoundError(COLD_PATH)

    print(f"cold: {COLD_PATH}")
    df = load_cold(COLD_PATH)
    df = df[df["model"].isin(MODELS)].copy()
    print(f"rows (gemmas only): {len(df)}")

    fit_rows: list[dict] = []
    status_rows: list[dict] = []

    for model in MODELS:
        for variant in VARIANTS:
            g = df[(df["model"] == model) & (df["quant_variant"] == variant)]
            n_total = len(g)
            for metric in METRICS:
                vals = valid_vals(g, metric)
                n_valid = len(vals)
                st = cell_status(n_total, n_valid)
                status_rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "metric": metric,
                        "n_total": n_total,
                        "n_valid": n_valid,
                        "status": st,
                    }
                )

                if n_valid:
                    nfit = fit_normal_seconds(vals)
                    lfit = fit_lognormal_seconds(vals)
                else:
                    nfit = {
                        "mean_s": float("nan"),
                        "std_s": float("nan"),
                        "lo_2sig_s": float("nan"),
                        "hi_2sig_s": float("nan"),
                        "aic_norm": float("nan"),
                    }
                    lfit = {
                        "mu_log": float("nan"),
                        "sigma_log": float("nan"),
                        "model_median_s": float("nan"),
                        "lo_exp_mu_minus_2sig_s": float("nan"),
                        "hi_exp_mu_plus_2sig_s": float("nan"),
                        "aic_lognorm": float("nan"),
                    }
                pref = preferred_model(nfit.get("aic_norm", float("nan")), lfit.get("aic_lognorm", float("nan")))
                fit_rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "metric": metric,
                        "n": n_valid,
                        "n_total": n_total,
                        "status": st,
                        "mean_s": nfit.get("mean_s"),
                        "std_s": nfit.get("std_s"),
                        "lo_2sig_s": nfit.get("lo_2sig_s"),
                        "hi_2sig_s": nfit.get("hi_2sig_s"),
                        "aic_norm": nfit.get("aic_norm"),
                        "mu_log": lfit.get("mu_log"),
                        "sigma_log": lfit.get("sigma_log"),
                        "model_median_s": lfit.get("model_median_s"),
                        "lo_exp_mu_minus_2sig_s": lfit.get("lo_exp_mu_minus_2sig_s"),
                        "hi_exp_mu_plus_2sig_s": lfit.get("hi_exp_mu_plus_2sig_s"),
                        "aic_lognorm": lfit.get("aic_lognorm"),
                        "preferred": pref,
                    }
                )

                path = GROUPS_DIR / f"cold__{safe_name(model, variant, metric)}.png"
                plot_group_or_placeholder(
                    vals, model=model, variant=variant, metric=metric, n_total=n_total, out_path=path
                )
                print(f"  {path.name}  n_valid={n_valid} status={st} pref={pref}")

                if n_valid >= MIN_N_FULL:
                    qq_log(
                        vals,
                        f"QQ log({metric}) — cold {model}/{variant}",
                        FIG_DIR / f"qq_log_{safe_name(model, variant, metric)}.png",
                    )

    fit_df = pd.DataFrame(fit_rows)
    status_df = pd.DataFrame(status_rows)
    fit_df.to_csv(TABLE_DIR / "cold_gemma_lognormal.csv", index=False)
    status_df.to_csv(TABLE_DIR / "cold_gemma_status.csv", index=False)

    for model in MODELS:
        out = SHEETS_DIR / f"sheet_cold_{safe_name(model)}_generation.png"
        plot_sheet_generation(df, model, out)
        print(f"  wrote {out.name}")

    write_verdict(fit_df, status_df, COLD_PATH)
    print(f"Wrote {VERDICT}")

    gen = fit_df[(fit_df["metric"] == "generation_ms") & (fit_df["n"] >= MIN_N_FULL)]
    print(
        "DONE generation prefs:",
        gen["preferred"].value_counts().to_dict(),
        f"fitted={len(gen)}",
    )


if __name__ == "__main__":
    main()
