#!/usr/bin/env python3
"""Diagnostyka Gaussa dla warm-cache n=100 (quant_latency_warm_n100.jsonl).

Pisze do stat-model-diagnostics/warm-n100/ — nie nadpisuje artefaktów n=20
w latency-dist/ ani głównego MODEL_VERDICT.md.

Uruchomienie:
  python3 run_warm_n100_diagnostics.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import latency_distribution_plots as ldp
import stat_model_diagnostics as smd

SCRIPT_ROOT = Path(__file__).resolve().parent
WARM_N100 = SCRIPT_ROOT.parent / "results" / "quant_latency_warm_n100.jsonl"
WARM_N20 = SCRIPT_ROOT / "quant-warm-cache" / "warm_cache_generation_full_20260618_20260621.jsonl"

OUT = SCRIPT_ROOT / "stat-model-diagnostics" / "warm-n100"
GROUPS_DIR = OUT / "latency-dist" / "groups"
SHEETS_DIR = OUT / "latency-dist" / "sheets"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
VERDICT = OUT / "WARM_N100_VERDICT.md"

MODELS = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
VARIANTS = ["Q2_K", "Q4_0", "Q4_K_M", "Q8_0"]
METRICS = ("generation_ms", "ttft_ms", "e2e_ms")


def clean_warm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "error" in df.columns:
        err = df["error"]
        bad = err.notna() & (err.astype(str).str.strip() != "") & (err.astype(str).str.lower() != "nan")
        df = df.loc[~bad].copy()
    return df


def plot_sheet_model_generation(warm: pd.DataFrame, model: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Arkusz zbiorczy: warm-cache n=100 | {model} | generation_ms\n"
        "X = czas [s], Y = liczność   |   niebieski = Gauss, zielony = log-normal",
        fontsize=12,
        fontweight="bold",
    )

    for ax, var in zip(axes.ravel(), VARIANTS):
        g = warm[(warm["model"] == model) & (warm["quant_variant"] == var)]
        vals_s = g["generation_ms"].dropna().to_numpy(float) / 1000.0
        if len(vals_s) == 0:
            ax.set_title(f"{var}  [brak danych]")
            ax.axis("off")
            continue
        st = ldp.group_stats(vals_s)
        verd = ldp.verdict_for(st)
        n = st["n"]
        bins = max(8, min(25, n // 4))
        _, edges, _ = ax.hist(vals_s, bins=bins, color="#aec7e8", edgecolor="#1f77b4", alpha=0.85)
        bw = edges[1] - edges[0]
        xg = np.linspace(0, vals_s.max() * 1.15, 200)
        if st["std"] > 0:
            ax.plot(xg, stats.norm.pdf(xg, st["mean"], st["std"]) * n * bw, color="#1f77b4", lw=2)
        pos = vals_s[vals_s > 0]
        if len(pos) >= 3:
            shape, loc, scale = stats.lognorm.fit(pos, floc=0)
            ax.plot(
                xg,
                stats.lognorm.pdf(xg, shape, loc=loc, scale=scale) * n * bw,
                color="#2ca02c",
                lw=1.8,
            )
        ax.axvline(st["mean"], color="#d62728", lw=1.4, label="średnia")
        ax.axvline(st["p50"], color="#2ca02c", lw=1.4, ls="--", label="p50")
        flag = "FAIL" if verd == "GAUSS_FAIL" else "OK"
        ax.set_title(
            f"{var}  [{flag}]  n={n}  mean−2σ={st['mean_minus_2sigma']:.2f}s  skew={st['skew']:.2f}",
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


def stats_row_from_vals(vals_ms: np.ndarray, *, experiment: str, model: str, variant: str, metric: str) -> dict:
    vals_s = vals_ms.astype(float) / 1000.0
    st = ldp.group_stats(vals_s)
    verd = ldp.verdict_for(st)
    return {
        "experiment": experiment,
        "model": model,
        "variant": variant,
        "metric": metric,
        "n": st["n"],
        "mean": st["mean"],
        "std": st["std"],
        "median": st["median"],
        "p50": st["p50"],
        "p90": st["p90"],
        "p95": st["p95"],
        "skew": st["skew"],
        "mean_minus_2sigma": st["mean_minus_2sigma"],
        "mean_over_median": st["mean_over_median"],
        "verdict": verd,
    }


def build_comparison(n20: pd.DataFrame, n100: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for model in MODELS:
            for variant in VARIANTS:
                a = n20[(n20["model"] == model) & (n20["quant_variant"] == variant)][metric].dropna().to_numpy(float)
                b = n100[(n100["model"] == model) & (n100["quant_variant"] == variant)][metric].dropna().to_numpy(float)
                if len(a) == 0 and len(b) == 0:
                    continue
                ra = stats_row_from_vals(a, experiment="warm-n20", model=model, variant=variant, metric=metric) if len(a) else None
                rb = stats_row_from_vals(b, experiment="warm-n100", model=model, variant=variant, metric=metric) if len(b) else None
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "metric": metric,
                        "n20": ra["n"] if ra else 0,
                        "n100": rb["n"] if rb else 0,
                        "skew_n20": ra["skew"] if ra else float("nan"),
                        "skew_n100": rb["skew"] if rb else float("nan"),
                        "mean_over_med_n20": ra["mean_over_median"] if ra else float("nan"),
                        "mean_over_med_n100": rb["mean_over_median"] if rb else float("nan"),
                        "mean_minus_2sigma_n20": ra["mean_minus_2sigma"] if ra else float("nan"),
                        "mean_minus_2sigma_n100": rb["mean_minus_2sigma"] if rb else float("nan"),
                        "verdict_n20": ra["verdict"] if ra else "—",
                        "verdict_n100": rb["verdict"] if rb else "—",
                        "verdict_changed": (
                            (ra["verdict"] != rb["verdict"]) if (ra and rb) else False
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_verdict_md(
    summary: pd.DataFrame,
    lat: pd.DataFrame,
    cmp: pd.DataFrame,
    warm_path: Path,
    n20_path: Path | None,
) -> None:
    gen = summary[summary["metric"] == "generation_ms"]
    ttft = summary[summary["metric"] == "ttft_ms"]
    e2e = summary[summary["metric"] == "e2e_ms"]

    def counts(df: pd.DataFrame) -> tuple[int, int]:
        ok = int((df["verdict"] == "GAUSS_OK").sum())
        fail = int((df["verdict"] == "GAUSS_FAIL").sum())
        return ok, fail

    g_ok, g_fail = counts(gen)
    t_ok, t_fail = counts(ttft)
    e_ok, e_fail = counts(e2e)

    cmp_gen = cmp[cmp["metric"] == "generation_ms"]
    flipped = cmp_gen[cmp_gen["verdict_changed"]]

    lines = [
        "# Warm-cache n=100 — werdykt Gaussa",
        "",
        "Wygenerowane przez `benchmark/analysis/run_warm_n100_diagnostics.py`.",
        "",
        f"- Dane n=100: `{warm_path}`",
        f"- Dane n=20 (porównanie): `{n20_path if n20_path else 'brak'}`",
        f"- Artefakty: `{OUT}`",
        "",
        "## Teza",
        "",
        "Przy n=20 warm `generation_ms` padał na GAUSS_FAIL; hipoteza: większe n (100) "
        "upodobni rozkład do Optuna E2E (GAUSS_OK).",
        "",
        "## Reguła werdyktu (bez zmian)",
        "",
        "- `GAUSS_FAIL` — gdy `średnia − 2σ < 0` **lub** `|skew| > 1` **lub** `mean/median > 1.15`.",
        "- `GAUSS_OK` — w przeciwnym razie.",
        "",
        "## Wynik n=100",
        "",
        f"| metryka | GAUSS_OK | GAUSS_FAIL |",
        f"|---|---:|---:|",
        f"| `generation_ms` | {g_ok} | {g_fail} |",
        f"| `ttft_ms` | {t_ok} | {t_fail} |",
        f"| `e2e_ms` | {e_ok} | {e_fail} |",
        "",
        "### generation_ms (kluczowe dla Tab. 5.7 / Rys. 5.7–5.10)",
        "",
        "| model | wariant | n | mean [s] | median [s] | σ [s] | mean−2σ | skew | mean/med | werdykt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in gen.sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| `{r['model']}` | {r['variant']} | {int(r['n'])} | {r['mean']:.2f} | {r['median']:.2f} | "
            f"{r['std']:.2f} | {r['mean_minus_2sigma']:.2f} | {r['skew']:.2f} | "
            f"{r['mean_over_median']:.3f} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "### TTFT",
        "",
        "| model | wariant | n | mean [s] | σ [s] | skew | mean/med | werdykt |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in ttft.sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| `{r['model']}` | {r['variant']} | {int(r['n'])} | {r['mean']:.2f} | "
            f"{r['std']:.2f} | {r['skew']:.2f} | {r['mean_over_median']:.3f} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## Porównanie n=20 vs n=100 (`generation_ms`)",
        "",
        "| model | wariant | n20→n100 | skew 20→100 | mean/med 20→100 | werdykt 20 | werdykt 100 | zmiana? |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for _, r in cmp_gen.sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| `{r['model']}` | {r['variant']} | {int(r['n20'])}→{int(r['n100'])} | "
            f"{r['skew_n20']:.2f}→{r['skew_n100']:.2f} | "
            f"{r['mean_over_med_n20']:.3f}→{r['mean_over_med_n100']:.3f} | "
            f"**{r['verdict_n20']}** | **{r['verdict_n100']}** | "
            f"{'TAK' if r['verdict_changed'] else 'nie'} |"
        )

    if len(flipped) == 0:
        flip_txt = (
            "Żadna grupa `generation_ms` **nie zmieniła** werdyktu FAIL→OK przy przejściu n=20→100. "
            "Teza „więcej case’ów naprawi Gaussa” **nie potwierdza się** dla warm generation."
        )
    else:
        flip_txt = (
            f"Zmiana werdyktu w {len(flipped)} grupach generation — szczegóły w tabeli powyżej."
        )

    # rekomendacje z diagnose_warm
    gen_lat = lat[lat["metric"] == "generation_ms"]
    ttft_lat = lat[lat["metric"] == "ttft_ms"]

    lines += [
        "",
        "## Wniosek",
        "",
        flip_txt,
        "",
        "- Warm `generation_ms` przy n=100: nadal **strukturalnie skośny** (ogon odpowiedzi / wariancja długości). "
        "Nie raportować mean±2σ jako „95% pomiarów”.",
        "- Warm `ttft_ms`: nadal zbliżony do Gaussa — **mean ± σ** obronne.",
        "- Optuna Optimization Full E2E pozostaje osobną historią (GAUSS_OK przy dużym n) — inny eksperyment / inna metryka.",
        "",
        "## Rekomendacja do TeXu (bez regeneracji teraz)",
        "",
        "| Miejsce | Estymator |",
        "|---|---|",
        "| Tab. 5.7 / Rys. 5.7–5.8–5.10 generation | **median + MAD** (lub p50/p90) |",
        "| Rys. 5.12 TTFT warm | **mean ± σ** |",
        "",
        "### Rekomendacje z `continuous_summary` (n=100)",
        "",
        "| model/wariant | generation family | location | scale |",
        "|---|---|---|---|",
    ]
    for _, r in gen_lat.sort_values(["model", "quant_variant"]).iterrows():
        lines.append(
            f"| `{r['model']}/{r['quant_variant']}` | {r['model_family']} | "
            f"{r['recommended_location']} | {r['recommended_scale']} |"
        )

    q2k = gen[gen["model"].eq("gemma4-e2b") & gen["variant"].eq("Q2_K")]
    if len(q2k) and int(q2k.iloc[0]["n"]) < 80:
        lines += [
            "",
            "## Uwaga o danych",
            "",
            f"- `gemma4-e2b/Q2_K` generation: n={int(q2k.iloc[0]['n'])} "
            "(mniej niż 100 ważnych `generation_ms` — część rekordów bez TTFT/gen).",
        ]

    lines += [
        "",
        "## Gdzie patrzeć",
        "",
        f"- Arkusze: `{SHEETS_DIR}/`",
        f"- Grupy 3-panel: `{GROUPS_DIR}/`",
        f"- QQ: `{FIG_DIR}/`",
        f"- CSV: `{TABLE_DIR}/warm_n100_latency.csv`, `{TABLE_DIR}/warm_n20_vs_n100.csv`",
        "",
    ]
    VERDICT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for d in (OUT, GROUPS_DIR, SHEETS_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not WARM_N100.exists():
        raise FileNotFoundError(WARM_N100)

    print(f"warm n=100: {WARM_N100}")
    warm = clean_warm(ldp.load_jsonl_file(WARM_N100))
    print(f"  rows after error filter: {len(warm)}")

    n20 = None
    n20_path = None
    if WARM_N20.exists():
        n20_path = WARM_N20
        n20 = clean_warm(ldp.load_jsonl_file(WARM_N20))
        print(f"warm n=20:  {WARM_N20} (rows={len(n20)})")
    else:
        print("WARN: brak legacy n=20 — porównanie będzie niepełne")

    # Redirect plot helpers' mental model: we pass explicit out paths
    rows: list[dict] = []
    for (model, variant), g in warm.groupby(["model", "quant_variant"], sort=True):
        for metric in METRICS:
            mask = g[metric].notna()
            vals = g.loc[mask, metric].to_numpy(float)
            if len(vals) == 0:
                continue
            if "question_idx" in g.columns:
                # unique index across repeats: use row order 1..n for panel A
                x_idx = np.arange(1, len(vals) + 1)
            else:
                x_idx = None
            path = GROUPS_DIR / f"warm_n100__{ldp.safe_name(model, variant, metric)}.png"
            st = ldp.plot_group(
                vals,
                experiment="warm-cache n=100",
                model=str(model),
                variant=str(variant),
                metric=metric,
                x_index=x_idx,
                out_path=path,
            )
            rows.append(st)
            print(f"  wrote {path.name}  [{st['verdict']}]")

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "warm_n100_gauss_summary.csv", index=False)

    for model in MODELS:
        out = SHEETS_DIR / f"sheet_warm_{model.replace('.', '_')}_generation.png"
        plot_sheet_model_generation(warm, model, out)
        print(f"  wrote {out.name}")

    # QQ + continuous_summary tables
    old_fig = smd.FIG
    smd.FIG = FIG_DIR
    try:
        smd.write_qq_warm(warm)
    finally:
        smd.FIG = old_fig
    print(f"  wrote QQ plots → {FIG_DIR}")

    lat = smd.diagnose_warm(warm)
    lat["dataset"] = "warm_n100"
    lat.to_csv(TABLE_DIR / "warm_n100_latency.csv", index=False)

    if n20 is not None:
        cmp = build_comparison(n20, warm)
    else:
        cmp = pd.DataFrame()
    cmp.to_csv(TABLE_DIR / "warm_n20_vs_n100.csv", index=False)

    write_verdict_md(summary, lat, cmp, WARM_N100, n20_path)
    print(f"Wrote {VERDICT}")

    gen = summary[summary["metric"] == "generation_ms"]
    print(
        f"DONE generation: OK={(gen['verdict']=='GAUSS_OK').sum()} "
        f"FAIL={(gen['verdict']=='GAUSS_FAIL').sum()} / {len(gen)}"
    )


if __name__ == "__main__":
    main()
