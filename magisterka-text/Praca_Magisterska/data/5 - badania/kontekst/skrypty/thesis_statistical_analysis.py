#!/usr/bin/env python3
"""Statistical analysis for thesis-ready benchmark reporting.

Outputs are written to thesis-analysis-output:
- CSV tables with model, trial, case, decomposition, regression and Pareto data.
- Markdown report with numbers that can be adapted into a thesis chapter.
- Static figures in PNG, SVG and PDF.

The analysis focuses on the hypothesis that Optuna-selected hyperparameters
have limited practical impact on benchmark quality, while the composite score
is dominated by hallucination flags and latency is the main optimization axis.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats


ROOT = SCRIPT_ROOT
OUT = ROOT / "thesis-analysis-output"
FIG = OUT / "figures"
TABLE = OUT / "tables"

HPARAMS = ["temperature", "num_ctx", "num_predict", "top_p", "repeat_penalty"]
SCORE_COLS = ["faithfulness", "answer_relevancy", "conciseness", "polish_quality", "composite_score"]

MODEL_PALETTE = {
    "llama3.2:3b": "#1f77b4",  # niebieski
    "gemma2:2b": "#d62728",  # czerwony
    "gemma3:4b": "#2ca02c",  # zielony
    "qwen2.5:3b": "#9467bd",  # fioletowy
    "phi3.5": "#ff7f0e",  # pomaranczowy
    "batiai/gemma4-e2b:q4": "#8c564b",  # brazowy
    "gemma4:e2b": "#e377c2",  # rozowy
}

TARGET_LABELS = {
    "score_mean": "sredni wynik laczny",
    "nonhall_score_mean": "sredni wynik bez halucynacji",
    "hallucination_rate": "odsetek halucynacji",
    "e2e_median_ms": "mediana opoznienia E2E",
}

REGRESSION_LABELS = {
    "hallucination_only": "tylko halucynacje",
    "hparams_only": "tylko hiperparametry",
    "model_only": "tylko model",
    "model_plus_hparams": "model + hiperparametry",
    "model_plus_hallucination": "model + halucynacje",
    "full": "pelny model",
}


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results-*/*.jsonl"))):
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_dataset"] = p.parent.name
                row["_file"] = p.name
                row["_line"] = line_no
                rows.append(row)

    df = pd.DataFrame(rows)
    df["hallucination"] = df["hallucination_flag"].astype(int)
    df["nonhallucinated"] = 1 - df["hallucination"]
    df["empty_answer"] = df["model_answer"].fillna("").str.strip().eq("").astype(int)
    df["e2e_s"] = df["e2e_ms"] / 1000
    df["ttft_s"] = df["ttft_ms"] / 1000
    df["trial_key"] = (
        df["_dataset"].astype(str)
        + "|"
        + df["_file"].astype(str)
        + "|"
        + df["model"].astype(str)
        + "|"
        + df["trial"].astype(str)
    )
    return df


def trial_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["_dataset", "_file", "model", "trial"]
    base = (
        df.groupby(group_cols)
        .agg(
            n_cases=("case_id", "count"),
            score_mean=("composite_score", "mean"),
            score_sd_cases=("composite_score", "std"),
            hallucination_rate=("hallucination", "mean"),
            empty_rate=("empty_answer", "mean"),
            e2e_median_ms=("e2e_ms", "median"),
            e2e_p90_ms=("e2e_ms", lambda s: s.quantile(0.90)),
            ttft_median_ms=("ttft_ms", "median"),
        )
        .reset_index()
    )
    hp = df.groupby(group_cols)[HPARAMS].first().reset_index()
    nonhall = (
        df[df["hallucination"] == 0]
        .groupby(group_cols)
        .agg(
            nonhall_n=("case_id", "count"),
            nonhall_score_mean=("composite_score", "mean"),
            nonhall_score_sd_cases=("composite_score", "std"),
        )
        .reset_index()
    )
    out = base.merge(hp, on=group_cols, how="left").merge(nonhall, on=group_cols, how="left")
    out["pred_score_from_decomposition"] = (1 - out["hallucination_rate"]) * out["nonhall_score_mean"]
    out["decomposition_abs_error"] = (out["score_mean"] - out["pred_score_from_decomposition"]).abs()
    return out


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), g in df.groupby(["_dataset", "model"]):
        non = g[g["hallucination"] == 0]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "rows": len(g),
                "trials": g[["_file", "trial"]].drop_duplicates().shape[0],
                "cases": g["case_id"].nunique(),
                "score_mean": g["composite_score"].mean(),
                "score_sd": g["composite_score"].std(),
                "hallucination_rate": g["hallucination"].mean(),
                "nonhall_score_mean": non["composite_score"].mean(),
                "nonhall_score_sd": non["composite_score"].std(),
                "e2e_median_ms": g["e2e_ms"].median(),
                "e2e_p90_ms": g["e2e_ms"].quantile(0.90),
                "ttft_median_ms": g["ttft_ms"].median(),
                "empty_rate": g["empty_answer"].mean(),
                "throttling_rate": g["throttling"].astype(int).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "score_mean"], ascending=[True, False])


def bootstrap_ci(values: pd.Series, n_boot: int = 5000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def selected_model_ci(df: pd.DataFrame) -> pd.DataFrame:
    sel = df[df["_dataset"] == "results-selected-9.05"].copy()
    rows = []
    for model, g in sel.groupby("model"):
        score_ci = bootstrap_ci(g["composite_score"])
        hall_ci = bootstrap_ci(g["hallucination"])
        latency_ci = bootstrap_ci(g["e2e_ms"])
        rows.append(
            {
                "model": model,
                "score_mean": g["composite_score"].mean(),
                "score_ci_low": score_ci[0],
                "score_ci_high": score_ci[1],
                "hallucination_rate": g["hallucination"].mean(),
                "hallucination_ci_low": hall_ci[0],
                "hallucination_ci_high": hall_ci[1],
                "e2e_mean_ms": g["e2e_ms"].mean(),
                "e2e_mean_ci_low": latency_ci[0],
                "e2e_mean_ci_high": latency_ci[1],
                "e2e_median_ms": g["e2e_ms"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False)


def decomposition_table(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def summarize(label_dataset: str, label_file: str, label_model: str, g: pd.DataFrame) -> dict:
        reg = stats.linregress(g["hallucination_rate"], g["score_mean"]) if g["hallucination_rate"].nunique() > 1 else None
        score_sd = g["score_mean"].std()
        nonhall_sd = g["nonhall_score_mean"].std()
        return {
            "dataset": label_dataset,
            "file": label_file,
            "model": label_model,
            "n_trials": len(g),
            "score_mean": g["score_mean"].mean(),
            "score_sd_between_trials": score_sd,
            "hallucination_rate_mean": g["hallucination_rate"].mean(),
            "hallucination_rate_sd": g["hallucination_rate"].std(),
            "nonhall_score_mean": g["nonhall_score_mean"].mean(),
            "nonhall_score_sd_between_trials": nonhall_sd,
            "nonhall_sd_to_score_sd_ratio": nonhall_sd / score_sd if score_sd and not np.isnan(score_sd) else np.nan,
            "corr_score_hallucination": reg.rvalue if reg else np.nan,
            "r2_score_from_hallucination_rate": reg.rvalue**2 if reg else np.nan,
            "slope_score_from_hallucination_rate": reg.slope if reg else np.nan,
            "max_decomposition_abs_error": g["decomposition_abs_error"].max(),
        }

    rows.append(summarize("__ALL__", "__ALL__", "__ALL__", ts.dropna(subset=["nonhall_score_mean"])))
    for (dataset, file_name, model), g in ts.groupby(["_dataset", "_file", "model"]):
        rows.append(summarize(dataset, file_name, model, g.dropna(subset=["nonhall_score_mean"])))
    return pd.DataFrame(rows)


def regression_comparison(ts: pd.DataFrame) -> pd.DataFrame:
    clean = ts.dropna(subset=["score_mean", "hallucination_rate", "nonhall_score_mean", *HPARAMS]).copy()
    selected = clean[clean["_dataset"] == "results-selected-9.05"].copy()
    datasets = [("__ALL__", clean), ("results-selected-9.05", selected)]
    targets = ["score_mean", "nonhall_score_mean", "hallucination_rate", "e2e_median_ms"]
    formulas = {
        "hallucination_only": "{target} ~ hallucination_rate",
        "hparams_only": "{target} ~ temperature + C(num_ctx) + num_predict + top_p + repeat_penalty",
        "model_only": "{target} ~ C(model)",
        "model_plus_hparams": "{target} ~ C(model) + temperature + C(num_ctx) + num_predict + top_p + repeat_penalty",
        "model_plus_hallucination": "{target} ~ C(model) + hallucination_rate",
        "full": "{target} ~ C(model) + hallucination_rate + temperature + C(num_ctx) + num_predict + top_p + repeat_penalty",
    }

    rows = []
    for dataset_name, data in datasets:
        for target in targets:
            for model_name, formula_template in formulas.items():
                if target == "hallucination_rate" and "hallucination_rate" in formula_template.split("~", 1)[1]:
                    continue
                formula = formula_template.format(target=target)
                try:
                    fit = smf.ols(formula, data=data).fit()
                except Exception as exc:  # keep table complete for singular small groups
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "target": target,
                            "regression": model_name,
                            "n": len(data),
                            "r2": np.nan,
                            "adj_r2": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "error": str(exc),
                        }
                    )
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "target": target,
                        "regression": model_name,
                        "n": int(fit.nobs),
                        "r2": fit.rsquared,
                        "adj_r2": fit.rsquared_adj,
                        "aic": fit.aic,
                        "bic": fit.bic,
                        "error": "",
                    }
                )
    return pd.DataFrame(rows)


def hparam_correlations(ts: pd.DataFrame) -> pd.DataFrame:
    targets = ["score_mean", "hallucination_rate", "nonhall_score_mean", "e2e_median_ms"]
    rows = []
    for (dataset, file_name, model), g in ts.groupby(["_dataset", "_file", "model"]):
        for target in targets:
            for hp in HPARAMS:
                valid = g[[hp, target]].replace([np.inf, -np.inf], np.nan).dropna()
                x = valid[hp]
                y = valid[target]
                if len(valid) < 3 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                    pearson_r = np.nan
                    pearson_p = np.nan
                    spearman_r = np.nan
                    spearman_p = np.nan
                else:
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                    spearman_r, spearman_p = stats.spearmanr(x, y)
                rows.append(
                    {
                        "dataset": dataset,
                        "file": file_name,
                        "model": model,
                        "n_trials": len(g),
                        "target": target,
                        "hparam": hp,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                    }
                )
    return pd.DataFrame(rows)


def incremental_r2(reg_tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, target), g in reg_tbl.groupby(["dataset", "target"]):
        values = g.set_index("regression")["r2"].to_dict()
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "model_only_r2": values.get("model_only", np.nan),
                "hparams_only_r2": values.get("hparams_only", np.nan),
                "hallucination_only_r2": values.get("hallucination_only", np.nan),
                "hparams_added_after_model_delta_r2": values.get("model_plus_hparams", np.nan)
                - values.get("model_only", np.nan),
                "hallucination_added_after_model_delta_r2": values.get("model_plus_hallucination", np.nan)
                - values.get("model_only", np.nan),
                "hparams_added_after_model_and_hallucination_delta_r2": values.get("full", np.nan)
                - values.get("model_plus_hallucination", np.nan),
                "hallucination_added_after_model_and_hparams_delta_r2": values.get("full", np.nan)
                - values.get("model_plus_hparams", np.nan),
            }
        )
    return pd.DataFrame(rows)


def case_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["_dataset", "case_id", "category"])
        .agg(
            n=("case_id", "count"),
            score_mean=("composite_score", "mean"),
            hallucination_rate=("hallucination", "mean"),
            e2e_median_ms=("e2e_ms", "median"),
            empty_rate=("empty_answer", "mean"),
        )
        .reset_index()
        .sort_values(["_dataset", "hallucination_rate"], ascending=[True, False])
    )


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["_dataset", "category"])
        .agg(
            n=("case_id", "count"),
            score_mean=("composite_score", "mean"),
            hallucination_rate=("hallucination", "mean"),
            e2e_median_ms=("e2e_ms", "median"),
            empty_rate=("empty_answer", "mean"),
        )
        .reset_index()
        .sort_values(["_dataset", "hallucination_rate"], ascending=[True, False])
    )


def best_worst_trials(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, file_name, model), g in ts.groupby(["_dataset", "_file", "model"]):
        if len(g) < 2:
            continue
        best = g.loc[g["score_mean"].idxmax()]
        worst = g.loc[g["score_mean"].idxmin()]
        rows.append(
            {
                "dataset": dataset,
                "file": file_name,
                "model": model,
                "n_trials": len(g),
                "best_trial": int(best["trial"]),
                "best_score": best["score_mean"],
                "best_hallucination_rate": best["hallucination_rate"],
                "best_nonhall_score": best["nonhall_score_mean"],
                "worst_trial": int(worst["trial"]),
                "worst_score": worst["score_mean"],
                "worst_hallucination_rate": worst["hallucination_rate"],
                "worst_nonhall_score": worst["nonhall_score_mean"],
                "score_gap": best["score_mean"] - worst["score_mean"],
                "hallucination_rate_gap": worst["hallucination_rate"] - best["hallucination_rate"],
                "nonhall_score_gap": best["nonhall_score_mean"] - worst["nonhall_score_mean"],
            }
        )
    return pd.DataFrame(rows)


def savefig(name: str) -> None:
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def make_figures(df: pd.DataFrame, ts: pd.DataFrame, model_tbl: pd.DataFrame, case_tbl: pd.DataFrame, reg_tbl: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)

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
    savefig("01_score_vs_hallucination_all")

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
    savefig("02_score_vs_hallucination_selected")

    plt.figure(figsize=(8.2, 5.2))
    sns.stripplot(data=selected, x="model", y="nonhall_score_mean", hue="model", palette=MODEL_PALETTE, jitter=0.18, size=6, legend=False)
    sns.pointplot(data=selected, x="model", y="nonhall_score_mean", color="black", errorbar="sd", markers="_", linestyles="none")
    plt.title("Jakość warunkowa po usunięciu odpowiedzi z halucynacją")
    plt.xlabel("")
    plt.ylabel("Średni wynik łączny przy hallucination_flag=false")
    plt.xticks(rotation=20, ha="right")
    savefig("03_nonhall_score_spread_selected")

    selected_models = model_tbl[model_tbl["dataset"] == "results-selected-9.05"].copy()
    plt.figure(figsize=(7.6, 5.2))
    sns.scatterplot(
        data=selected_models,
        x="e2e_median_ms",
        y="hallucination_rate",
        hue="model",
        palette=MODEL_PALETTE,
        size="score_mean",
        sizes=(120, 360),
        legend=False,
    )
    for _, row in selected_models.iterrows():
        plt.annotate(row["model"], (row["e2e_median_ms"], row["hallucination_rate"]), xytext=(7, 3), textcoords="offset points")
    plt.title("Kompromis Pareto: opóźnienie a halucynacje")
    plt.xlabel("Mediana opóźnienia E2E [ms]")
    plt.ylabel("Odsetek halucynacji")
    savefig("04_latency_hallucination_pareto_selected")

    selected_cases = case_tbl[case_tbl["_dataset"] == "results-selected-9.05"].nlargest(12, "hallucination_rate")
    plt.figure(figsize=(8.5, 5.2))
    sns.barplot(data=selected_cases, y="case_id", x="hallucination_rate", hue="category", dodge=False)
    plt.title("Przypadki testowe najczęściej oznaczane jako halucynacje")
    plt.xlabel("Odsetek halucynacji")
    plt.ylabel("")
    plt.legend(title="Kategoria", loc="lower right", frameon=False)
    savefig("05_case_hallucination_rates_selected")

    selected_corr = hparam_correlations(selected).query("target in ['score_mean', 'hallucination_rate', 'nonhall_score_mean', 'e2e_median_ms']")
    selected_corr["target"] = selected_corr["target"].map(TARGET_LABELS)
    selected_corr = selected_corr.pivot_table(index=["model", "target"], columns="hparam", values="pearson_r")
    plt.figure(figsize=(8.2, 7.2))
    sns.heatmap(selected_corr, cmap="vlag", center=0, annot=True, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Jednowymiarowe korelacje hiperparametrów w wybranych runach")
    plt.xlabel("")
    plt.ylabel("")
    savefig("06_hparam_correlation_heatmap_selected")

    r2_plot = reg_tbl[(reg_tbl["dataset"] == "results-selected-9.05") & (reg_tbl["target"].isin(["score_mean", "nonhall_score_mean", "e2e_median_ms"]))]
    r2_plot = r2_plot.copy()
    r2_plot["target"] = r2_plot["target"].map(TARGET_LABELS)
    r2_plot["regression"] = r2_plot["regression"].map(REGRESSION_LABELS)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=r2_plot, x="target", y="r2", hue="regression")
    plt.title("Siła wyjaśniająca modeli regresyjnych w wybranych runach")
    plt.xlabel("")
    plt.ylabel("R²")
    plt.xticks(rotation=10, ha="right")
    plt.legend(title="Regresja", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    savefig("07_regression_r2_comparison_selected")


def write_report(
    df: pd.DataFrame,
    ts: pd.DataFrame,
    model_tbl: pd.DataFrame,
    decomp_tbl: pd.DataFrame,
    reg_tbl: pd.DataFrame,
    incremental_tbl: pd.DataFrame,
    case_tbl: pd.DataFrame,
    category_tbl: pd.DataFrame,
    best_worst_tbl: pd.DataFrame,
    ci_tbl: pd.DataFrame,
) -> None:
    all_decomp = decomp_tbl[decomp_tbl["dataset"] == "__ALL__"].iloc[0]
    selected_decomp = decomp_tbl[decomp_tbl["dataset"] == "results-selected-9.05"].copy()
    selected_models = model_tbl[model_tbl["dataset"] == "results-selected-9.05"].copy()
    selected_cases = case_tbl[case_tbl["_dataset"] == "results-selected-9.05"].nlargest(8, "hallucination_rate")
    selected_best = best_worst_tbl[best_worst_tbl["dataset"] == "results-selected-9.05"].copy()

    flag_true_nonzero = int(((df["hallucination"] == 1) & (df["composite_score"] != 0)).sum())
    flag_false_zero = int(((df["hallucination"] == 0) & (df["composite_score"] == 0)).sum())
    max_decomp_error = ts["decomposition_abs_error"].max()

    r2_selected = reg_tbl[
        (reg_tbl["dataset"] == "results-selected-9.05")
        & (reg_tbl["target"] == "score_mean")
    ][["regression", "r2", "adj_r2", "aic"]]
    incremental_selected = incremental_tbl[
        (incremental_tbl["dataset"] == "results-selected-9.05")
        & (incremental_tbl["target"].isin(["score_mean", "nonhall_score_mean", "hallucination_rate", "e2e_median_ms"]))
    ].copy()
    incremental_report = incremental_selected[
        [
            "target",
            "model_only_r2",
            "hparams_only_r2",
            "hallucination_only_r2",
            "hparams_added_after_model_delta_r2",
            "hallucination_added_after_model_delta_r2",
            "hparams_added_after_model_and_hallucination_delta_r2",
            "hallucination_added_after_model_and_hparams_delta_r2",
        ]
    ].copy()
    for col in incremental_report.columns:
        if col != "target":
            incremental_report[col] = incremental_report[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")

    lines = [
        "# Analiza statystyczna wyników benchmarków",
        "",
        "## Zbiór danych",
        "",
        f"- Liczba rekordów: {len(df)}",
        f"- Liczba podsumowań triali: {len(ts)}",
        f"- Liczba plików: {df['_file'].nunique()}",
        f"- Liczba modeli: {df['model'].nunique()}",
        "",
        "## Główna dekompozycja wyniku",
        "",
        (
            "W tym benchmarku każdy rekord z `hallucination_flag=true` otrzymuje `composite_score=0`. "
            f"Liczba rekordów z halucynacją i niezerowym wynikiem: {flag_true_nonzero}. "
            f"Liczba rekordów bez halucynacji i zerowym wynikiem: {flag_false_zero}."
        ),
        "",
        (
            "Dlatego na poziomie triala zachodzi zależność: "
            "`mean_score = (1 - hallucination_rate) * mean_score_given_no_hallucination`. "
            f"Maksymalny błąd numerycznej rekonstrukcji w danych wynosi {max_decomp_error:.8f}."
        ),
        "",
        (
            f"Dla wszystkich triali regresja liniowa `mean_score ~ hallucination_rate` daje "
            f"R^2={all_decomp['r2_score_from_hallucination_rate']:.4f}, "
            f"korelację Pearsona r={all_decomp['corr_score_hallucination']:.4f} oraz współczynnik kierunkowy "
            f"{all_decomp['slope_score_from_hallucination_rate']:.4f}."
        ),
        "",
        "Interpretacja: większość zmienności `composite_score` jest wyjaśniana przez częstość binarnej flagi halucynacji.",
        "",
        "## Wybrane runy: podsumowanie modeli",
        "",
        selected_models[
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
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Wybrane runy: przedziały ufności",
        "",
        ci_tbl.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Wybrane runy: dekompozycja według modelu i pliku runu",
        "",
        selected_decomp[
            [
                "model",
                "n_trials",
                "score_sd_between_trials",
                "hallucination_rate_sd",
                "nonhall_score_sd_between_trials",
                "nonhall_sd_to_score_sd_ratio",
                "r2_score_from_hallucination_rate",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Porównanie regresji dla wybranych triali",
        "",
        r2_selected.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Przyrostowe R² dla wybranych triali",
        "",
        incremental_report.to_markdown(index=False),
        "",
        (
            "Najważniejsze porównanie dotyczy `score_mean`: regresja używająca wyłącznie `hallucination_rate` "
            "ma bardzo wysoką siłę wyjaśniającą. Hiperparametry dodają znacznie mniej wyjaśnianej wariancji, "
            "gdy uwzględniona jest już tożsamość modelu. Dla `nonhall_score_mean` wartości R² należy "
            "interpretować ostrożnie, ponieważ po usunięciu odpowiedzi z halucynacją zakres tej zmiennej jest wąski."
        ),
        "",
        "## Porównanie najlepszego i najgorszego triala",
        "",
        selected_best[
            [
                "model",
                "best_score",
                "best_hallucination_rate",
                "best_nonhall_score",
                "worst_score",
                "worst_hallucination_rate",
                "worst_nonhall_score",
                "score_gap",
                "hallucination_rate_gap",
                "nonhall_score_gap",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Przypadki testowe najczęściej oznaczane jako halucynacje",
        "",
        selected_cases[
            ["case_id", "category", "n", "score_mean", "hallucination_rate", "e2e_median_ms"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Podsumowanie kategorii",
        "",
        category_tbl[category_tbl["_dataset"] == "results-selected-9.05"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Wykresy",
        "",
        "- `figures/01_score_vs_hallucination_all.*`",
        "- `figures/02_score_vs_hallucination_selected.*`",
        "- `figures/03_nonhall_score_spread_selected.*`",
        "- `figures/04_latency_hallucination_pareto_selected.*`",
        "- `figures/05_case_hallucination_rates_selected.*`",
        "- `figures/06_hparam_correlation_heatmap_selected.*`",
        "- `figures/07_regression_r2_comparison_selected.*`",
        "",
        "## Wniosek do pracy",
        "",
        (
            "Benchmark nie dostarcza silnych dowodów na to, że hiperparametry dobierane przez Optunę "
            "istotnie zmieniają semantyczną jakość odpowiedzi. Wynik łączny jest zdominowany przez binarną "
            "karę za halucynację. Po usunięciu odpowiedzi z flagą halucynacji wynik warunkowy zmienia się "
            "między trialami tylko nieznacznie. W konsekwencji dalsze testowanie modeli na tym samym "
            "benchmarku ma ograniczoną wartość jako metoda wnioskowania o hiperparametrach. Bardziej "
            "praktycznym kryterium wyboru jest optymalizacja Pareto względem opóźnienia i odsetka halucynacji."
        ),
        "",
        "## Zastrzeżenie metodologiczne",
        "",
        (
            "Korelacje i modele OLS mają charakter opisowy, a nie przyczynowy. Część plików zawiera tylko "
            "pięć triali, więc estymacje wielowymiarowe nie powinny być używane do mocnych twierdzeń "
            "przyczynowych. Najsilniejszy argument wynika z dokładnej dekompozycji wyniku oraz z obserwowanego "
            "spadku zmienności między trialami po warunkowaniu na odpowiedziach bez flagi halucynacji."
        ),
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    TABLE.mkdir(exist_ok=True)

    df = load_rows()
    ts = trial_summary(df)
    model_tbl = model_summary(df)
    decomp_tbl = decomposition_table(ts)
    reg_tbl = regression_comparison(ts)
    incremental_tbl = incremental_r2(reg_tbl)
    hcorr_tbl = hparam_correlations(ts)
    case_tbl = case_summary(df)
    category_tbl = category_summary(df)
    best_worst_tbl = best_worst_trials(ts)
    ci_tbl = selected_model_ci(df)

    df.to_csv(TABLE / "raw_rows_enriched.csv", index=False)
    ts.to_csv(TABLE / "trial_summary.csv", index=False)
    model_tbl.to_csv(TABLE / "model_summary.csv", index=False)
    decomp_tbl.to_csv(TABLE / "score_decomposition.csv", index=False)
    reg_tbl.to_csv(TABLE / "regression_comparison.csv", index=False)
    incremental_tbl.to_csv(TABLE / "incremental_r2.csv", index=False)
    hcorr_tbl.to_csv(TABLE / "hparam_correlations.csv", index=False)
    case_tbl.to_csv(TABLE / "case_summary.csv", index=False)
    category_tbl.to_csv(TABLE / "category_summary.csv", index=False)
    best_worst_tbl.to_csv(TABLE / "best_worst_trials.csv", index=False)
    ci_tbl.to_csv(TABLE / "selected_model_confidence_intervals.csv", index=False)

    make_figures(df, ts, model_tbl, case_tbl, reg_tbl)
    write_report(df, ts, model_tbl, decomp_tbl, reg_tbl, incremental_tbl, case_tbl, category_tbl, best_worst_tbl, ci_tbl)

    all_decomp = decomp_tbl[decomp_tbl["dataset"] == "__ALL__"].iloc[0]
    print(f"Rows: {len(df)}")
    print(f"Trials: {len(ts)}")
    print(f"Output: {OUT.relative_to(ROOT)}")
    print(f"R2(score_mean ~ hallucination_rate): {all_decomp['r2_score_from_hallucination_rate']:.4f}")
    print(f"corr(score_mean, hallucination_rate): {all_decomp['corr_score_hallucination']:.4f}")


if __name__ == "__main__":
    main()
