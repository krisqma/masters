"""Log-normal helpers for latency review figures (Ch. 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_lognormal(values: Any) -> dict[str, float]:
    """Fit LN on positive samples: μ, σ on log-scale; center = e^μ; bands e^{μ±σ}."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mu": float("nan"),
            "sigma": float("nan"),
            "center": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "lo2": float("nan"),
            "hi2": float("nan"),
        }
    log_x = np.log(arr)
    mu = float(np.mean(log_x))
    sigma = float(np.std(log_x, ddof=1)) if n > 1 else 0.0
    center = float(np.exp(mu))
    return {
        "n": float(n),
        "mu": mu,
        "sigma": sigma,
        "center": center,
        "lo": float(np.exp(mu - sigma)),
        "hi": float(np.exp(mu + sigma)),
        "lo2": float(np.exp(mu - 2.0 * sigma)),
        "hi2": float(np.exp(mu + 2.0 * sigma)),
    }


def ln_group_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Per-group LN fit; columns: {prefix}_n/mu/sigma/center/lo/hi/lo2/hi2."""
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fit = fit_lognormal(g[value_col])
        row = dict(zip(group_cols, keys))
        for k, v in fit.items():
            row[f"{prefix}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
