"""
analytics/overfit.py -- Optimizer overfitting diagnostics.
"""

from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


_EULER_GAMMA = 0.5772156649015329


def deflated_sharpe_ratio(
    sharpe_values: Iterable[float],
    selected_returns: pd.Series,
    threshold: float = 0.95,
) -> dict:
    """
    Compute a Deflated Sharpe Ratio diagnostic for a selected strategy.

    This is an informational multiple-testing adjustment over the searched
    Sharpe values. It does not change optimizer selection.
    """
    sharpe = pd.Series(list(sharpe_values), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n_trials = int(len(sharpe))
    if n_trials < 2:
        return _unavailable("Need at least 2 valid Sharpe trials.", n_trials=n_trials, threshold=threshold)

    returns = pd.Series(selected_returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    sample_size = int(len(returns))
    if sample_size < 3:
        return _unavailable(
            "Need at least 3 selected-return observations.",
            n_trials=n_trials,
            threshold=threshold,
            sample_size=sample_size,
        )

    best_sharpe = float(sharpe.max())
    expected_max = _expected_max_sharpe(sharpe, n_trials)
    skew = float(returns.skew())
    kurtosis = float(returns.kurt() + 3.0)

    if any(np.isnan(v) for v in (best_sharpe, expected_max, skew, kurtosis)):
        return _unavailable(
            "Sharpe or return moments are not finite.",
            n_trials=n_trials,
            threshold=threshold,
            sample_size=sample_size,
        )

    denominator = 1.0 - skew * best_sharpe + ((kurtosis - 1.0) / 4.0) * best_sharpe ** 2
    if denominator <= 0:
        return _unavailable(
            "Deflated Sharpe denominator is non-positive.",
            n_trials=n_trials,
            threshold=threshold,
            sample_size=sample_size,
        )

    z_score = (best_sharpe - expected_max) * np.sqrt(sample_size - 1.0) / np.sqrt(denominator)
    probability = float(NormalDist().cdf(float(z_score)))

    return {
        "available": True,
        "reason": "",
        "n_trials": n_trials,
        "sample_size": sample_size,
        "best_is_sharpe": best_sharpe,
        "expected_max_sharpe": float(expected_max),
        "deflated_sharpe_prob": probability,
        "threshold": threshold,
        "warning": probability < threshold,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def _expected_max_sharpe(sharpe: pd.Series, n_trials: int) -> float:
    mean = float(sharpe.mean())
    std = float(sharpe.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return mean

    normal = NormalDist()
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * np.e)
    return mean + std * (
        (1.0 - _EULER_GAMMA) * normal.inv_cdf(p1)
        + _EULER_GAMMA * normal.inv_cdf(p2)
    )


def _unavailable(reason: str, n_trials: int, threshold: float, sample_size: int = 0) -> dict:
    return {
        "available": False,
        "reason": reason,
        "n_trials": n_trials,
        "sample_size": sample_size,
        "best_is_sharpe": float("nan"),
        "expected_max_sharpe": float("nan"),
        "deflated_sharpe_prob": float("nan"),
        "threshold": threshold,
        "warning": True,
        "skew": float("nan"),
        "kurtosis": float("nan"),
    }
