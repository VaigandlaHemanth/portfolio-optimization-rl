"""Volatility-based market regime labelling.

A regime is one of {0: Low vol, 1: High vol, 2: Crash}, assigned from the
rolling volatility of the average asset return relative to a baseline.

Important for out-of-sample evaluation: the baseline volatility must be fixed
on the training set and *reused* on the test set. Recomputing the baseline on
the test window leaks information about the test period into its own labels.
Pass ``baseline_vol`` on the test set to avoid that lookahead.
"""
from __future__ import annotations

import numpy as np

LOW, HIGH, CRASH = 0, 1, 2


def rolling_volatility(market_returns: np.ndarray, vol_window: int) -> np.ndarray:
    """Trailing std of the market return, same length as the input."""
    market_returns = np.asarray(market_returns, dtype=np.float64)
    n = len(market_returns)
    vol = np.zeros(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - vol_window + 1)
        vol[t] = np.std(market_returns[start:t + 1])
    return vol


def compute_regimes(
    returns: np.ndarray,
    vol_window: int = 20,
    baseline_vol: float | None = None,
) -> tuple[np.ndarray, float]:
    """Label each timestep with a regime.

    Args:
        returns: (T, n_assets) array of per-asset daily returns.
        vol_window: rolling window length in days.
        baseline_vol: mean volatility to compare against. If None it is
            computed from this data (use only on the training set); on the
            test set pass the training baseline to prevent lookahead.

    Returns:
        (regimes array of shape (T,), baseline_vol used).
    """
    returns = np.asarray(returns, dtype=np.float64)
    market_ret = returns.mean(axis=1)
    vol = rolling_volatility(market_ret, vol_window)

    if baseline_vol is None:
        baseline_vol = float(np.mean(vol))

    regimes = np.full(len(vol), LOW, dtype=np.int64)
    regimes[(vol >= baseline_vol) & (vol < 2 * baseline_vol)] = HIGH
    regimes[vol >= 2 * baseline_vol] = CRASH
    return regimes, baseline_vol


def regime_counts(regimes: np.ndarray) -> dict:
    """Count of steps per regime, keyed by name."""
    names = {LOW: "low_vol", HIGH: "high_vol", CRASH: "crash"}
    unique, counts = np.unique(regimes, return_counts=True)
    out = {v: 0 for v in names.values()}
    for u, c in zip(unique, counts):
        out[names[int(u)]] = int(c)
    return out
