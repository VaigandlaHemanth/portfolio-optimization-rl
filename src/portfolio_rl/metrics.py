"""Backtest performance metrics.

Pure NumPy functions — no market data, no RL, no I/O — so they are trivially
unit-testable and reusable across the RL agent and the baseline strategies.
All functions take a 1-D array of portfolio *values* (an equity curve) unless
noted otherwise.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def daily_returns(values: np.ndarray) -> np.ndarray:
    """Simple daily returns from an equity curve of length T -> length T-1."""
    values = np.asarray(values, dtype=np.float64)
    return values[1:] / values[:-1] - 1.0


def total_return(values: np.ndarray) -> float:
    """Total return over the whole curve, e.g. 0.5 == +50%."""
    values = np.asarray(values, dtype=np.float64)
    return float(values[-1] / values[0] - 1.0)


def annualized_sharpe(values: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio from an equity curve (risk_free is per-day)."""
    rets = daily_returns(values)
    if rets.size < 2:
        return 0.0
    excess = rets - risk_free
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def max_drawdown(values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction (e.g. -0.30)."""
    values = np.asarray(values, dtype=np.float64)
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1.0
    return float(drawdowns.min())


def annualized_volatility(values: np.ndarray) -> float:
    """Annualized volatility of daily returns."""
    rets = daily_returns(values)
    if rets.size < 2:
        return 0.0
    return float(np.std(rets) * np.sqrt(TRADING_DAYS))


def summary(values: np.ndarray, turnover: float | None = None) -> dict:
    """Bundle the headline metrics for a single strategy's equity curve."""
    out = {
        "total_return": total_return(values),
        "annualized_sharpe": annualized_sharpe(values),
        "annualized_vol": annualized_volatility(values),
        "max_drawdown": max_drawdown(values),
        "final_value": float(np.asarray(values, dtype=np.float64)[-1]),
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover)
    return out
