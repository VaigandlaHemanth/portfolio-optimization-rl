"""Non-RL baseline strategies to compare the PPO agent against.

Without a baseline, a portfolio "6x'd" number is meaningless — during a
2018-2023 mega-cap bull run, simply holding the assets also multiplies. These
backtests run on the SAME price series and cost model as the RL environment so
the comparison is apples-to-apples.

Each backtest returns an equity curve (np.ndarray of portfolio values) aligned
step-for-step with ``PortfolioEnv`` (which starts acting at index ``window``).
"""
from __future__ import annotations

import numpy as np


def _run(weights_fn, returns, window, initial_capital, transaction_cost):
    """Generic long-only backtest given a per-step target-weights function."""
    returns = np.asarray(returns, dtype=np.float64)
    n_assets = returns.shape[1]
    value = float(initial_capital)
    prev_w = np.ones(n_assets) / n_assets
    curve = [value]

    # range matches PortfolioEnv exactly: the env's done-check fires before it
    # would consume the final return row, so it takes steps window..T-1. We
    # mirror that here so agent and baseline curves are step-for-step aligned.
    for step in range(window, len(returns)):
        target_w = weights_fn(prev_w, step)
        asset_ret = returns[step - 1]
        gross = float(np.dot(target_w, asset_ret))
        turnover = float(np.sum(np.abs(target_w - prev_w)))
        net = gross - transaction_cost * turnover
        value *= (1.0 + net)
        curve.append(value)
        prev_w = target_w

    return np.array(curve, dtype=np.float64)


def equal_weight(returns, window=30, initial_capital=1_000_000, transaction_cost=0.001):
    """Rebalance to equal weights every day."""
    n_assets = np.asarray(returns).shape[1]
    target = np.ones(n_assets) / n_assets
    return _run(lambda prev_w, step: target, returns, window, initial_capital, transaction_cost)


def buy_and_hold(returns, window=30, initial_capital=1_000_000, transaction_cost=0.001):
    """Buy equal weights once at the start and let them drift (no rebalancing).

    Weights drift with realized returns, so turnover is ~0 after entry — the
    honest "do nothing" benchmark.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n_assets = returns.shape[1]
    value = float(initial_capital)
    # Allocate equally by capital at entry, then hold shares.
    holdings = np.ones(n_assets) / n_assets  # value fraction per asset
    curve = [value]

    for step in range(window, len(returns)):  # matches PortfolioEnv (see _run)
        asset_ret = returns[step - 1]
        holdings = holdings * (1.0 + asset_ret)  # each sleeve grows with its asset
        new_value = value * float(np.sum(holdings))
        # renormalize sleeves to fractions of the new value
        holdings = holdings / np.sum(holdings)
        value = new_value
        curve.append(value)

    return np.array(curve, dtype=np.float64)
