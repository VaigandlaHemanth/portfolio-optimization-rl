"""Evaluate the trained PPO agent OUT-OF-SAMPLE against baselines.

Runs the saved policy on the held-out test window and compares it to two
non-RL benchmarks (equal-weight rebalanced, buy-and-hold) on the same prices
and cost model, then prints a metrics table. This is the number that matters:
in-sample a policy can trivially "6x" during a bull run — only the held-out
comparison against buy-and-hold says whether the agent added anything.

Usage:
    python evaluate.py
"""
from __future__ import annotations

import numpy as np

from src.portfolio_rl import PortfolioEnv, baselines, metrics
from src.portfolio_rl.data import download_prices, train_test_split
from train import DATA_END, DATA_START, SPLIT_DATE, TICKERS, SEED

WINDOW = 30
VOL_WINDOW = 20
INITIAL = 1_000_000
COST = 0.001


def run_agent(model, prices, dates, baseline_vol):
    env = PortfolioEnv(
        prices, dates=dates, window_size=WINDOW, vol_window=VOL_WINDOW,
        baseline_vol=baseline_vol, initial_capital=INITIAL, transaction_cost=COST,
        seed=SEED,
    )
    obs = env.reset()
    curve = [env.initial_capital]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        curve.append(info["portfolio_value"])
    return np.array(curve), env.avg_turnover


def main():
    from stable_baselines3 import PPO

    prices, dates = download_prices(TICKERS, DATA_START, DATA_END)
    _, (test_prices, test_dates) = train_test_split(prices, dates, SPLIT_DATE)
    baseline_vol = float(np.load("train_baseline_vol.npy")[0])

    model = PPO.load("ppo_portfolio_model")
    agent_curve, agent_turnover = run_agent(model, test_prices, test_dates, baseline_vol)

    test_returns = test_prices[1:] / test_prices[:-1] - 1.0
    eq_curve = baselines.equal_weight(test_returns, WINDOW, INITIAL, COST)
    bh_curve = baselines.buy_and_hold(test_returns, WINDOW, INITIAL, COST)

    rows = {
        "PPO (regime-aware)": metrics.summary(agent_curve, turnover=agent_turnover),
        "Equal-weight (daily)": metrics.summary(eq_curve),
        "Buy-and-hold": metrics.summary(bh_curve),
    }

    print(f"\nOut-of-sample test window: {SPLIT_DATE} -> {DATA_END}  ({len(test_prices)} days)\n")
    header = f"{'Strategy':<22}{'TotRet':>9}{'Sharpe':>9}{'Vol':>8}{'MaxDD':>9}{'Final $':>14}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        print(
            f"{name:<22}{m['total_return']*100:>8.1f}%{m['annualized_sharpe']:>9.2f}"
            f"{m['annualized_vol']*100:>7.1f}%{m['max_drawdown']*100:>8.1f}%"
            f"{m['final_value']:>14,.0f}"
        )
    print("\nRead the PPO row against Buy-and-hold — beating it out-of-sample is the bar.")


if __name__ == "__main__":
    main()
