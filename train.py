"""Train a PPO portfolio-allocation agent on the TRAINING window only.

Usage:
    python train.py                 # defaults below
    python train.py --timesteps 200000

The train/test split is chronological: the agent never sees test-period prices
during training, so the metrics reported by ``evaluate.py`` are out-of-sample.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.portfolio_rl import PortfolioEnv
from src.portfolio_rl.data import download_prices, train_test_split

# Chronological split — train on the earlier years, hold out the later ones.
TICKERS = ("AAPL", "MSFT", "GOOGL")
DATA_START = "2018-01-01"
DATA_END = "2024-01-01"
SPLIT_DATE = "2022-01-01"     # train < 2022-01-01, test >= 2022-01-01
SEED = 42


def build_train_env(window_size=30, vol_window=20):
    prices, dates = download_prices(TICKERS, DATA_START, DATA_END)
    (train_prices, train_dates), _ = train_test_split(prices, dates, SPLIT_DATE)
    env = PortfolioEnv(
        train_prices, dates=train_dates, window_size=window_size,
        vol_window=vol_window, seed=SEED,
    )
    # Persist the training baseline so evaluation labels the test regimes the
    # same way (no lookahead). Written next to the model.
    np.save("train_baseline_vol.npy", np.array([env.baseline_vol]))
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", default="ppo_portfolio_model")
    args = parser.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    vec_env = DummyVecEnv([build_train_env])
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=SEED,
        learning_rate=3e-4, batch_size=64, n_steps=2048, gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip — now run: python evaluate.py")


if __name__ == "__main__":
    main()
