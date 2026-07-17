"""Long-only portfolio allocation environment (classic gym <= 0.25 API).

Design change from the original notebook: price data is *injected* rather than
downloaded inside the environment. That lets the same class serve a training
window and a disjoint test window (see ``data.train_test_split``), and lets the
env be unit-tested on synthetic prices with no network access.

Observation = [flattened window of returns, previous weights, regime one-hot].
Action      = raw scores, softmax-normalized into long-only weights.
Reward      = log return of the portfolio, net of turnover-based costs.
"""
from __future__ import annotations

import gym
import numpy as np
from gym import spaces

from .regimes import compute_regimes


class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        dates=None,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        window_size: int = 30,
        vol_window: int = 20,
        baseline_vol: float | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        if self.prices.ndim != 2:
            raise ValueError("prices must be a 2-D array of shape (T, n_assets)")
        self.dates = dates
        self.n_assets = self.prices.shape[1]
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.vol_window = vol_window

        # Daily returns and regime labels (baseline fixed on train, reused on test)
        self.returns = self.prices[1:] / self.prices[:-1] - 1.0
        self.regimes, self.baseline_vol = compute_regimes(
            self.returns, vol_window=vol_window, baseline_vol=baseline_vol
        )

        if len(self.returns) <= self.window_size:
            raise ValueError(
                f"need more than window_size={self.window_size} return rows, "
                f"got {len(self.returns)}"
            )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        obs_dim = self.window_size * self.n_assets + self.n_assets + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = None
        self.portfolio_value = None
        self.prev_weights = None
        self._turnovers: list[float] = []

        if seed is not None:
            self.seed(seed)

    # ------------------------------------------------------------------ gym API
    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        try:
            self.action_space.seed(seed)
        except Exception:
            pass
        return [seed]

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_capital
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self._turnovers = []
        return self._get_observation()

    def step(self, action):
        weights = self._softmax(action)
        asset_returns = self.returns[self.current_step - 1]

        gross = float(np.dot(weights, asset_returns))
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        net = gross - self.transaction_cost * turnover

        prev_value = self.portfolio_value
        self.portfolio_value *= (1.0 + net)
        reward = float(np.log(self.portfolio_value / prev_value))

        self._turnovers.append(turnover)
        self.prev_weights = weights
        self.current_step += 1

        done = bool(
            self.portfolio_value < 0.5 * self.initial_capital
            or self.current_step >= len(self.returns)
        )
        info = {
            "portfolio_value": float(self.portfolio_value),
            "weights": weights,
            "turnover": turnover,
            "step": int(self.current_step),
        }
        return self._get_observation(), reward, done, info

    # ------------------------------------------------------------- observation
    def _get_observation(self):
        start = self.current_step - self.window_size
        window_flat = self.returns[start:self.current_step].flatten()
        regime_one_hot = np.zeros(3, dtype=np.float32)
        regime_one_hot[self.regimes[self.current_step - 1]] = 1.0
        obs = np.concatenate([window_flat, self.prev_weights, regime_one_hot])
        return obs.astype(np.float32)

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self._turnovers)) if self._turnovers else 0.0

    # -------------------------------------------------------------------- utils
    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def render(self, mode="human"):
        print(f"Step {self.current_step}, value {self.portfolio_value:,.2f}")
