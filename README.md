# portfolio-optimization-rl

Reinforcement learning setup for long-only portfolio allocation with a custom gym environment that is aware of market regimes, transaction costs, and risk-sensitive rewards. Training uses PPO from Stable-Baselines3 on daily price data from Yahoo Finance.

## Highlights
- Custom `PortfolioEnv` with softmax-constrained weights, turnover-based transaction costs, and ruin cutoff.
- Regime detection via rolling volatility (Low, High, Crash) fed into observations for regime-aware policies.
- PPO training with Stable-Baselines3 2.1.0 on a classic gym (<=0.25) API via `DummyVecEnv`.
- Evaluation notebook cells for portfolio value curves, regime paths, and Sharpe ratios overall and per regime.

## Project Layout
- [RL.ipynb](RL.ipynb): end-to-end notebook with environment, training, evaluation, and plots.

## Setup
1) Create and activate a virtual environment (example on Windows PowerShell):
```
python -m venv .venv
.venv\Scripts\activate
```
2) Install dependencies:
```
pip install -r requirements.txt
```
3) (Optional) Install a CPU-only PyTorch build if the pinned version fails on your platform: see https://pytorch.org for the correct command.

## Usage
- Open and run [RL.ipynb](RL.ipynb) top to bottom. The first cell pins `stable-baselines3==2.1.0`, `gym==0.25.2`, and `shimmy==1.1.0` to match the custom environment.
- Training uses default tickers `(AAPL, MSFT, GOOGL)` from 2018-01-01 to 2023-01-01. Edit the `tickers`, `start`, and `end` parameters in `PortfolioEnv` or `make_env()` to customize.
- Outputs include portfolio value trajectories, regime frequency plots, and Sharpe ratios (overall and by regime).

## Notes
- The environment follows the classic gym API (`reset() -> obs`, `step() -> obs, reward, done, info`) to align with Stable-Baselines3 2.1.0.
- Actions are softmax-normalized to weights; rewards use log-returns after transaction costs; a ruin condition triggers `done` if capital falls below 50% of the initial value.
- Market regimes are inferred from rolling volatility of average returns (vol window configurable via `vol_window`).

## Next Steps
- Add transaction-cost schedules (e.g., nonlinear or asset-specific fees) and shorting constraints.
- Hyperparameter sweeps for PPO (learning rate, batch size, `n_steps`) and longer training horizons.
- Expand evaluation to include max drawdown, Calmar ratio, and constraint-aware backtests.
- Package the environment as a Python module with unit tests and a scriptable training entrypoint.
