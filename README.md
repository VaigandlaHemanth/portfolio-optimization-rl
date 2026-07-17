# portfolio-optimization-rl

A regime-aware **PPO** agent for long-only portfolio allocation, built on a custom Gym environment with turnover-based transaction costs — and, importantly, evaluated **out-of-sample against real baselines** rather than in-sample.

[![tests](https://img.shields.io/badge/tests-11%20passing-brightgreen.svg)](tests/test_portfolio_rl.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## The idea, and the honest evaluation

The easy mistake in RL-for-trading is to train and evaluate on the same price
history. During a 2018–2023 mega-cap bull run almost any long-only policy
"makes money" in-sample, so that number says nothing. This project is built
around avoiding that:

- **Chronological split** — train on `2018-01-01 … 2022-01-01`, hold out
  `2022-01-01 … 2024-01-01`. The agent never sees test prices while training.
- **Out-of-sample metrics** — total return, annualized Sharpe, volatility, and
  max drawdown are reported on the held-out window only.
- **Real baselines** — the agent is compared to **equal-weight (daily rebalanced)**
  and **buy-and-hold** on the same prices and cost model. Beating buy-and-hold
  out-of-sample on a risk-adjusted basis is the bar.
- **No lookahead in regimes** — the volatility baseline that defines the
  low/high/crash regimes is fixed on the training set and reused on the test set.

## What's in the environment

- Custom `PortfolioEnv` (classic gym ≤ 0.25 API, works with Stable-Baselines3 2.1):
  softmax-normalized long-only weights, log-return reward net of turnover cost,
  and a ruin cutoff at 50% of initial capital.
- Observation = flattened window of recent returns + previous weights +
  regime one-hot (low / high / crash from rolling volatility).
- Price data is **injected**, not downloaded inside the env — so the same class
  serves the train and test windows and can be unit-tested offline.

## Layout

```
src/portfolio_rl/
  env.py        # PortfolioEnv (gym)
  data.py       # yfinance download + chronological train_test_split
  regimes.py    # volatility regime labelling (baseline reusable across splits)
  metrics.py    # total return, Sharpe, max drawdown, volatility (pure numpy)
  baselines.py  # equal-weight and buy-and-hold backtests
train.py        # train PPO on the training window, save the model
evaluate.py     # out-of-sample run + benchmark comparison table
tests/          # 11 offline unit tests (no network, no training)
RL.ipynb        # narrated walkthrough of the whole flow
```

## Run it

```bash
pip install -r requirements.txt

# offline unit tests (metrics / regimes / baselines / env contract)
pytest -q

# full pipeline (needs network for prices + a few minutes to train)
python train.py --timesteps 100000
python evaluate.py
```

`evaluate.py` prints a comparison table:

```
Strategy                 TotRet   Sharpe     Vol    MaxDD       Final $
------------------------------------------------------------------------
PPO (regime-aware)         ...%     ....    ...%    ...%       ...
Equal-weight (daily)       ...%     ....    ...%    ...%       ...
Buy-and-hold               ...%     ....    ...%    ...%       ...
```

Paste your real `evaluate.py` numbers here — the head-to-head against
buy-and-hold is the whole point of the project.

## Limitations (stated honestly)

- 3-ticker mega-cap universe: highly correlated and survivorship-biased.
- A single test period is one regime path; walk-forward validation is the next step.
- The reward is log-return net of cost, with no explicit risk penalty yet.

Research exercise in RL environment design and evaluation — not financial advice.

## License

[MIT](LICENSE).
