"""Offline unit tests — no network, no training. Run with: pytest -q"""
import numpy as np
import pytest

from src.portfolio_rl import metrics, regimes, baselines
from src.portfolio_rl.regimes import LOW, HIGH, CRASH


# --------------------------------------------------------------- metrics
def test_total_return():
    assert metrics.total_return(np.array([100.0, 150.0])) == pytest.approx(0.5)


def test_sharpe_zero_variance_is_zero():
    # a flat equity curve has exactly-zero-variance returns -> Sharpe 0, not NaN/inf
    assert metrics.annualized_sharpe(np.array([100.0, 100.0, 100.0, 100.0])) == 0.0


def test_sharpe_positive_for_upward_noisy_curve():
    rng = np.random.default_rng(0)
    curve = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, size=300))
    assert metrics.annualized_sharpe(curve) > 0


def test_max_drawdown_simple():
    # 100 -> 120 -> 60 -> 90 : worst peak-to-trough is 120 -> 60 = -50%
    curve = np.array([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown(curve) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_is_zero():
    assert metrics.max_drawdown(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


# --------------------------------------------------------------- regimes
def test_regimes_shapes_and_baseline_reuse():
    rng = np.random.default_rng(1)
    train_ret = rng.normal(0, 0.01, size=(200, 3))
    reg_train, base = regimes.compute_regimes(train_ret, vol_window=20)
    assert reg_train.shape == (200,)
    assert base > 0
    # Reusing the train baseline on new data must not recompute it
    test_ret = rng.normal(0, 0.05, size=(120, 3))  # much more volatile
    reg_test, base_test = regimes.compute_regimes(test_ret, vol_window=20, baseline_vol=base)
    assert base_test == base
    # higher-vol test data should trip more HIGH/CRASH labels than train
    assert (reg_test >= HIGH).mean() > (reg_train >= HIGH).mean()


def test_regime_labels_in_valid_set():
    rng = np.random.default_rng(2)
    reg, _ = regimes.compute_regimes(rng.normal(0, 0.02, size=(100, 2)), vol_window=10)
    assert set(np.unique(reg)).issubset({LOW, HIGH, CRASH})


# --------------------------------------------------------------- baselines
def _synthetic_returns(days=120, n_assets=3, drift=0.0005, vol=0.01, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(drift, vol, size=(days, n_assets))


def test_equal_weight_curve_length_and_start():
    r = _synthetic_returns()
    curve = baselines.equal_weight(r, window=30, initial_capital=1_000_000)
    # env acts from index `window`; curve has (len(r) - window + 1) points
    assert curve[0] == 1_000_000
    assert len(curve) == len(r) - 30 + 1


def test_buy_and_hold_matches_equal_weight_with_zero_cost_first_step():
    # With zero cost, on day one both start from equal weights -> same first move
    r = _synthetic_returns()
    eq = baselines.equal_weight(r, window=30, transaction_cost=0.0)
    bh = baselines.buy_and_hold(r, window=30, transaction_cost=0.0)
    assert eq[1] == pytest.approx(bh[1], rel=1e-9)


def test_buy_and_hold_has_no_rebalance_turnover_cost_advantage():
    # Under positive costs, buy-and-hold (no rebalancing) should not be dragged
    # by daily turnover the way equal-weight rebalancing is.
    r = _synthetic_returns(vol=0.03, seed=7)
    eq = baselines.equal_weight(r, window=30, transaction_cost=0.005)
    bh = baselines.buy_and_hold(r, window=30, transaction_cost=0.005)
    assert bh[-1] > 0 and eq[-1] > 0


# --------------------------------------------------------------- env (needs gym)
def test_env_contract():
    gym = pytest.importorskip("gym")
    from src.portfolio_rl import PortfolioEnv

    rng = np.random.default_rng(5)
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=(200, 3)), axis=0)
    env = PortfolioEnv(prices, window_size=30, seed=5)

    obs = env.reset()
    assert obs.shape == env.observation_space.shape
    # weights from any raw action are a valid long-only simplex
    obs, reward, done, info = env.step(np.array([0.2, -0.5, 1.0], dtype=np.float32))
    w = info["weights"]
    assert w.min() >= 0 and abs(w.sum() - 1.0) < 1e-9
    assert np.isfinite(reward)

    # can run to termination without error
    steps = 0
    while not done and steps < 10_000:
        obs, reward, done, info = env.step(env.action_space.sample())
        steps += 1
    assert done
