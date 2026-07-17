"""Regime-aware PPO portfolio allocation.

``metrics``, ``regimes``, ``baselines`` and ``data`` are pure (NumPy / stdlib)
and import without gym or torch. ``PortfolioEnv`` is loaded lazily so importing
the metrics/baselines does not require the RL stack to be installed.
"""
from . import baselines, data, metrics, regimes

__all__ = ["PortfolioEnv", "metrics", "regimes", "baselines", "data"]


def __getattr__(name):
    if name == "PortfolioEnv":
        from .env import PortfolioEnv
        return PortfolioEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
