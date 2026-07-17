"""Price data loading and chronological train/test splitting.

Keeping the yfinance dependency here (out of the environment) means the env and
all metric/baseline logic can be tested with synthetic data and no network.
"""
from __future__ import annotations

import numpy as np


def download_prices(tickers, start, end):
    """Download adjusted close prices. Returns (prices ndarray (T,n), dates).

    Requires network access and yfinance.
    """
    import yfinance as yf

    data = yf.download(
        list(tickers), start=start, end=end, progress=False, auto_adjust=False
    )
    prices = data["Adj Close"].dropna()
    # Preserve ticker column order
    prices = prices[list(tickers)]
    return prices.values, prices.index


def train_test_split(prices, dates, split_date):
    """Split a price series at ``split_date`` (inclusive on the test side).

    A ``window_size``-day overlap is intentionally NOT added here; callers that
    need warm-up context on the test set should pass ``window_size`` extra rows
    when slicing. For this project the test window is long enough that the
    warm-up cost is negligible and a clean date cut keeps the split honest.
    """
    import pandas as pd

    dates = pd.DatetimeIndex(dates)
    split = pd.Timestamp(split_date)
    mask = dates < split
    cut = int(mask.sum())
    return (
        (np.asarray(prices)[:cut], dates[:cut]),
        (np.asarray(prices)[cut:], dates[cut:]),
    )
