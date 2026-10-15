"""DIS Lab engine: shared data loading and metrics for every DIS app and notebook.

    from engine import data, metrics
    prices = data.load_prices(["SPY", "AGG"])
    metrics.summary_table(prices, benchmark="SPY", rf=0.02)
"""

from . import checks, data, metrics, portfolio, stylized

__version__ = "0.2.0"
__all__ = ["checks", "data", "metrics", "portfolio", "stylized"]
