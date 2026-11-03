"""DIS Lab engine: shared data loading and metrics for every DIS app and notebook.

    from engine import data, metrics
    prices = data.load_prices(["SPY", "AGG"])
    metrics.summary_table(prices, benchmark="SPY", rf=0.02)
"""

from . import checks, data, factors, fx, metrics, portfolio, robo, stylized

__version__ = "0.4.0"
__all__ = ["checks", "data", "factors", "fx", "metrics", "portfolio", "robo", "stylized"]
