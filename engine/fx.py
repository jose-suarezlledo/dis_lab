"""Currency views of a foreign asset: local currency, unhedged, and hedged.

A euro investor who buys a dollar asset can see it three ways:

- **Local (USD):** the asset's own return. What a US investor earns.
- **Unhedged (EUR):** local return plus the EUR/USD move. What `data.convert_currency`
  gives: EUR value = USD value / (USD per EUR).
- **Hedged (EUR):** local return plus the *carry*, i.e. the short-rate difference
  between the two currencies. Selling USD forward for EUR earns the EUR rate and
  pays the USD rate (covered interest parity), so with rates r_EUR and r_USD the
  hedge adds roughly (r_EUR − r_USD) per year. When US rates are above euro rates,
  hedging costs money; when they are below, it pays.

The hedged series is built as `local NAV × cash_EUR / cash_USD`, with each cash
leg accrued from the published short rates (`portfolio.cash_from_rate`). This is a
*synthetic* hedge: a monthly-rolled forward with no transaction costs and a perfect
hedge ratio. Real hedged ETFs (e.g. IWDE.L for MSCI World) include the roll costs
and a small tracking difference; both are in the data snapshot so the two can be
compared.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import EURO_RF_SERIES, FX_SERIES, load_fred
from .metrics import align
from .portfolio import cash_from_rate

__all__ = ["RATE_SERIES", "hedged_levels", "currency_views"]

# Short rate used for each currency's cash leg (FRED ids; both quoted in % per year).
RATE_SERIES = {"USD": "DTB3", "EUR": EURO_RF_SERIES}


def hedged_levels(
    prices: pd.DataFrame | pd.Series,
    currency_from: str = "USD",
    to: str = "EUR",
    rates: dict[str, pd.Series] | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame | pd.Series:
    """Levels of foreign assets as a hedged investor in `to` sees them.

    prices: levels in `currency_from` (one column per asset). The result has the same
    shape, rebased so that it starts at the same value as the input.
    rates: {"USD": series, "EUR": series} in % per year; by default the cached FRED
    series in RATE_SERIES. The euro series is monthly; between prints the last
    published rate is used, which is what an investor would have known.
    """
    if currency_from == to:
        raise ValueError("currency_from and to must differ")
    rates = rates or {}
    idx = prices.index
    r_from = rates.get(currency_from)
    r_to = rates.get(to)
    if r_from is None:
        r_from = load_fred(RATE_SERIES[currency_from], data_dir)
    if r_to is None:
        r_to = load_fred(RATE_SERIES[to], data_dir)
    cash_from = cash_from_rate(r_from, idx)
    cash_to = cash_from_rate(r_to, idx)
    carry = cash_to / cash_from  # growth of 1 unit of `to` lent, minus 1 unit of `from` borrowed
    if isinstance(prices, pd.Series):
        return (prices * carry).rename(prices.name)
    return prices.mul(carry, axis=0)


def currency_views(
    local: pd.Series,
    currency_from: str = "USD",
    to: str = "EUR",
    rates: dict[str, pd.Series] | None = None,
    fx: pd.Series | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """The three views of one foreign asset, aligned on common dates and rebased to 100.

    Columns: "<from> (local)", "<to> unhedged", "<to> hedged". Only USD→EUR (and the
    reverse) are supported, through the Fed's DEXUSEU rate (USD per EUR), unless `fx`
    (units of `currency_from` per unit of `to`) is given.
    """
    if fx is None:
        fx = load_fred(FX_SERIES, data_dir)  # USD per EUR
        if (currency_from, to) == ("EUR", "USD"):
            fx = 1 / fx
        elif (currency_from, to) != ("USD", "EUR"):
            raise ValueError("only USD/EUR views are cached; pass fx for other pairs")
    local = local.dropna()
    unhedged = (local / fx.reindex(local.index)).dropna()
    frame = align(local.rename("local"), unhedged.rename("unhedged"))
    hedged = hedged_levels(frame["local"], currency_from, to, rates, data_dir)
    out = pd.DataFrame({
        f"{currency_from} (local)": frame["local"],
        f"{to} unhedged": frame["unhedged"],
        f"{to} hedged": hedged,
    })
    return out / out.iloc[0] * 100
