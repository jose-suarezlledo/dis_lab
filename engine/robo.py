"""The reference robo-advisor: questionnaire → risk profile → model portfolio → backtest, and goals.

This is the glass-box version of what Indexa, inbestMe, Betterment or a bank's
"profiled" portfolio do. Every rule is visible and documented here so that a
student's own AI-built robo can be checked against it.

1. **Questionnaire** (`QUESTIONS`): seven questions in three groups.
   - *Risk tolerance*: how much volatility the investor can stomach (feelings).
   - *Risk capacity*: how much loss the investor can afford (facts: horizon, share
     of wealth, income stability, emergency fund).
   - *Risk need*: how much return the goal requires.
   Each answer scores 1–10; each group is the average of its questions.
2. **Profile** (`risk_profile`): the lower of tolerance and capacity, rounded to 1–10.
   An investor is held to the lower of what they can bear emotionally and what they
   can afford financially. If the *need* is above the profile, the robo says so and
   does not raise the risk: the honest fixes are a bigger contribution, a longer
   horizon or a smaller goal.
3. **Model portfolios** (`MODEL_PORTFOLIOS`): a fixed table, one row per profile,
   with UCITS ETFs a euro investor can buy. Robos do not run an optimiser per client:
   mean-variance weights are unstable to small changes in expected returns, so the
   industry builds a table once and maps clients to it.
4. **Backtest** (`robo_backtest`): the portfolio held at its weights, rebalanced by
   calendar or by drift band, net of trading costs and of the robo's fee layer
   (advisory + custody; fund TERs are already in the prices).
5. **Goals** (`goal_simulation`): the same money run two ways on bootstrapped
   history: one profile-based portfolio for everything, versus one bucket per goal
   (goal-based). Reports how often each goal is met.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import align, to_returns
from .portfolio import PortfolioResult, apply_fee, drift_band_portfolio, fixed_weight_portfolio

__all__ = [
    "Question", "QUESTIONS", "DIMENSIONS", "RiskProfile", "risk_profile",
    "ASSETS", "MODEL_PORTFOLIOS", "model_weights", "equity_share",
    "RoboResult", "robo_backtest", "REBALANCE_CHOICES",
    "Goal", "GoalSimulation", "goal_simulation", "bootstrap_paths", "aggregate_weights",
]

# --------------------------------------------------------------------------- questionnaire

DIMENSIONS = {
    "tolerance": "Risk tolerance: how much volatility you can stomach",
    "capacity": "Risk capacity: how much loss you can afford",
    "need": "Risk need: how much return your goal requires",
}


@dataclass(frozen=True)
class Question:
    id: str
    dimension: str  # key of DIMENSIONS
    text: str
    options: tuple[tuple[str, int], ...]  # (label, score 1-10)


QUESTIONS: tuple[Question, ...] = (
    Question("loss", "tolerance",
             "Your €100,000 portfolio falls to €80,000 in three months. What do you do?",
             (("Sell everything: I cannot sleep", 1), ("Sell part of it", 3),
              ("Nothing: I wait", 7), ("Buy more: it is on sale", 10))),
    Question("swings", "tolerance",
             "Which range of one-year outcomes would you accept, in exchange for a higher expected return?",
             (("−2% to +6%", 1), ("−8% to +12%", 4), ("−18% to +25%", 7), ("−35% to +45%", 10))),
    Question("experience", "tolerance",
             "Your experience with stocks and funds",
             (("None", 2), ("Some, through a bank or a pension plan", 5),
              ("I manage my own investments", 8), ("Professional", 10))),
    Question("horizon", "capacity",
             "When will you need most of this money?",
             (("Within 3 years", 1), ("In 3 to 5 years", 3), ("In 5 to 10 years", 6),
              ("In 10 to 20 years", 8), ("In more than 20 years", 10))),
    Question("share", "capacity",
             "What share of your total wealth (excluding your home) is this money?",
             (("More than 75%", 2), ("50% to 75%", 4), ("25% to 50%", 7), ("Less than 25%", 10))),
    Question("income", "capacity",
             "How stable is your income, and do you have an emergency fund?",
             (("Unstable income, no emergency fund", 1), ("Stable income, no emergency fund", 4),
              ("Stable income, 3 to 6 months of expenses saved", 7),
              ("Stable income, more than 6 months saved", 10))),
    Question("need", "need",
             "What yearly return does your goal require from this money?",
             (("Keep up with inflation, about 2%", 2), ("About 4%", 5), ("About 6%", 8),
              ("8% or more", 10))),
)


@dataclass(frozen=True)
class RiskProfile:
    profile: int  # 1 (most conservative) to 10 (most aggressive)
    tolerance: float  # 1-10
    capacity: float  # 1-10
    need: float  # 1-10
    binding: str  # "tolerance" or "capacity": which one set the profile
    need_exceeds_profile: bool
    note: str


def risk_profile(answers: dict[str, int]) -> RiskProfile:
    """answers: question id -> index of the chosen option (0-based). All questions required."""
    missing = [q.id for q in QUESTIONS if q.id not in answers]
    if missing:
        raise KeyError(f"unanswered questions: {missing}")
    scores: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    for q in QUESTIONS:
        i = answers[q.id]
        if not 0 <= i < len(q.options):
            raise ValueError(f"question {q.id!r}: option index {i} out of range")
        scores[q.dimension].append(q.options[i][1])
    tol = float(np.mean(scores["tolerance"]))
    cap = float(np.mean(scores["capacity"]))
    need = float(np.mean(scores["need"]))
    binding = "capacity" if cap < tol else "tolerance"
    profile = int(min(10, max(1, round(min(tol, cap)))))
    exceeds = need > profile + 0.5
    if exceeds:
        note = (f"Your goal needs more risk (need {need:.0f}) than your {binding} allows (profile {profile}). "
                "The robo does not raise the risk: save more, extend the horizon or lower the goal.")
    else:
        note = f"Profile {profile}, set by your {binding} (tolerance {tol:.1f}, capacity {cap:.1f})."
    return RiskProfile(profile, tol, cap, need, binding, exceeds, note)


# --------------------------------------------------------------------------- model portfolios

# Asset menu: UCITS ETFs in EUR from the data snapshot (ticker -> what it is).
ASSETS = {
    "IWDA.AS": "World equity (MSCI World, unhedged)",
    "IS3N.DE": "Emerging-market equity (unhedged)",
    "EUNH.DE": "Euro government bonds",
    "XG7S.DE": "Global government bonds, hedged to EUR",
    "XEON.DE": "Cash (euro overnight rate)",
}

# Equity share per profile; the rest is bonds, except a cash sleeve in profiles 1-2.
_EQUITY = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50, 6: 0.60, 7: 0.70, 8: 0.80, 9: 0.90, 10: 0.95}
_CASH = {1: 0.10, 2: 0.05}
_WORLD_SHARE = 0.85  # of equity; the rest emerging markets
_EURO_GOV_SHARE = 0.50  # of bonds; the rest global hedged


def _build_table() -> pd.DataFrame:
    rows = {}
    for p, eq in _EQUITY.items():
        cash = _CASH.get(p, 0.0)
        bonds = 1 - eq - cash
        rows[p] = {
            "IWDA.AS": eq * _WORLD_SHARE,
            "IS3N.DE": eq * (1 - _WORLD_SHARE),
            "EUNH.DE": bonds * _EURO_GOV_SHARE,
            "XG7S.DE": bonds * (1 - _EURO_GOV_SHARE),
            "XEON.DE": cash,
        }
    t = pd.DataFrame(rows).T.round(4)
    t.index.name = "profile"
    return t


MODEL_PORTFOLIOS: pd.DataFrame = _build_table()


def model_weights(profile: int, table: pd.DataFrame | None = None) -> dict[str, float]:
    """Target weights of one profile, without the zero positions."""
    table = MODEL_PORTFOLIOS if table is None else table
    if profile not in table.index:
        raise KeyError(f"profile {profile} not in the table (rows {list(table.index)})")
    w = table.loc[profile]
    w = w[w > 0]
    return (w / w.sum()).to_dict()


def equity_share(weights: dict[str, float], equity: tuple[str, ...] = ("IWDA.AS", "IS3N.DE")) -> float:
    return float(sum(v for k, v in weights.items() if k in equity))


# --------------------------------------------------------------------------- backtest

REBALANCE_CHOICES = ("none", "monthly", "quarterly", "yearly", "band")


@dataclass(frozen=True)
class RoboResult:
    nav: pd.Series  # net of trading costs and of the fee layer
    gross: PortfolioResult  # before the fee layer (net of trading costs)
    fee: float  # annual fee layer applied (advisory + custody), as a decimal
    rebalance: str
    band: float | None


def robo_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    fee: float = 0.0053,
    rebalance: str = "monthly",
    band: float = 0.05,
    cost_bps: float = 10.0,
    start_value: float = 100.0,
    name: str = "Robo",
) -> RoboResult:
    """A model portfolio as a client experiences it.

    fee: the robo's own layers per year (Indexa's average is about 0.53% all-in, of which
    about 0.10% is the funds' TER already in the prices; pass the layers *on top* of the
    funds, e.g. 0.0043). rebalance: calendar ("monthly", "quarterly", "yearly"), "none"
    (buy and hold) or "band" (reset only when a weight drifts more than `band`, checked
    monthly). cost_bps: one-way trading cost per unit traded.
    """
    if rebalance not in REBALANCE_CHOICES:
        raise ValueError(f"rebalance must be one of {REBALANCE_CHOICES}")
    if rebalance == "band":
        gross = drift_band_portfolio(prices, weights, band=band, check="monthly", cost_bps=cost_bps,
                                     start_value=start_value, name=name)
    else:
        gross = fixed_weight_portfolio(prices, weights, rebalance=rebalance, cost_bps=cost_bps,
                                       start_value=start_value, name=name)
    nav = apply_fee(gross.nav, fee)
    return RoboResult(nav=nav, gross=gross, fee=fee, rebalance=rebalance,
                      band=band if rebalance == "band" else None)


# --------------------------------------------------------------------------- goals


@dataclass(frozen=True)
class Goal:
    name: str
    amount: float  # money set aside for it today
    target: float  # what it must be worth at the horizon
    years: float
    weights: dict[str, float]  # the bucket's portfolio (goal-based approach)
    contribution: float = 0.0  # added every month to this bucket


@dataclass(frozen=True)
class GoalSimulation:
    table: pd.DataFrame  # rows: (approach, goal); columns: p_met, median_wealth, p5_wealth, target
    p_all_met: dict[str, float]  # approach -> probability every goal is met
    equity_at_start: dict[str, float]  # approach -> equity share of the total money at the start
    n_sims: int
    months: int
    block: int


def aggregate_weights(goals: list[Goal]) -> dict[str, float]:
    """Weights of the whole pot when each goal holds its own bucket (money-weighted)."""
    total = sum(g.amount for g in goals)
    if total <= 0:
        raise ValueError("the goals hold no money")
    agg: dict[str, float] = {}
    for g in goals:
        for k, w in g.weights.items():
            agg[k] = agg.get(k, 0.0) + w * g.amount / total
    return agg


def bootstrap_paths(
    monthly_returns: pd.DataFrame, months: int, n_sims: int, block: int = 12, seed: int = 0
) -> np.ndarray:
    """Joint block bootstrap of monthly returns: shape (n_sims, months, n_assets).

    Blocks of `block` consecutive months are drawn with replacement (circularly), so
    that correlations between assets and some of the persistence of volatility
    survive. Returns are the historical ones: no expected-return assumption is made.
    """
    r = monthly_returns.dropna().to_numpy()
    n, k = r.shape
    if n < block:
        raise ValueError("history shorter than one block")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(months / block))
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n  # circular
    paths = r[idx.reshape(n_sims, -1)][:, :months, :]
    return paths


def _bucket_wealth(paths: np.ndarray, assets: list[str], weights: dict[str, float], amount: float,
                   contribution: float, months: int, fee: float) -> np.ndarray:
    w = np.array([weights.get(a, 0.0) for a in assets])
    if abs(w.sum() - 1) > 1e-6:
        raise ValueError("bucket weights must sum to 1")
    port = paths[:, :months, :] @ w  # monthly bucket returns (rebalanced monthly, no cost)
    net = (1 + port) * (1 - fee / 12)
    wealth = np.full(paths.shape[0], float(amount))
    out = np.empty((paths.shape[0], months + 1))
    out[:, 0] = wealth
    for t in range(months):
        wealth = wealth * net[:, t] + contribution
        out[:, t + 1] = wealth
    return out


def goal_simulation(
    monthly_returns: pd.DataFrame,
    goals: list[Goal],
    profile_weights: dict[str, float],
    n_sims: int = 2000,
    block: int = 12,
    fee: float = 0.0,
    seed: int = 0,
) -> GoalSimulation:
    """Profile-based (one portfolio for all goals) versus goal-based (one bucket per goal).

    Both approaches start with the same money (the sum of the goals' amounts), receive
    the same monthly contributions, pay the same fee, and face the same bootstrapped
    return paths. In the profile-based run the whole pot sits in `profile_weights`; at
    each goal's horizon the goal is met if the pot is worth at least the target, and the
    target is then withdrawn. In the goal-based run each bucket is judged on its own.
    """
    if not goals:
        raise ValueError("no goals")
    assets = list(monthly_returns.columns)
    for g in goals:
        unknown = set(g.weights) - set(assets)
        if unknown:
            raise KeyError(f"goal {g.name!r}: no returns for {sorted(unknown)}")
    months = int(round(max(g.years for g in goals) * 12))
    paths = bootstrap_paths(monthly_returns, months, n_sims, block, seed)
    total_amount = sum(g.amount for g in goals)
    total_contrib = sum(g.contribution for g in goals)
    rows = []

    # Goal-based: one bucket per goal.
    met_all_goal = np.ones(n_sims, dtype=bool)
    for g in goals:
        m = int(round(g.years * 12))
        wealth = _bucket_wealth(paths, assets, g.weights, g.amount, g.contribution, m, fee)[:, -1]
        met = wealth >= g.target
        met_all_goal &= met
        rows.append(("Goal-based", g.name, met.mean(), np.median(wealth), np.percentile(wealth, 5), g.target))

    # Profile-based: one pot, goals withdrawn as they come due.
    pot = _bucket_wealth(paths, assets, profile_weights, total_amount, total_contrib, months, fee)
    pot = pot.copy()
    met_all_prof = np.ones(n_sims, dtype=bool)
    for g in sorted(goals, key=lambda g: g.years):
        m = int(round(g.years * 12))
        wealth = pot[:, m]
        met = wealth >= g.target
        met_all_prof &= met
        rows.append(("Profile-based", g.name, met.mean(), np.median(wealth), np.percentile(wealth, 5), g.target))
        taken = np.minimum(wealth, g.target)
        # Withdraw the target (or what is there): scale the rest of the path down proportionally.
        remaining = np.where(wealth > 0, (wealth - taken) / np.where(wealth > 0, wealth, 1), 0.0)
        pot[:, m:] = pot[:, m:] * remaining[:, None]

    table = pd.DataFrame(rows, columns=["approach", "goal", "p_met", "median_wealth", "p5_wealth", "target"])
    table = table.set_index(["approach", "goal"])
    return GoalSimulation(
        table=table,
        p_all_met={"Goal-based": float(met_all_goal.mean()), "Profile-based": float(met_all_prof.mean())},
        equity_at_start={"Goal-based": equity_share(aggregate_weights(goals)),
                         "Profile-based": equity_share(profile_weights)},
        n_sims=n_sims, months=months, block=block,
    )


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total returns of the asset menu, on common dates (helper for goal_simulation)."""
    lv = align(*[prices[c] for c in prices.columns]).resample("ME").last()
    return to_returns(lv).dropna()
