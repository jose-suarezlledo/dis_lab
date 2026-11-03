"""Smoke tests: every app runs end to end on the real snapshot, in every scenario."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "apps" / "performance_analyzer.py")


def run(at: AppTest) -> AppTest:
    at.run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def app():
    return run(AppTest.from_file(APP, default_timeout=60))


def test_default_scenario_renders(app):
    assert not app.error
    assert any("Common window" in md.value for md in app.markdown)
    assert len(app.tabs) == 8
    assert app.radio(key="session").value == "Session 1"


@pytest.mark.parametrize("session,index", [("Session 1", i) for i in range(5)] + [("Session 2", i) for i in range(6)] + [("Session 3", i) for i in range(4)])
def test_every_scenario_renders(app, session, index):
    if session != app.radio(key="session").value:
        app.radio(key="session").set_value(session)
        run(app)
    sb = app.selectbox(key="scenario")
    sb.set_value(sb.options[index])
    run(app)
    assert not app.error, [e.value for e in app.error]
    assert not app.warning or all("Fewer than" not in w.value for w in app.warning)


def test_session_link_opens_session_2():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["session"] = "2"
    run(at)
    assert at.radio(key="session").value == "Session 2"
    assert at.selectbox(key="scenario").value.startswith("Claim: equal weight")


def test_correlation_toggle_shows_the_price_level_bug():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["session"] = "2"
    run(at)
    at.selectbox(key="scenario").set_value("Claim: factors beat the market")
    run(at)
    cap = " ".join(c.value for c in at.caption if "Average pairwise" in c.value)
    on_ret = float(cap.split("**")[1].split(" on returns")[0])
    on_lvl = float(cap.split(" vs ")[1].split(" on price levels")[0])
    assert on_lvl > on_ret  # trending levels inflate correlation
    at.radio(key="corr_on").set_value("levels")
    run(at)
    assert not at.error


@pytest.mark.parametrize("currency", ["EUR", "USD", "As listed"])
@pytest.mark.parametrize("freq", ["daily", "weekly", "monthly"])
def test_currency_and_frequency(app, currency, freq):
    app.radio(key="currency").set_value(currency)
    app.radio(key="freq").set_value(freq)
    run(app)
    assert not app.error, [e.value for e in app.error]


def test_empty_selection_stops_politely(app):
    app.toggle(key="pf_on").set_value(False)
    app.multiselect(key="assets").set_value([])
    run(app)
    assert any("Pick at least one" in i.value for i in app.info)


def test_session_link_opens_session_3_with_indexa_profiles():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["session"] = "3"
    run(at)
    assert at.radio(key="session").value == "Session 3"
    assert at.selectbox(key="benchmark").value == "INDEXA-6"
    assert any("Published by the provider" in w.value for w in at.info)


# --------------------------------------------------------------------------- robo simulator

ROBO = str(Path(__file__).resolve().parents[1] / "apps" / "robo_simulator.py")


@pytest.fixture
def robo_app():
    return run(AppTest.from_file(ROBO, default_timeout=120))


def test_robo_default_renders(robo_app):
    at = robo_app
    assert not at.error, [e.value for e in at.error]
    labels = {x.label: x.value for x in at.metric}
    assert labels["Your profile"].endswith("/ 10")
    assert "Rebalances" in labels and "All goals met: goal-based" in labels


def test_robo_override_profile_and_excel_way(robo_app):
    at = robo_app
    at.selectbox(key="profile_override").set_value(10)
    at.toggle(key="excel_way").set_value(True)
    run(at)
    assert not at.error, [e.value for e in at.error]
    assert any("The Excel way" in w.value for w in at.warning)
    assert at.selectbox(key="bench").value == "Indexa 10"


@pytest.mark.parametrize("rebalance", ["none", "monthly", "yearly"])
def test_robo_rebalance_modes(robo_app, rebalance):
    at = robo_app
    at.selectbox(key="rebalance").set_value(rebalance)
    run(at)
    assert not at.error, [e.value for e in at.error]


def test_robo_long_us_history_and_other_benchmark(robo_app):
    at = robo_app
    at.radio(key="history").set_value("Long US history (1987–2026)")
    at.selectbox(key="bench").set_value("V60A.DE")
    run(at)
    assert not at.error, [e.value for e in at.error]
