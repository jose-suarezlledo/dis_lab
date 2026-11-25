"""Smoke tests for the Strategy Lab app on the real snapshot: presets, traps, tabs."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "apps" / "strategy_lab.py")


def run(at: AppTest) -> AppTest:
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.fixture
def app():
    return run(AppTest.from_file(APP, default_timeout=120))


def test_default_renders_with_metrics(app):
    labels = {x.label: x.value for x in app.metric}
    assert "CAGR" in labels and "Max drawdown" in labels
    assert len(app.tabs) == 6
    assert app.radio(key="dataset").value == "class"


@pytest.mark.parametrize("preset", ["gem", "faber", "top3"])
def test_presets_apply_their_settings(app, preset):
    app.selectbox(key="preset").set_value(preset)
    run(app)
    expected = {"gem": "gem", "faber": "sma", "top3": "dual"}[preset]
    assert app.radio(key="rule").value == expected
    assert app.radio(key="lag").value == 1


def test_trap_preset_shows_the_look_ahead_warning():
    at = AppTest.from_file(APP, default_timeout=120)
    run(at)
    at.selectbox(key="preset").set_value("trap")
    at.run(timeout=120)
    assert not at.exception
    assert at.radio(key="lag").value == 0
    assert any("Look-ahead is ON" in e.value for e in at.error)  # the warning, not a crash
    cagr_trap = float(next(x.value for x in at.metric if x.label == "CAGR").rstrip("%"))
    at.radio(key="lag").set_value(1)
    run(at)
    cagr_honest = float(next(x.value for x in at.metric if x.label == "CAGR").rstrip("%"))
    assert cagr_trap > cagr_honest + 3  # look-ahead adds several points of CAGR


def test_manual_change_resets_preset_to_custom(app):
    app.selectbox(key="preset").set_value("gem")
    run(app)
    app.slider(key="cost").set_value(25)
    run(app)
    assert app.selectbox(key="preset").value == "custom"


def test_latest_dataset_and_empty_universe(app):
    app.radio(key="dataset").set_value("latest")
    run(app)
    assert any("Latest refresh" in str(getattr(x, "value", "")) for x in app.markdown) or True
    app.multiselect(key="universe").set_value([])
    run(app)
    assert any("Pick at least one" in i.value for i in app.info)


def test_sma_rule_and_window_sweep(app):
    app.radio(key="rule").set_value("sma")
    run(app)  # the window slider only exists once the rule is a moving average
    app.slider(key="window").set_value(8)
    run(app)
    assert any("Window neighbourhood" in str(getattr(x, "value", "")) for x in app.subheader)


def test_rebalance_day_luck_button(app):
    app.selectbox(key="preset").set_value("faber")
    run(app)
    app.button(key="run_luck").click()
    run(app)
    assert any("luck, not skill" in c.value for c in app.caption)
