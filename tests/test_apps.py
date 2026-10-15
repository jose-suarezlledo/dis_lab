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
    assert len(app.tabs) == 7


@pytest.mark.parametrize("index", range(5))
def test_every_scenario_renders(app, index):
    sb = app.selectbox(key="scenario")
    sb.set_value(sb.options[index])
    run(app)
    assert not app.error, [e.value for e in app.error]


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
