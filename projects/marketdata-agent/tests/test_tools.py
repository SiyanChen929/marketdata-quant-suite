"""Tool outputs versus independent numpy/pandas computations on synthetic data."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from marketdata_agent import FrameBarSource, Policy, ToolRuntime
from marketdata_agent.analytics import max_drawdown as analytic_drawdown

from conftest import AS_OF, LATE_SYMBOL, brute_force_drawdown, closes


def _ok(runtime: ToolRuntime, tool: str, args: dict):
    outcome = runtime.call_tool(tool, args, tool_use_id="toolu_test")
    assert not outcome.is_error, outcome.content
    return outcome.result


def test_period_return_matches_close_to_close(runtime, panel):
    result = _ok(runtime, "period_return", {"symbol": "SYN02", "start": "2023-01-01", "end": "2023-06-30"})
    series = closes(panel, "SYN02", "2023-01-01", "2023-06-30")
    expected = series.iloc[-1] / series.iloc[0] - 1.0
    assert result.payload["simple_return"] == pytest.approx(expected, rel=1e-9)
    assert result.payload["log_return"] == pytest.approx(math.log(series.iloc[-1] / series.iloc[0]), rel=1e-9)
    assert result.payload["start_date"] == "2023-01-02"  # first weekday on/after start
    assert result.payload["end_date"] == "2023-06-30"
    assert result.provenance.row_count == len(series)
    assert result.provenance.outputs["simple_return"] == result.payload["simple_return"]


@pytest.mark.parametrize("window", [2, 20, 63, 252])
def test_realized_volatility_matches_pandas(runtime, panel, window):
    result = _ok(runtime, "realized_volatility", {"symbol": "SYN03", "window": window, "end": "2023-05-31"})
    series = closes(panel, "SYN03", end="2023-05-31")
    expected = np.log(series).diff().dropna().tail(window).std(ddof=1) * math.sqrt(252)
    assert result.payload["annualized_volatility"] == pytest.approx(expected, rel=1e-9)
    assert result.payload["observations"] == window
    assert result.payload["end_date"] == "2023-05-31"
    assert result.payload["window_start_date"] == series.index[-(window + 1)].date().isoformat()
    assert result.provenance.row_count == window + 1


def test_latest_and_explicit_as_of_give_the_same_result(runtime):
    latest = _ok(runtime, "realized_volatility", {"symbol": "SYN01", "window": 20, "end": "latest"})
    explicit = _ok(runtime, "realized_volatility", {"symbol": "SYN01", "window": 20, "end": AS_OF.isoformat()})
    assert latest.result_id == explicit.result_id
    assert latest.payload == explicit.payload
    assert latest.provenance.args["end"] == AS_OF.isoformat()


def test_rolling_correlation_matches_pandas_on_common_dates(panel):
    gappy = panel.drop(panel.index[(panel["symbol"] == "SYN02") & panel["date"].isin(pd.bdate_range("2023-06-05", "2023-06-09"))])
    runtime = ToolRuntime.for_source(FrameBarSource(gappy), AS_OF)
    window = 40
    result = _ok(runtime, "rolling_correlation", {"a": "SYN01", "b": "syn02", "window": window, "end": "latest"})
    wide = gappy.pivot(index="date", columns="symbol", values="close")[["SYN01", "SYN02"]]
    wide = wide.loc[wide.index <= pd.Timestamp(AS_OF)].dropna().tail(window + 1)
    returns = np.log(wide).diff().dropna()
    expected = returns["SYN01"].corr(returns["SYN02"])
    assert result.payload["correlation"] == pytest.approx(expected, rel=1e-9)
    assert result.payload["b"] == "SYN02"
    assert result.provenance.row_count == 2 * (window + 1)


def test_max_drawdown_matches_brute_force(runtime, panel):
    result = _ok(runtime, "max_drawdown", {"symbol": "SYN04", "start": "2022-01-01", "end": "2023-06-30"})
    series = closes(panel, "SYN04", "2022-01-01", "2023-06-30")
    expected = brute_force_drawdown(series.to_numpy())
    assert result.payload["max_drawdown"] == pytest.approx(expected, rel=1e-9)
    peak, trough = result.payload["peak_close"], result.payload["trough_close"]
    assert trough / peak - 1.0 == pytest.approx(expected, rel=1e-8)
    assert result.payload["peak_date"] <= result.payload["trough_date"]
    recovery = result.payload["recovery_date"]
    after = series.loc[series.index > pd.Timestamp(result.payload["trough_date"])]
    if recovery is None:
        assert (after < peak).all()
    else:
        assert series.loc[pd.Timestamp(recovery)] >= peak


def test_drawdown_edge_cases():
    assert analytic_drawdown([1.0, 2.0, 3.0]).max_drawdown == 0.0
    recovered = analytic_drawdown([10.0, 8.0, 9.0, 10.0, 7.0])
    assert recovered.max_drawdown == pytest.approx(-0.3)
    assert (recovered.peak_index, recovered.trough_index, recovered.recovery_index) == (0, 4, None)
    shallow = analytic_drawdown([10.0, 8.0, 11.0])
    assert (shallow.peak_index, shallow.trough_index, shallow.recovery_index) == (0, 1, 2)


def test_compare_returns_matches_independent_ranking(runtime, panel):
    symbols = ["SYN01", "SYN02", "SYN03", "SYN05"]
    result = _ok(runtime, "compare_returns", {"symbols": symbols, "start": "2023-01-02", "end": "latest"})
    expected = {s: closes(panel, s, "2023-01-02", "2023-06-30") for s in symbols}
    expected_returns = {s: c.iloc[-1] / c.iloc[0] - 1.0 for s, c in expected.items()}
    order = sorted(symbols, key=lambda s: -expected_returns[s])
    assert [entry["symbol"] for entry in result.payload["ranking"]] == order
    for entry in result.payload["ranking"]:
        assert entry["simple_return"] == pytest.approx(expected_returns[entry["symbol"]], rel=1e-9)
    assert result.payload["insufficient_data"] == []


def test_compare_returns_lists_symbols_without_two_closes(runtime):
    result = _ok(runtime, "compare_returns", {"symbols": ["SYN01", LATE_SYMBOL], "start": "2023-02-01", "end": "2023-03-01"})
    assert result.payload["insufficient_data"] == [LATE_SYMBOL]
    assert [entry["symbol"] for entry in result.payload["ranking"]] == ["SYN01"]
    assert set(result.provenance.symbols) == {"SYN01"}


def test_get_daily_bars_truncates_to_most_recent_rows(panel, source):
    runtime = ToolRuntime.for_source(source, AS_OF, policy=Policy(max_rows_per_result=5))
    result = _ok(runtime, "get_daily_bars", {"symbol": "SYN01", "start": "2023-06-01", "end": "latest"})
    series = closes(panel, "SYN01", "2023-06-01", "2023-06-30")
    payload = result.payload
    assert payload["truncated"] is True
    assert payload["rows_total"] == len(series)
    assert payload["rows_returned"] == 5
    assert [row["date"] for row in payload["rows"]] == [d.date().isoformat() for d in series.index[-5:]]
    assert payload["summary"]["period_return"] == pytest.approx(series.iloc[-1] / series.iloc[0] - 1.0, rel=1e-9)
    assert result.provenance.row_count == len(series)  # provenance covers every row used for the summary
    assert result.provenance.outputs[f"close[{payload['rows'][-1]['date']}]"] == pytest.approx(series.iloc[-1])
    short = _ok(runtime, "get_daily_bars", {"symbol": "SYN01", "start": "2023-06-27", "end": "2023-06-29"})
    assert short.payload["truncated"] is False and short.payload["rows_returned"] == 3


def test_list_symbols_reports_only_symbols_known_at_as_of(source):
    early = ToolRuntime.for_source(source, date(2023, 2, 15))
    result = _ok(early, "list_symbols", {})
    names = [row["symbol"] for row in result.payload["symbols"]]
    assert LATE_SYMBOL not in names and len(names) == 5
    assert result.payload["latest_session"] == "2023-02-15"
    later = _ok(ToolRuntime.for_source(source, AS_OF), "list_symbols", {})
    assert later.payload["count"] == 6


@pytest.mark.parametrize(
    ("tool", "args", "code"),
    [
        ("realized_volatility", {"symbol": "SYN01", "window": 700, "end": "2021-06-30"}, "insufficient_data"),
        ("period_return", {"symbol": "NOPE", "start": "2023-01-02", "end": "2023-06-30"}, "unknown_symbol"),
        ("period_return", {"symbol": "SYN01", "start": "2023-06-30", "end": "2023-06-30"}, "insufficient_data"),
        ("rolling_correlation", {"a": "SYN01", "b": "syn01", "window": 20, "end": "latest"}, "invalid_arguments"),
        ("compare_returns", {"symbols": ["SYN01", "syn01"], "start": "2023-01-02", "end": "latest"}, "invalid_arguments"),
        ("period_return", {"symbol": LATE_SYMBOL, "start": "2023-01-02", "end": "2023-02-15"}, "insufficient_data"),
        ("compare_returns", {"symbols": ["SYN01", "SYN02"], "start": "2023-03-01", "end": "2023-03-01"}, "insufficient_data"),
        ("get_daily_bars", {"symbol": "SYN01", "start": "2023-07-01", "end": "latest"}, "lookahead_violation"),
    ],
)
def test_tool_level_failures_become_coded_errors(runtime, tool, args, code):
    outcome = runtime.call_tool(tool, args, tool_use_id="toolu_x")
    assert outcome.is_error
    assert str(outcome.error_code) == code
    assert outcome.content.startswith(f"[error:{code}]")


def test_start_after_resolved_latest_is_rejected_by_the_handler(source):
    """Weekend as-of: the gate sees start <= as_of, but 'latest' resolves to the prior Friday."""

    runtime = ToolRuntime.for_source(source, date(2023, 7, 2))
    outcome = runtime.call_tool("get_daily_bars", {"symbol": "SYN01", "start": "2023-07-01", "end": "latest"})
    assert outcome.decision.allowed and outcome.error_code is not None
    assert str(outcome.error_code) == "invalid_arguments"
