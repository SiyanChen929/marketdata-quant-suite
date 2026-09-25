"""The independent reference agrees with the tools, and with brute-force and pandas formulations."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from marketdata_agent import FrameBarSource, ToolRuntime
from marketdata_agent.bench import DEFAULT_SEED, generate_suite
from marketdata_agent.bench import reference as ref
from marketdata_agent.bench.generator import DEFAULT_DATASET, dataset_frame

from conftest import brute_force_drawdown


@pytest.fixture(scope="module")
def frame():
    return dataset_frame(DEFAULT_DATASET)


@pytest.fixture(scope="module")
def raw(frame):
    return ref.RawPanel.from_frame(frame)


@pytest.fixture(scope="module")
def suite():
    return generate_suite(DEFAULT_SEED)


def _tool_value(runtime: ToolRuntime, task):
    """Run the reference plan through the real tools and read the answer the task asks for."""

    form = task.answer_form
    outcomes = [runtime.call_tool(step.tool, dict(step.args), tool_use_id=f"t{i}") for i, step in enumerate(task.reference_plan)]
    assert all(not o.is_error for o in outcomes), [o.content for o in outcomes]
    payloads = [o.result.payload for o in outcomes]
    if form.kind == "scalar":
        payload = payloads[0]
        return payload.get(form.key, payload.get("summary", {}).get(form.key))
    if form.kind == "ranking":
        return tuple(entry["symbol"] for entry in payloads[0]["ranking"])
    values = {p["symbol"]: p[form.key] for p in payloads}
    return (ref.top_by(values, form.direction),)


def test_reference_ground_truth_agrees_with_the_tools_on_every_answerable_task(frame, suite):
    source = FrameBarSource(frame)
    checked = 0
    for task in suite:
        if task.expected.kind not in {"numeric", "ranking"}:
            continue
        value = _tool_value(ToolRuntime.for_source(source, task.as_of), task)
        if task.expected.kind == "numeric":
            assert value == pytest.approx(task.expected.value, rel=1e-8, abs=1e-12), task.id
        else:
            assert value == task.expected.value, task.id
        checked += 1
    assert checked >= 110


def test_hindsight_values_agree_with_the_tools_run_without_a_cutoff(frame, suite):
    source = FrameBarSource(frame)
    for task in suite:
        if task.category != "pit_trap":
            continue
        runtime = ToolRuntime.for_source(source, date(9999, 12, 31))
        (step,) = task.naive_plan
        outcome = runtime.call_tool(step.tool, dict(step.args), tool_use_id="t")
        payload = outcome.result.payload
        value = payload.get(task.answer_form.key, payload.get("summary", {}).get(task.answer_form.key))
        assert value == pytest.approx(task.hindsight.value, rel=1e-8), task.id


@pytest.mark.parametrize("symbol", ["SYN01", "SYN04", "SYN09"])
def test_drawdown_matches_the_quadratic_definition(raw, frame, symbol):
    for start, end in [("2021-01-04", "2021-12-31"), ("2022-03-01", "2023-06-30"), ("2023-01-02", "2023-12-29")]:
        closes = frame.loc[(frame["symbol"] == symbol) & frame["date"].between(start, end)].sort_values("date")["close"]
        if len(closes) < 2:
            continue
        assert ref.max_drawdown(raw, symbol, start, end) == pytest.approx(brute_force_drawdown(closes.to_numpy()), abs=1e-15)


def test_volatility_and_correlation_match_pandas(raw, frame):
    wide = frame.pivot(index="date", columns="symbol", values="close")
    for window, end in [(20, "2022-06-30"), (63, "2023-03-31"), (252, "2023-06-30")]:
        closes = wide["SYN03"].loc[:end].dropna()
        expected = np.log(closes).diff().dropna().tail(window).std(ddof=1) * math.sqrt(252)
        assert ref.realized_volatility(raw, "SYN03", window, end) == pytest.approx(expected, rel=1e-12)
        pair = wide[["SYN02", "SYN09"]].loc[:end].dropna().tail(window + 1)
        returns = np.log(pair).diff().dropna()
        if len(returns) == window:
            assert ref.correlation(raw, "SYN02", "SYN09", window, end) == pytest.approx(
                returns["SYN02"].corr(returns["SYN09"]), rel=1e-10
            )


def test_period_return_uses_first_session_on_or_after_start(raw, frame):
    closes = frame.loc[frame["symbol"] == "SYN05"].set_index("date")["close"]
    # 2022-12-31 is a Saturday: the first session on or after it is 2023-01-02.
    expected = closes.loc[pd.Timestamp("2023-03-31")] / closes.loc[pd.Timestamp("2023-01-02")] - 1.0
    assert ref.period_return(raw, "SYN05", "2022-12-31", "2023-03-31") == pytest.approx(expected, rel=1e-14)


def test_universe_and_latest_session_apply_the_cutoff(raw):
    assert "SYN09" not in raw.universe("2022-08-31") and "SYN09" in raw.universe("2022-09-01")
    assert "SYN10" not in raw.universe("2023-06-30")
    assert raw.latest_session("2023-07-02") == "2023-06-30"  # weekend cutoff
    with pytest.raises(ValueError):
        ref.close_on(raw, "SYN01", "2023-07-01")  # no bar on a Saturday
