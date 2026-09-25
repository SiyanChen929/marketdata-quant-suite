from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from marketdata_agent import FrameBarSource, ToolRuntime, canonical_json, hash_bars, result_id
from marketdata_agent.provenance import RESULT_ID_LENGTH, json_safe

from conftest import AS_OF


BATTERY = [
    ("list_symbols", {}),
    ("get_daily_bars", {"symbol": "SYN01", "start": "2023-05-01", "end": "latest"}),
    ("period_return", {"symbol": "SYN02", "start": "2023-01-02", "end": "2023-06-30"}),
    ("realized_volatility", {"symbol": "SYN03", "window": 63, "end": "latest"}),
    ("rolling_correlation", {"a": "SYN01", "b": "SYN04", "window": 120, "end": "2023-06-30"}),
    ("max_drawdown", {"symbol": "SYN05", "start": "2022-01-03", "end": "latest"}),
    ("compare_returns", {"symbols": ["SYN01", "SYN02", "SYN06"], "start": "2023-03-01", "end": "latest"}),
    ("propose_order", {"symbol": "SYN01", "side": "buy", "quantity": 10, "rationale": "test"}),
]


def _run_battery(frame: pd.DataFrame) -> list:
    runtime = ToolRuntime.for_source(FrameBarSource(frame), AS_OF)
    outcomes = [runtime.call_tool(tool, dict(args), tool_use_id=f"t{i}") for i, (tool, args) in enumerate(BATTERY)]
    assert not any(o.is_error for o in outcomes), [o.content for o in outcomes if o.is_error]
    return [o.result for o in outcomes]


def test_hash_bars_is_order_invariant_and_value_sensitive(panel):
    subset = panel.loc[panel["symbol"].isin(["SYN01", "SYN02"])].head(200)
    shuffled = subset.sample(frac=1.0, random_state=3)
    assert hash_bars(subset) == hash_bars(shuffled)
    bumped = subset.copy()
    bumped.loc[bumped.index[17], "close"] += 1e-9
    assert hash_bars(bumped) != hash_bars(subset)
    relabelled = subset.copy()
    relabelled["source"] = "other"
    assert hash_bars(relabelled) != hash_bars(subset)
    with pytest.raises(ValueError):
        hash_bars(subset.drop(columns="volume"))


def test_canonical_json_is_sorted_compact_and_nan_free():
    text = canonical_json({"b": 1.5, "a": [np.float64(2.0), float("nan"), pd.Timestamp("2023-06-30")], "c": np.bool_(True)})
    assert text == '{"a":[2.0,null,"2023-06-30"],"b":1.5,"c":true}'
    assert json.loads(text)["a"][1] is None
    assert json_safe({1, 3, 2}) == [1, 2, 3]


def test_result_ids_are_short_stable_and_reproducible_across_runtimes(panel):
    first, second = _run_battery(panel), _run_battery(panel.copy())
    for a, b in zip(first, second):
        assert a.result_id == b.result_id
        assert a.provenance.data_sha256 == b.provenance.data_sha256
        assert a.text == b.text
        assert re.fullmatch(rf"[0-9a-f]{{{RESULT_ID_LENGTH}}}", a.result_id)
        assert a.text.startswith(f"[r:{a.result_id}] ")
        assert a.provenance.finality == "confirmed"
        assert a.provenance.result_id == result_id(a.tool, a.provenance.args, AS_OF, a.provenance.data_sha256)
    assert len({r.result_id for r in first}) == len(first)


def test_results_do_not_depend_on_data_after_as_of(panel):
    """Rewriting every bar after as_of must leave every tool output bit-identical."""

    future = panel["date"] > pd.Timestamp(AS_OF)
    altered = panel.copy()
    scale = np.random.default_rng(99).uniform(0.5, 1.5, int(future.sum()))
    for column in ("open", "high", "low", "close"):
        altered.loc[future, column] = altered.loc[future, column] * scale
    altered.loc[future, "volume"] = 1.0
    assert hash_bars(altered) != hash_bars(panel)
    for original, perturbed in zip(_run_battery(panel), _run_battery(altered)):
        assert original.text == perturbed.text
        assert original.provenance.outputs == perturbed.provenance.outputs


def test_results_change_when_used_data_change(panel):
    in_range = (panel["symbol"] == "SYN02") & (panel["date"] == pd.Timestamp("2023-03-15"))
    altered = panel.copy()
    altered.loc[in_range, ["close", "high"]] = altered.loc[in_range, ["close", "high"]] * 1.01
    original, perturbed = _run_battery(panel)[2], _run_battery(altered)[2]
    assert original.payload == perturbed.payload  # an interior close does not move a close-to-close return
    assert original.provenance.data_sha256 != perturbed.provenance.data_sha256
    assert original.result_id != perturbed.result_id


def test_data_sha256_is_the_hash_of_exactly_the_rows_used(panel):
    result = _run_battery(panel)[2]
    rows = panel.loc[
        (panel["symbol"] == "SYN02") & panel["date"].between(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-06-30"))
    ]
    assert result.provenance.data_sha256 == hash_bars(rows)
    assert result.provenance.row_count == len(rows)
    assert result.provenance.first_date == "2023-01-02" and result.provenance.last_date == "2023-06-30"
    assert result.provenance.symbols == ("SYN02",)
    assert result.provenance.sources == ("synthetic",)
    vol = _run_battery(panel)[3]
    window_rows = panel.loc[(panel["symbol"] == "SYN03") & (panel["date"] <= pd.Timestamp(AS_OF))].tail(64)
    assert vol.provenance.data_sha256 == hash_bars(window_rows)


def test_provenance_serializes_to_json(panel):
    for result in _run_battery(panel):
        payload = result.to_dict()
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert set(payload["provenance"]) >= {
            "result_id",
            "tool",
            "args",
            "as_of",
            "data_sha256",
            "finality",
            "row_count",
            "outputs",
        }
        assert all(isinstance(v, float) for v in payload["provenance"]["outputs"].values())
