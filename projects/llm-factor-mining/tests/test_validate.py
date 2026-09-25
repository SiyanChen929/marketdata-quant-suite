from __future__ import annotations

import json

import pytest

from llm_factor_mining.dsl import (
    ERROR_CODES,
    OPERATORS,
    TERMINALS,
    ArgKind,
    DSLLimits,
    describe_language,
    lookback,
    parse,
    validate,
)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("ts_mean(close", "PARSE"),
        ("foo(close)", "UNKNOWN_OP"),
        ("price * 2", "UNKNOWN_TERMINAL"),
        ("ts_mean(close)", "ARITY"),
        ("add(close)", "ARITY"),
        ("neg(close, open)", "ARITY"),
        ("ts_mean(close, volume)", "TYPE"),
        ("ts_mean(close, 5.0)", "TYPE"),
        ("ts_mean(close, 2.5)", "TYPE"),
        ("power(close, volume)", "TYPE"),
        ("ts_mean(close, 1)", "WINDOW_RANGE"),
        ("ts_mean(close, 253)", "WINDOW_RANGE"),
        ("delay(close, 0)", "WINDOW_RANGE"),
        ("delay(close, -1)", "WINDOW_RANGE"),
        ("delta(close, -5)", "WINDOW_RANGE"),
        ("ts_corr(close, volume, 1)", "WINDOW_RANGE"),
        ("power(close, 5)", "CONST_RANGE"),
        ("close * 1e7", "CONST_RANGE"),
        ("1 + 2", "CONSTANT_ONLY"),
        ("ts_mean(5, 10)", "CONSTANT_ONLY"),
        ("delay(delay(delay(close, 252), 252), 1)", "LOOKBACK"),
    ],
)
def test_error_codes(text: str, code: str) -> None:
    result = validate(text)
    assert not result.ok
    assert code in result.codes
    assert set(result.codes) <= set(ERROR_CODES)


@pytest.mark.parametrize(
    "text",
    [
        "close",
        "delay(close, 1)",
        "delta(close, 1)",
        "ts_mean(close, 2)",
        "ts_mean(close, 252)",
        "power(returns, -4)",
        "power(returns, 0.5)",
        "delay(delay(close, 252), 252)",
        "close - 1",
        "1 / close",
    ],
)
def test_valid_expressions(text: str) -> None:
    result = validate(text)
    assert result.ok, result.errors
    assert result.errors == ()
    assert result.node == parse(text)


def test_parse_error_carries_position() -> None:
    result = validate("ts_mean(close, 5")
    assert result.codes == ("PARSE",)
    assert result.errors[0].position == 16
    assert result.node is None and result.stats is None


def test_depth_and_size_limits() -> None:
    deep = "close"
    for _ in range(10):
        deep = f"neg({deep})"
    result = validate(deep)
    assert "DEPTH" in result.codes and result.stats is not None and result.stats.depth == 11
    wide = " + ".join(["close"] * 25)
    result = validate(wide)
    assert "SIZE" in result.codes and result.stats is not None and result.stats.n_nodes == 49
    assert validate(wide, DSLLimits(max_nodes=60, max_depth=30)).ok


def test_custom_lookback_limit() -> None:
    limits = DSLLimits(max_lookback=30)
    assert validate("ts_mean(ts_mean(close, 16), 16)", limits).ok  # 15 + 15
    assert validate("ts_mean(ts_mean(close, 20), 20)", limits).codes == ("LOOKBACK",)


def test_multiple_errors_are_all_reported_with_paths() -> None:
    result = validate("add(ts_mean(close, 1.5), foo(bar))")
    codes = sorted(result.codes)
    assert codes == ["TYPE", "UNKNOWN_OP", "UNKNOWN_TERMINAL"]
    paths = {issue.code: issue.path for issue in result.errors}
    assert paths["TYPE"] == (0, 1)
    assert paths["UNKNOWN_OP"] == (1,)
    assert paths["UNKNOWN_TERMINAL"] == (1, 0)
    json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("close", 0),
        ("returns", 1),
        ("delay(close, 5)", 5),
        ("delta(close, 5)", 5),
        ("ts_mean(close, 5)", 4),
        ("ts_std(returns, 20)", 20),
        ("ts_corr(delay(close, 3), volume, 10)", 12),
        ("cs_rank(ts_mean(close, 5)) + delay(returns, 10)", 11),
        ("delay(close, 21) / delay(close, 252) - 1", 252),
    ],
)
def test_lookback(text: str, expected: int) -> None:
    assert lookback(parse(text)) == expected


def test_lookahead_is_impossible_by_construction() -> None:
    """No operator can reach forward: windows are positive and shifts backward-only."""

    for spec in OPERATORS.values():
        assert spec.category in {"elementwise", "time_series", "cross_sectional"}
        if spec.window_slot is None:
            assert spec.category != "time_series"
            continue
        assert spec.min_window >= 1
        assert spec.window_lookback_offset in (-1, 0)
        assert spec.lookback(spec.min_window) >= 0
        series = ", ".join("close" for kind in spec.arg_kinds if kind is ArgKind.SERIES)
        for window in (0, -1, -21):
            result = validate(f"{spec.name}({series}, {window})")
            assert "WINDOW_RANGE" in result.codes, spec.name
    assert all(spec.lookback >= 0 for spec in TERMINALS.values())
    assert "forward" not in " ".join(TERMINALS).lower()


def test_limits_validation() -> None:
    with pytest.raises(ValueError):
        DSLLimits(max_window=1)
    with pytest.raises(ValueError):
        DSLLimits(max_depth=0)


def test_grammar_card_lists_everything() -> None:
    card = describe_language()
    for name in [*OPERATORS, *TERMINALS]:
        assert name in card
