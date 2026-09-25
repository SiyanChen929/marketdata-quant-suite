from __future__ import annotations

from pathlib import Path
import re

import pytest

from llm_factor_mining.dsl import (
    Call,
    Constant,
    ParseError,
    Terminal,
    canonical_string,
    canonicalize,
    parse,
    to_expression,
    tokenize,
)
from llm_factor_mining.dsl.library import LIBRARY


ROUND_TRIP = [
    "close",
    "-close",
    "close + open * 2",
    "(close - open) / (high - low)",
    "ts_mean(close, 20) / ts_mean(close, 5) - 1",
    "cs_rank(-ts_std(returns, 20))",
    "power(returns, 0.5) + power(returns, -1.5)",
    "ts_corr(cs_rank(delta(log(volume), 2)), cs_rank((close - open) / open), 6)",
    "decay_linear(ts_zscore(vwap_proxy, 10), 5) * sign(delta(close, 1))",
    "1e-3 * dollar_volume + .5 - 2.",
]


@pytest.mark.parametrize("text", ROUND_TRIP + [factor.expression for factor in LIBRARY])
def test_faithful_round_trip(text: str) -> None:
    node = parse(text)
    assert parse(to_expression(node)) == node


@pytest.mark.parametrize("text", ROUND_TRIP + [factor.expression for factor in LIBRARY])
def test_canonical_round_trip(text: str) -> None:
    canonical = canonicalize(parse(text))
    assert canonicalize(parse(canonical_string(canonical))) == canonical
    assert canonical_string(parse(canonical_string(canonical))) == canonical_string(canonical)


def test_precedence_and_left_associativity() -> None:
    a, b, c = Terminal("open"), Terminal("high"), Terminal("low")
    assert parse("open + high * low") == Call("add", (a, Call("mul", (b, c))))
    assert parse("open - high - low") == Call("sub", (Call("sub", (a, b)), c))
    assert parse("open / high / low") == Call("div", (Call("div", (a, b)), c))
    assert parse("(open + high) * low") == Call("mul", (Call("add", (a, b)), c))
    assert parse("open - -high") == Call("sub", (a, Call("neg", (b,))))


def test_unary_minus_and_literals() -> None:
    assert parse("-5") == Constant(-5.0, True)
    assert parse("- -5") == Constant(5.0, True)
    assert parse("-(2.5)") == Constant(-2.5, False)
    assert parse("--close") == Call("neg", (Call("neg", (Terminal("close"),)),))
    assert parse("-5 * close") == Call("mul", (Constant(-5.0, True), Terminal("close")))
    assert parse("7") == Constant(7.0, True)
    for text in (".5", "5.", "1e-3", "2E2", "3.0"):
        node = parse(text)
        assert isinstance(node, Constant) and not node.integer
    assert parse("0") == parse("-0")


def test_identifiers_are_case_insensitive() -> None:
    assert parse("TS_MEAN(Close, 5)") == parse("ts_mean(close, 5)")


def test_function_call_arguments() -> None:
    assert parse("f()") == Call("f", ())
    node = parse("ts_corr(close, volume, 10)")
    assert node == Call("ts_corr", (Terminal("close"), Terminal("volume"), Constant(10.0, True)))


@pytest.mark.parametrize(
    ("text", "position", "fragment"),
    [
        ("", 0, "empty expression"),
        ("   ", 0, "empty expression"),
        ("ts_mean(close, 5", 16, "expected ',' or ')'"),
        ("close $ open", 6, "unexpected character '$'"),
        ("close open", 6, "unexpected token 'open'"),
        ("ts_mean(close,,5)", 14, "expected a number"),
        ("ts_mean(close, 5,)", 17, "expected a number"),
        ("1.2.3", 0, "malformed number"),
        ("5d", 0, "malformed number"),
        ("(close", 6, "expected ')'"),
        (")", 0, "expected a number"),
        ("close +", 7, "end of input"),
        ("1e999", 0, "not finite"),
        ("__import__('os')", 11, "unexpected character"),
        ("close; open", 5, "unexpected character ';'"),
        ("close ** 2", 7, "expected a number"),
    ],
)
def test_parse_errors_report_positions(text: str, position: int, fragment: str) -> None:
    with pytest.raises(ParseError) as info:
        parse(text)
    assert info.value.position == position
    assert fragment in info.value.message
    assert f"position {position}" in str(info.value)


def test_pretty_error_has_caret() -> None:
    with pytest.raises(ParseError) as info:
        parse("close $ open")
    lines = info.value.pretty().splitlines()
    assert lines[-1].index("^") == lines[-2].index("$")


def test_length_and_nesting_limits() -> None:
    with pytest.raises(ParseError, match="longer than"):
        parse("close+" * 400 + "close")
    with pytest.raises(ParseError, match="nesting"):
        parse("(" * 80 + "close" + ")" * 80)
    with pytest.raises(ParseError, match="nesting"):
        parse("-" * 80 + "close")
    with pytest.raises(TypeError):
        parse(42)  # type: ignore[arg-type]


def test_tokenizer_kinds() -> None:
    kinds = [token.kind for token in tokenize("a(1, -b) * 2 / c")]
    assert kinds == [
        "IDENT", "LPAREN", "NUMBER", "COMMA", "MINUS", "IDENT", "RPAREN",
        "STAR", "NUMBER", "SLASH", "IDENT", "EOF",
    ]


def test_package_never_evaluates_text() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "llm_factor_mining"
    # builtins only: ``re.compile(`` and method names such as ``evaluate(`` are fine
    forbidden = re.compile(r"(?<![\w.])(?:eval|exec|compile)\s*\(|literal_eval|__import__")
    offenders = [
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
