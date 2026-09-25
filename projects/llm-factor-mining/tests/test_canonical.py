from __future__ import annotations

import re

import pytest

from llm_factor_mining.dsl import (
    LIBRARY,
    canonical_string,
    canonicalize,
    complexity,
    library_nodes,
    nearest_reference,
    parse,
    structural_hash,
    structural_similarity,
    subtree_hashes,
)


def h(text: str) -> str:
    return structural_hash(parse(text))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("close + open", "open + close"),
        ("add(close, open)", "open + close"),
        ("close * volume", "volume * close"),
        ("ts_corr(close, volume, 10)", "ts_corr(volume, close, 10)"),
        ("ts_cov(returns, volume, 5)", "ts_cov(volume, returns, 5)"),
        (
            "cs_rank(ts_mean(close, 5) + ts_std(returns, 20)) * volume",
            "volume * cs_rank(ts_std(returns, 20) + ts_mean(close, 5))",
        ),
        ("power(close, 2)", "power(close, 2.0)"),
        ("close + 1", "close + 1.0"),
        ("close + 1", "1 + close"),
        ("-(close)", "neg(close)"),
        ("TS_MEAN(Close, 5)", "ts_mean(close,5)"),
    ],
)
def test_hash_invariance(left: str, right: str) -> None:
    assert h(left) == h(right)
    assert canonical_string(parse(left)) == canonical_string(parse(right))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("close - open", "open - close"),
        ("close / open", "open / close"),
        ("ts_mean(close, 5)", "ts_mean(close, 10)"),
        ("delay(close, 1)", "delta(close, 1)"),
        ("(close + open) + high", "close + (open + high)"),  # associativity is not normalized
    ],
)
def test_hash_distinguishes(left: str, right: str) -> None:
    assert h(left) != h(right)


def test_hash_format_and_canonical_idempotence() -> None:
    for factor in LIBRARY:
        node = factor.node
        digest = structural_hash(node)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        canonical = canonicalize(node)
        assert canonicalize(canonical) == canonical
        assert canonical_string(canonical) == canonical_string(node)
        assert structural_hash(canonical) == digest


def test_canonical_sorting_is_by_string() -> None:
    assert canonical_string(parse("volume + close")) == "add(close,volume)"
    assert canonical_string(parse("ts_corr(volume, close, 5)")) == "ts_corr(close,volume,5)"


def test_complexity() -> None:
    assert complexity(parse("close")) == 1.0
    assert complexity(parse("ts_mean(close, 5)")) == 2.5
    # 4 operators + 2 terminals + 0.5 * 2 literals
    assert complexity(parse("cs_rank(ts_mean(close, 5) - ts_mean(close, 20))")) == 7.0


def test_subtree_hashes_exclude_bare_literals() -> None:
    hashes = subtree_hashes(parse("ts_mean(close, 5) + 1"))
    assert len(hashes) == 3  # close, ts_mean(close,5), add(...)


def test_structural_similarity() -> None:
    close5 = parse("ts_mean(close, 5)")
    close10 = parse("ts_mean(close, 10)")
    assert structural_similarity(close5, close5) == 1.0
    assert structural_similarity(parse("close"), parse("open")) == 0.0
    assert structural_similarity(close5, close10) == pytest.approx(1.0 / 3.0)
    assert structural_similarity(close5, close10, abstract_literals=True) == 1.0
    assert structural_similarity(
        parse("close * volume + open"), parse("open + volume * close")
    ) == 1.0
    partial = structural_similarity(
        parse("cs_rank(ts_std(returns, 20))"), parse("-ts_std(returns, 20)")
    )
    assert 0.0 < partial < 1.0


def test_nearest_reference_in_library() -> None:
    references = library_nodes()
    variant = parse("delay(close, 21) / delay(close, 200) - 1")
    exact = nearest_reference(variant, references)
    abstract = nearest_reference(variant, references, abstract_literals=True)
    names = [factor.name for factor in LIBRARY]
    assert names[abstract.index] in {"momentum_12_1", "momentum_6_1"}
    assert abstract.similarity == 1.0
    assert 0.0 < exact.similarity < 1.0
    assert nearest_reference(variant, []).index is None
