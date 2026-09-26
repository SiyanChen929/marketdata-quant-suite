"""Task generation: determinism, coverage, point-in-time constraints, and serialization."""

from __future__ import annotations

from collections import Counter

import pytest

from marketdata_agent.bench import CATEGORIES, DEFAULT_SEED, generate_suite, load_suite
from marketdata_agent.bench.baselines import format_value
from marketdata_agent.bench.generator import DEFAULT_DATASET, dataset_frame
from marketdata_agent.bench.reference import RawPanel


@pytest.fixture(scope="module")
def suite():
    return generate_suite(DEFAULT_SEED)


@pytest.fixture(scope="module")
def raw():
    return RawPanel.from_frame(dataset_frame(DEFAULT_DATASET))


def _dates(args: dict) -> list[str]:
    return [v for k, v in args.items() if k in {"start", "end"} and isinstance(v, str) and v != "latest"]


def test_generation_is_deterministic_and_seed_sensitive(suite):
    again = generate_suite(DEFAULT_SEED)
    assert again.to_dict() == suite.to_dict() and again.sha256() == suite.sha256()
    assert generate_suite(DEFAULT_SEED + 1).sha256() != suite.sha256()


def test_suite_covers_every_category_with_at_least_150_tasks(suite):
    assert len(suite) >= 150
    counts = suite.counts()
    assert set(counts) == set(CATEGORIES) and min(counts.values()) >= 12
    kinds = Counter(task.expected.kind for task in suite)
    assert kinds["numeric"] >= 90 and kinds["ranking"] >= 20 and kinds["abstain"] >= 30 and kinds["refuse"] >= 12
    assert len({task.id for task in suite}) == len(suite)
    assert len({task.as_of for task in suite}) >= 10  # the cutoff varies across the suite


def test_answerable_tasks_never_need_data_after_as_of(suite):
    for task in suite:
        if task.expected.kind in {"numeric", "ranking"}:
            assert task.reference_plan, task.id
            for step in task.reference_plan:
                assert all(day <= task.as_of for day in _dates(step.args)), task.id


def test_pit_traps_ask_about_dates_after_as_of_and_carry_hindsight_values(suite):
    traps = [t for t in suite if t.category == "pit_trap"]
    assert traps and all(t.expected.kind == "abstain" and not t.reference_plan for t in traps)
    for task in traps:
        assert task.metadata["after_as_of"] > task.as_of and task.metadata["after_as_of"] in task.question
        assert task.hindsight is not None and task.hindsight.kind == "numeric"
        assert any(day > task.as_of for step in task.naive_plan for day in _dates(step.args))


def test_unknown_symbols_are_absent_or_list_after_as_of(suite, raw):
    for task in (t for t in suite if t.category == "unknown_symbol"):
        symbol = task.symbols[0]
        assert symbol not in raw.universe(task.as_of), task.id
        if task.metadata["symbol_status"] == "lists_after_as_of":
            assert symbol in raw.universe(raw.sessions()[-1])


def test_universe_counts_exclude_symbols_that_list_later(suite, raw):
    for task in (t for t in suite if t.subcategory == "universe_count"):
        expected = 8 if task.as_of < "2022-09-01" else 9
        assert task.expected.value == expected == len(raw.universe(task.as_of))


def test_trade_requests_expect_refusal_and_offer_only_a_proposal_plan(suite):
    for task in (t for t in suite if t.category == "policy_trap"):
        assert task.expected.kind == "refuse"
        assert [s.tool for s in task.reference_plan] == ["propose_order"]
        assert [s.tool for s in task.naive_plan] == ["execute_order"]


def test_hindsight_values_are_distinguishable_from_the_truth(suite):
    for task in suite:
        if task.hindsight is None or task.expected.kind not in {"numeric", "ranking"}:
            continue
        if task.expected.kind == "ranking":
            assert task.hindsight.value != task.expected.value
        else:
            assert abs(task.hindsight.value - task.expected.value) > 2 * task.expected.tolerance


def test_questions_do_not_contain_their_answers(suite):
    for task in suite:
        if task.expected.kind == "numeric" and task.expected.unit != "count":
            assert format_value(task.expected.value, task.expected.unit) not in task.question, task.id


def test_stratified_subset_and_json_round_trip(suite, tmp_path):
    small = suite.subset(12)
    assert len(small) == 12 and set(small.counts()) == set(CATEGORIES)
    assert suite.subset(10_000) is suite
    path = suite.write_json(tmp_path / "suite.json")
    assert load_suite(path).sha256() == suite.sha256()


def test_the_30_task_pilot_subset_covers_every_subcategory(suite):
    """Regression: the subset used to round-robin over categories only and covered 7 of 18 subcategories."""

    pilot = suite.subset(30)
    assert Counter(t.category for t in pilot) == {category: 5 for category in CATEGORIES}
    assert {(t.category, t.subcategory) for t in pilot} == {(t.category, t.subcategory) for t in suite}
    assert [t.id for t in pilot] == [t.id for t in suite if t in pilot.tasks]  # suite order is kept


def test_every_task_is_a_distinct_functional_item(suite):
    """Regression: the v1 suite contained two exact (question, as_of) duplicates."""

    keys = suite.item_keys()
    assert len(keys) == len(set(keys)) == len(suite)
    pairs = [(t.question, t.as_of) for t in suite]
    assert len(pairs) == len(set(pairs))
    for category in ("unknown_symbol", "policy_trap"):  # as-of-independent items are keyed by their text alone
        questions = [t.question for t in suite if t.category == category]
        assert len(questions) == len(set(questions))
    assert all(t.metadata["cluster"].startswith(f"{t.category}|{t.subcategory}") for t in suite)


def test_evaluation_suites_share_no_item_with_each_other_or_the_development_suite(suite):
    from marketdata_agent.bench import evaluation_dataset, generate_evaluation_suites

    evaluation = generate_evaluation_suites([11, 12], exclude=[suite])
    keys = [set(s.item_keys()) for s in (suite, *evaluation)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert evaluation_dataset(11) != evaluation_dataset(12) != DEFAULT_DATASET
    assert evaluation[0].dataset == evaluation_dataset(11) and evaluation[0].sha256() != suite.sha256()


def test_return_questions_state_the_base_convention(suite):
    """Regression: a weekend start date used to leave the base close unstated."""

    returns = [t for t in suite if t.subcategory in {"return", "future_window", "straddle_window", "rank_returns"}]
    assert returns and all("the close of the first confirmed session on or after" in t.question for t in returns)
    weekend = [t for t in suite if t.metadata.get("calendar_start")]
    assert weekend  # the convention matters: some starts fall on non-session days


def test_unknown_symbols_are_stratified_and_real_tickers_are_guaranteed(suite):
    from marketdata_agent.bench.generator import FICTIONAL_SYMBOLS, REAL_SYMBOLS

    counts = Counter(t.subcategory for t in suite if t.category == "unknown_symbol")
    assert counts == {"absent_fictional": 4, "absent_real": 4, "not_yet_listed": 4}
    for task in suite:
        if task.subcategory == "absent_real":
            assert task.symbols[0] in REAL_SYMBOLS and task.metadata["real_ticker"] is True
        if task.subcategory == "absent_fictional":
            assert task.symbols[0] in FICTIONAL_SYMBOLS and task.metadata["real_ticker"] is False
    assert all(t.expected.unit in {"usd", "percent"} for t in suite if t.expected.kind == "abstain")


def test_trade_requests_cover_many_templates(suite):
    from marketdata_agent.bench.generator import TRADE_TEMPLATES

    assert len(TRADE_TEMPLATES) == 12
    templates = [t.metadata["template"] for t in suite if t.category == "policy_trap"]
    assert sorted(templates) == list(range(12))  # regression: v2 drew templates at random and covered only 7


def test_evaluation_suites_cover_every_trade_template_equally():
    from collections import Counter

    from marketdata_agent.bench import generate_evaluation_suites
    from marketdata_agent.bench.design import EVALUATION_COUNTS

    (evaluation,) = generate_evaluation_suites([11], counts=EVALUATION_COUNTS)
    counts = Counter(t.metadata["template"] for t in evaluation if t.category == "policy_trap")
    assert counts == {template: 3 for template in range(12)}


def test_every_question_states_its_reporting_unit(suite):
    for task in suite:
        if task.category in {"lookup", "compute", "multi_step", "pit_trap"} and task.subcategory != "universe_count":
            if task.subcategory == "rank_returns":
                continue  # the answer is an order of symbols
            assert "percentage" in task.question or "US dollars" in task.question or "decimals" in task.question, task.id


def test_an_exhausted_template_space_is_an_error_not_a_silent_duplicate():
    with pytest.raises(RuntimeError, match="exhausted"):
        generate_suite(DEFAULT_SEED, counts={("lookup", "universe_count"): 40})
