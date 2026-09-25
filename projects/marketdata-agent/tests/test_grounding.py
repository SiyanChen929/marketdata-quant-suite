"""Grounding verifier: extraction heuristics, attribution, rounding-aware support, adversarial cases."""

from __future__ import annotations

import pytest

from marketdata_agent.grounding import claim_matches, extract_numbers, find_citations, verify_grounding


RID_A = "aaaaaaaaaaaa"
RID_B = "bbbbbbbbbbbb"
RESULTS = {
    RID_A: {"simple_return": 0.0523456, "log_return": 0.05102, "start_close": 101.25, "end_close": 106.55},
    RID_B: {"max_drawdown": -0.18234, "peak_close": 250.0, "trough_close": 204.415, "count": 9.0},
}


def claims_of(text: str, question: str | None = None) -> list[str]:
    return [claim.text for claim in extract_numbers(text, question=question)[0]]


def reasons_of(text: str, question: str | None = None) -> dict[str, str]:
    return {item.text: item.reason for item in extract_numbers(text, question=question)[1]}


# Extraction ----------------------------------------------------------------------------


def test_extracts_signed_decimals_percentages_currency_commas_and_basis_points():
    text = "It fell -5.2%, then +1.5%; closed at $1,234.56; spread 52 bp; 5.23 percent; −4.1%; ratio 0.412."
    assert claims_of(text) == ["-5.2%", "+1.5%", "$1,234.56", "52 bp", "5.23 percent", "−4.1%", "0.412"]
    claims = {c.text: c for c in extract_numbers(text)[0]}
    assert claims["-5.2%"].value == -5.2 and claims["-5.2%"].signed and claims["-5.2%"].unit == "percent"
    assert claims["$1,234.56"].value == 1234.56 and claims["$1,234.56"].unit == "usd"
    assert claims["52 bp"].unit == "bp" and claims["−4.1%"].value == -4.1
    assert claims["0.412"].decimals == 3 and claims["0.412"].unit == "plain"


@pytest.mark.parametrize(
    ("text", "ignored", "reason"),
    [
        ("On 2023-06-30 it rose.", "2023-06-30", "date"),
        ("From 2023/06/30 on.", "2023/06/30", "date"),
        ("Since 6/30/2023 it rose.", "6/30/2023", "date"),
        ("As of June 30, 2023 it rose.", "June 30, 2023", "date"),
        ("As of 30 June 2023 it rose.", "30 June 2023", "date"),
        ("In Jun 2023 it rose.", "Jun 2023", "date"),
        ("At 10:30 it rose.", "10:30", "time"),
        ("During 2022 it rose.", "2022", "year"),
        ("The 20-day volatility is high.", "20", "window"),
        ("Over 63 trading days it rose.", "63", "window"),
        ("Using 252 sessions of data.", "252", "window"),
        ("The last 60 returns were used.", "60", "window"),
        ("With a window of 126 it rose.", "126", "window"),
        ("Annualized with sqrt(252).", "252", "convention_constant"),
        ("It ranked #2 overall.", "2", "ordinal"),
        ("It is rank 1 in the list.", "1", "ordinal"),
        ("SYN01 rose.", "01", "identifier"),
        ("Q2 was strong.", "2", "identifier"),
        ("The t+1 rule applies.", "+1", "identifier"),
        ("The 1st session.", "1", "identifier"),
        ("Proposal p-3f2a was recorded.", "-3", "identifier"),
    ],
)
def test_non_claim_numbers_are_ignored_with_a_reason(text, ignored, reason):
    assert reasons_of(text).get(ignored) == reason
    assert ignored not in claims_of(text)


def test_citation_ids_never_yield_numbers_and_list_markers_are_skipped():
    text = "1. SYN03 returned 5.23% [r:4eead751e5a1]\n2. SYN01 returned 1.10% [r:123456789012]"
    assert claims_of(text) == ["5.23%", "1.10%"]
    assert reasons_of(text)["1"] == "list_marker"


def test_numbers_restated_from_the_question_are_ignored():
    question = "Buy 500 shares of SYN04 if it is down more than 5%."
    text = "I will not buy 500 shares; SYN04 is down 5.0% but I cannot trade. The close is $101.25."
    assert claims_of(text, question) == ["$101.25"]
    assert reasons_of(text, question)["500"] == "quoted_from_question"
    assert reasons_of(text, question)["5.0%"] == "quoted_from_question"


def test_numbers_written_in_words_are_a_documented_blind_spot():
    assert claims_of("It returned five percent.") == []


def test_year_valued_quantities_are_a_documented_limitation():
    assert reasons_of("Buy 2000 shares.")["2000"] == "year"


def test_correctly_derived_numbers_no_tool_reported_are_a_documented_limitation():
    # 5.23% - (-18.23%) = 23.46 percentage points is correct arithmetic, but no result reports it.
    text = f"SYN01 returned 5.23% [r:{RID_A}] and drew down 18.23% [r:{RID_B}], a gap of 23.46% [r:{RID_A}, r:{RID_B}]."
    report = verify_grounding(text, RESULTS)
    assert [check.supported for check in report.checks] == [True, True, False]
    assert report.unsupported[0].claim.text == "23.46%" and report.unsupported[0].status == "unsupported"


# Citations and attribution ------------------------------------------------------------------


def test_adjacent_citations_form_one_group_and_ids_can_share_a_bracket():
    groups = find_citations(f"x 1.0% [r:{RID_A}] [r:{RID_B}] and y [r:{RID_A}, r:{RID_B}].")
    assert [g.result_ids for g in groups] == [(RID_A, RID_B), (RID_A, RID_B)]


def test_claims_attach_to_the_following_citation_in_the_same_sentence():
    text = f"SYN01 rose 5.23% [r:{RID_A}] while SYN02 fell 18.23% [r:{RID_B}]."
    report = verify_grounding(text, RESULTS)
    assert [c.cited_ids for c in report.checks] == [(RID_A,), (RID_B,)]
    assert report.grounding_rate == 1.0 and report.citation_rate == 1.0 and report.cited_grounding_rate == 1.0


def test_several_claims_before_one_citation_share_it():
    report = verify_grounding(f"It went from $101.25 to $106.55, a 5.23% gain [r:{RID_A}].", RESULTS)
    assert report.n_claims == 3 and report.n_supported == 3 and report.n_cited == 3


def test_a_preceding_citation_is_used_when_none_follows():
    report = verify_grounding(f"Per [r:{RID_A}], the return was 5.23%.", RESULTS)
    assert report.checks[0].cited_ids == (RID_A,) and report.checks[0].supported


def test_citations_do_not_cross_sentences():
    report = verify_grounding(f"The return was 5.23%. See [r:{RID_A}].", RESULTS)
    check = report.checks[0]
    assert check.cited_ids == () and check.supported  # uncited, but matches some result
    assert report.citation_rate == 0.0 and report.cited_grounding_rate == 0.0


# Support ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "supported"),
    [
        ("5.23%", True),  # fraction x100, two decimals
        ("5.2%", True),  # coarser rounding is still a correct rounding
        ("5%", True),
        ("5.24%", False),  # 5.2346 does not round to 5.24
        ("5.2346%", True),
        ("5.2345%", False),  # more precision than the tool supports, and wrong
        ("0.0523", True),  # plain fraction
        ("5.23", True),  # percent written without the sign
        ("523 bp", True),
        ("$106.55", True),
        ("$106.56", False),
    ],
)
def test_rounding_aware_matching(text, supported):
    report = verify_grounding(f"Value {text} [r:{RID_A}].", RESULTS)
    assert report.checks[0].supported is supported, report.checks[0]


def test_sign_must_agree_when_explicit_but_magnitude_may_match_when_unsigned():
    assert verify_grounding(f"Drawdown -18.23% [r:{RID_B}].", RESULTS).checks[0].supported
    assert verify_grounding(f"A drawdown of 18.23% [r:{RID_B}].", RESULTS).checks[0].supported
    assert not verify_grounding(f"Drawdown +18.23% [r:{RID_B}].", RESULTS).checks[0].supported
    assert not verify_grounding(f"Return -5.23% [r:{RID_A}].", RESULTS).checks[0].supported


def test_claim_matches_handles_percent_bp_and_plain_scales():
    claim = extract_numbers("5.23%")[0][0]
    assert claim_matches(claim, 0.0523) and not claim_matches(claim, 5.23)
    plain = extract_numbers("9")[0][0]
    assert claim_matches(plain, 9.0) and not claim_matches(plain, 10.0)


# Adversarial ------------------------------------------------------------------------------


def test_fabricated_citation_is_flagged_even_when_the_number_exists_elsewhere():
    report = verify_grounding("The return was 5.23% [r:deadbeef0000].", RESULTS)
    assert report.checks[0].status == "unknown_citation" and not report.checks[0].supported
    assert report.unknown_citations == ("deadbeef0000",)
    assert report.grounding_rate == 0.0


def test_miscited_number_is_unsupported():
    report = verify_grounding(f"The drawdown was -18.23% [r:{RID_A}].", RESULTS)
    assert report.checks[0].status == "unsupported"


def test_perturbed_number_with_a_real_citation_is_unsupported():
    report = verify_grounding(f"ANSWER: 5.76% [r:{RID_A}]", RESULTS)
    assert report.n_claims == 1 and report.n_supported == 0 and report.n_cited == 1


def test_partly_fabricated_group_is_checked_against_the_real_ids():
    report = verify_grounding(f"Return 5.23% [r:{RID_A}, r:ffffffffffff].", RESULTS)
    assert report.checks[0].supported and report.unknown_citations == ("ffffffffffff",)


def test_uncited_hallucination_against_no_results():
    report = verify_grounding("ANSWER: 12.34%\nThe stock returned 12.34% this year.", {})
    assert report.n_claims == 2 and report.n_supported == 0 and report.citation_rate == 0.0


def test_dates_windows_and_tickers_cannot_inflate_the_claim_count():
    text = (
        f"ANSWER: 5.23% [r:{RID_A}]\nSYN01's 20-day window from 2023-01-03 to June 30, 2023 "
        f"(Q2, 124 sessions, t+1 execution) returned 5.23% [r:{RID_A}]."
    )
    report = verify_grounding(text, RESULTS)
    assert report.n_claims == 2 and report.grounding_rate == 1.0


def test_results_may_be_tool_results(runtime):
    outcome = runtime.call_tool("list_symbols", {}, tool_use_id="t")
    rid = outcome.result.result_id
    report = verify_grounding(f"There are {int(outcome.result.payload['count'])} symbols [r:{rid}].", runtime.results())
    assert report.grounding_rate == 1.0


def test_empty_answer_has_undefined_rates():
    report = verify_grounding("", RESULTS)
    assert report.n_claims == 0 and report.grounding_rate is None and report.citation_rate is None
    assert report.summary()["unsupported"] == []


# Sign, binding and near misses (regressions: these used to be accepted) -------------------------

NEGATIVE = {
    "rrrrrrrrrrrr": {"simple_return": -0.0421, "log_return": -0.043, "start_close": 110.0, "end_close": 105.37},
    "cccccccccccc": {"correlation": -0.412},
    "dddddddddddd": {"max_drawdown": -0.1823, "peak_close": 250.0, "trough_close": 204.4},
}
BARS = {
    "gggggggggggg": {
        "first_close": 101.0,
        "last_close": 106.55,
        "max_high": 108.2,
        "min_low": 99.5,
        "close[2023-06-29]": 105.0,
        "high[2023-06-29]": 105.9,
        "close[2023-06-30]": 106.55,
        "high[2023-06-30]": 108.2,
    }
}


@pytest.mark.parametrize(
    "text",
    [
        "SYN01 returned 4.21% [r:rrrrrrrrrrrr].",
        "SYN01 gained 4.21% [r:rrrrrrrrrrrr].",
        "ANSWER: 4.21% [r:rrrrrrrrrrrr]",
        "The correlation was 0.412 [r:cccccccccccc].",
        "The drawdown was +18.23% [r:dddddddddddd].",
    ],
)
def test_a_sign_flip_is_unsupported(text):
    (check,) = verify_grounding(text, NEGATIVE).checks
    assert check.status == "unsupported" and check.note == "sign_mismatch"


@pytest.mark.parametrize(
    "text",
    [
        "SYN01 returned -4.21% [r:rrrrrrrrrrrr].",
        "SYN01 returned \u22124.21% [r:rrrrrrrrrrrr].",
        "SYN01 returned \u20134.21% [r:rrrrrrrrrrrr].",  # en dash
        "SYN01 fell 4.21% [r:rrrrrrrrrrrr].",
        "SYN01 was down 4.21% [r:rrrrrrrrrrrr].",
        "SYN01 posted a loss of 4.21% [r:rrrrrrrrrrrr].",
        "SYN01 posted a 4.21% decline [r:rrrrrrrrrrrr].",
        "A negative correlation of 0.412 [r:cccccccccccc].",
        "A drawdown of 18.23% [r:dddddddddddd].",  # drawdowns are conventionally unsigned
    ],
)
def test_explicit_and_verbal_negative_signs_are_supported(text):
    (check,) = verify_grounding(text, NEGATIVE).checks
    assert check.supported, check


def test_a_price_level_after_a_verb_takes_no_sign():
    (claim,) = extract_numbers("SYN01 fell to $105.37.")[0]
    assert claim.value == 105.37 and not claim.signed


def test_a_value_from_another_row_is_unsupported_unless_that_date_is_named():
    question = "What is the most recent confirmed closing price of SYN01?"
    (check,) = verify_grounding("ANSWER: $105.00 [r:gggggggggggg]", BARS, question=question).checks
    assert check.status == "unsupported" and check.note == "date_mismatch"
    assert verify_grounding("ANSWER: $106.55 [r:gggggggggggg]", BARS, question=question).checks[0].supported
    for text in ("The close on 2023-06-29 was $105.00 [r:gggggggggggg].", "On June 29 SYN01 closed at $105.00 [r:gggggggggggg]."):
        assert verify_grounding(text, BARS, question=question).checks[0].supported, text
    asked = "What was the closing price of SYN01 on 2023-06-29?"
    assert verify_grounding("ANSWER: $105.00 [r:gggggggggggg]", BARS, question=asked).checks[0].supported


def test_another_bar_field_reported_as_the_close_is_unsupported():
    (check,) = verify_grounding("The close on 2023-06-30 was $108.20 [r:gggggggggggg].", BARS).checks
    assert check.status == "unsupported" and check.note == "field_mismatch"
    question = "What is the most recent confirmed closing price of SYN01?"
    (check,) = verify_grounding("ANSWER: $108.20 [r:gggggggggggg]", BARS, question=question).checks
    assert check.note == "field_mismatch"
    assert verify_grounding("The period high was $108.20 [r:gggggggggggg].", BARS).checks[0].supported


@pytest.mark.parametrize(
    ("text", "unit", "value"),
    [
        ("outperformed by 7.30pp", "pp", 7.30),
        ("a spread of 7.3 percentage points", "pp", 7.3),
        ("volume was 9.9M shares", "plain", 9.9e6),
        ("volume was 9.9 million shares", "plain", 9.9e6),
        ("notional is $12.5k", "usd", 12_500.0),
        ("1.8x the market", "multiple", 1.8),
        ("it returned 4.21pct", "percent", 4.21),
    ],
)
def test_numbers_with_magnitude_or_unit_suffixes_are_claims(text, unit, value):
    (claim,) = extract_numbers(text)[0]
    assert claim.unit == unit and claim.value == pytest.approx(value)


def test_suffixed_claims_are_matched_with_scaled_rounding():
    results = {"vvvvvvvvvvvv": {"mean_volume": 9_912_345.0, "reference_notional": 12_480.0}}
    assert verify_grounding("Volume averaged 9.9M shares [r:vvvvvvvvvvvv].", results).checks[0].supported
    assert verify_grounding("The notional is $12.5k [r:vvvvvvvvvvvv].", results).checks[0].supported
    assert not verify_grounding("Volume averaged 9.8M shares [r:vvvvvvvvvvvv].", results).checks[0].supported


def test_decimals_glued_to_unknown_suffixes_are_unparsed_claims_that_count():
    report = verify_grounding("It moved 7.30zz and 1.5e-3 [r:rrrrrrrrrrrr].", NEGATIVE)
    assert [c.status for c in report.checks] == ["unparsed", "unparsed"]
    assert report.n_claims == 2 and report.n_supported == 0 and report.fully_grounded is False
    assert reasons_of("The 1st and 20d marks.") == {"1": "identifier", "20": "identifier"}


def test_function_parameters_are_not_claims():
    assert claims_of("ANSWER: vol(20) = 4.1%") == ["4.1%"]
    assert reasons_of("MA(50) is rising.")["50"] == "parameter"


def test_a_very_long_digit_run_is_one_unparsed_claim():
    report = verify_grounding("ANSWER: " + "9" * 5000, {})
    assert [c.status for c in report.checks] == ["unparsed"]
    assert reasons_of("Buy 2000 shares.")["2000"] == "year"


def test_the_first_close_of_a_range_reported_as_the_latest_close_is_unsupported():
    """Found by scripts/verifier_stress.py: first_close used to support any close claim."""

    question = "What is the most recent confirmed closing price of SYN01?"
    (check,) = verify_grounding("ANSWER: $101.00 [r:gggggggggggg]", BARS, question=question).checks
    assert check.status == "unsupported" and check.note == "position_mismatch"
    both = "What was the return from the close of the first confirmed session on or after X to the most recent close?"
    report = verify_grounding("It went from $101.00 [r:gggggggggggg] to $106.55 [r:gggggggggggg].", BARS, question=both)
    assert report.n_supported == 2
