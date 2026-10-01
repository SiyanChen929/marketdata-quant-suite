# Research portfolio: a reviewer’s guide

## The research question

How can an AI-assisted quantitative-research workflow make its evidence auditable?
This repository studies two failure modes: selecting a factor after many hidden
trials, and reporting a numerically grounded answer that still uses future data.
The current contribution is research infrastructure and synthetic validation.
LLM evaluations and real-market replications are pending.

## A five-minute reading route

1. **Factor discovery:** read the [result card](../projects/llm-factor-mining/docs/result-card.md),
   then inspect the [search loop](../projects/llm-factor-mining/src/llm_factor_mining/search.py)
   and [hold-out protocol](../projects/llm-factor-mining/src/llm_factor_mining/protocol/seal.py).
   Random grammar search and genetic programming use matched trial budgets.
   Failed and invalid non-duplicate proposals remain counted trials.
2. **Agent evaluation:** inspect the [benchmark summary](../projects/marketdata-agent/results/benchmark/summary.md),
   [independent reference](../projects/marketdata-agent/src/marketdata_agent/bench/reference.py)
   and [as-of clock](../projects/marketdata-agent/src/marketdata_agent/clock.py).
   The benchmark has 174 tasks and six scripted baselines. It distinguishes
   numeric grounding, future-data leakage, refusals and task accuracy.
3. **Shared infrastructure:** inspect the [data gateway](../packages/quant-marketdata/README.md)
   and [research governance](../projects/quant-research-platform/docs/research-governance.md).
   Confirmed observations, provenance and chronological evaluation are shared
   contracts rather than separate conventions in each project.
4. **Reproduce a check:** follow the [offline verification guide](reproducibility.md).
   No vendor credentials or model calls are required for the documented checks.

## What the evidence supports

| Inspectable contribution | Evidence | Interpretation boundary |
|---|---|---|
| Factor-language timing constraints and search isolation | [Look-ahead tests](../projects/llm-factor-mining/tests/test_lookahead.py), [proposer-isolation tests](../projects/llm-factor-mining/tests/test_proposer_isolation.py) | Tests cover specified behaviours; they are not a proof of statistical validity for every future experiment |
| Trial accounting and sealed evaluation | [Protocol tests](../projects/llm-factor-mining/tests/test_protocol.py), [synthetic summary](../projects/llm-factor-mining/results/synthetic_benchmark/summary.md) | Synthetic recovery and null checks do not establish real-market alpha or an LLM advantage |
| Separate evaluation of grounding and temporal leakage | [Scoring tests](../projects/marketdata-agent/tests/test_scoring.py), [scripted baseline results](../projects/marketdata-agent/results/benchmark/summary.md) | Scripted policies validate instruments; their accuracy is not language-model performance |
| Repeatable evidence and document consistency | [CI configuration](../.github/workflows/ci.yml), [suite-contract tests](../tests/test_suite_contract.py) | An individual CI result applies only to its tested commit and environment |

## What remains open

- Run the pre-specified LLM arms with recorded model versions, costs and responses.
- Evaluate on licensed real-market inputs with explicit universe construction,
  timing, missingness, transaction costs and survivorship limitations.
- Compare methods at matched budgets and report ablations, uncertainty and
  negative results; more null seeds are needed before making a calibrated
  false-selection claim.
- Preserve the formation/validation/test separation and disclose every reveal.
- Complete and review the manuscripts. The current drafts are unpublished and
  have not been submitted; target venues are plans rather than acceptances.

The [research agenda](research-agenda.md) and each project's research plan define
these next experiments. The private equity-pairs study is separately disclosed
in its [README](../projects/equity-pairs-research/README.md#private-study-provenance-and-decision);
its excluded inputs prevent independent reproduction from this repository.

## Project description

A concise description suitable for linking from an application:

> Research software for trustworthy AI in quantitative finance: a typed
> factor-search language with trial accounting and sealed hold-out evaluation,
> plus a point-in-time copilot benchmark with provenance and grounding checks.
> Current evidence consists of synthetic search baselines and scripted-agent
> harness validation; LLM and real-market evaluations are pending.

Personal contribution statements should specify the work actually performed by
the applicant. Repository ownership alone does not establish sole authorship.
