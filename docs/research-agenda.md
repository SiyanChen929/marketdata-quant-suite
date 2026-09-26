# Research agenda: trustworthy AI for quantitative research

Status as of 2026-09-25. Both research lines are in preparation and nothing has been submitted. The committed evidence is synthetic harness validation; LLM and real-market results are pending. Dates are targets, not commitments.

## Thesis

Large language models are entering quantitative research in two roles: as generators of hypotheses and as interfaces to data. In both roles they inherit the two classic failures of empirical finance. The first is selection on noise across many tried specifications (Harvey, Liu & Zhu, 2016; White, 2000). The second is use of information that was not available at the time. An LLM makes both cheaper to commit and harder to see: it proposes candidates at almost no cost, and it may have read about the period it is tested on (Lopez-Lira & Tang, 2023; Glasserman & Lin, 2023).

The agenda is to build, and then evaluate, the controls that make an LLM's contribution measurable: count every trial, seal the test, and bind every reported number to data that existed at the time. The controls are enforced by construction and proved by tests. The model's behaviour inside them is the empirical question.

## Line 1: discovery under multiple-testing control

[`projects/llm-factor-mining`](../projects/llm-factor-mining) · [research plan](../projects/llm-factor-mining/docs/research-plan.md) · [paper draft](../projects/llm-factor-mining/paper/)

- **Question.** At an equal trial budget, does LLM-guided program search yield equity factors that survive multiple-testing control and one sealed test more often than random grammar search and genetic programming (Koza, 1992)? How much of any edge is rediscovery of known factors (Kakushadze, 2016) or memorization, rather than search? This carries LLM program search (Romera-Paredes et al., 2024) into a domain where every candidate is a test on one noisy history.
- **Method.** A typed factor language that cannot look ahead; formation-only feedback; a hash-chained ledger that logs every proposal and counts every non-duplicate one as a trial; two-stage Benjamini–Hochberg selection (Benjamini & Hochberg, 1995); a test hold-out committed before it is revealed; a planted-alpha benchmark with null markets; a knowledge-cutoff difference-in-differences for memorization.
- **Today.** The harness, two baselines and a committed synthetic benchmark. The synthetic results are consistent with the design argument that a formation screen alone is not a valid false-discovery control for a feedback-driven proposer: genetic programming passed candidates through the screen on 3 of 10 null markets, and confirmation on unseen validation data removed them all. They also show syntax-based novelty rating the selected factors as newer than value-based novelty does. With 3 seeds per planted cell and 10 null seeds, none of this is a statistical claim.
- **Next.** The LLM arm, a classic-library baseline, ablations, bootstrap tests (Hansen, 2005; Romano & Wolf, 2005) and sealed real-data reveals.

## Line 2: grounded, point-in-time tool use

[`projects/marketdata-agent`](../projects/marketdata-agent) · [research plan](../projects/marketdata-agent/docs/research-plan.md) · [safety model](../projects/marketdata-agent/docs/safety-model.md) · [paper draft](../projects/marketdata-agent/paper/)

- **Question.** Can a tool-using copilot (Yao et al., 2023) answer quantitative market questions correctly, with every number supported by the tool output it cites, while never reading data after its as-of date and never executing a trade? What do accuracy and safety cost in tokens and latency?
- **Method.** An as-of clock that refuses later dates, so attempts are counted; a policy gate that cannot enable execution; content-addressed provenance; a rounding-aware verifier that makes hallucination (Ji et al., 2023) measurable claim by claim; a benchmark with independent ground truth, cutoff traps and trade requests that claim prior approval. It adds point-in-time and action-safety constraints to document-based financial QA (Chen et al., 2021; Islam et al., 2023).
- **Today.** The governed runtime, a Claude backend with record and replay, the benchmark, six scripted baselines with byte-reproducible results, a verifier stress test and a power analysis.
- **Next.** All model runs, human validation of the verifier, an injection-through-tool-output suite, and a real-data replication on point-in-time-adjusted prices.

## Shared methods

- **Point-in-time by construction.** Look-ahead is impossible (the typed language) or refused and counted (the clock), never left to convention.
- **Everything on the record.** Ledgers and audit logs are hash-chained, with heads committed beside the results. Model calls record the served model and can be replayed; server-side fallback is off in experiments because it would change the model under test.
- **Decision rules before data.** Hypotheses and thresholds are fixed before any model run, "not supported" is a reportable outcome, and results climb the [evidence ladder](../projects/quant-research-platform/docs/research-governance.md) one rung at a time.

## Empirical substrate

| Component | Role for the research lines | State |
|---|---|---|
| [`quant-marketdata`](../packages/quant-marketdata) | The only price path: confirmed-only bars in a checksummed external store | in use |
| [`quant-research-platform`](../projects/quant-research-platform) | Research governance and the minimum result card that line 1 reports against; its cost-aware, next-session backtester is where a frozen factor set becomes a portfolio | governance in use; backtest link planned |
| [`equity-pairs-research`](../projects/equity-pairs-research) | Precedent for false-discovery control within a family and for reporting a negative held-out result | template for line 1's real-data report |
| [`index-rebalance-event-study`](../projects/index-rebalance-event-study) | A provider-neutral, point-in-time event schema for extending line 2 from price bars to dated announcements | planned, not started |

## Milestones, next 6 to 12 months (targets)

| Window | Line 1: factor mining | Line 2: copilot |
|---|---|---|
| Oct 2026 | Library baseline, ablations, bootstrap tests; freeze the plan and publish its hash before the first LLM call | Close the gaps before the first model run; pilot; freeze prompt and scorer |
| Nov 2026 | Synthetic LLM study on fresh seeds and null markets | Main evaluation with ablation arms |
| Dec 2026 – Jan 2027 | Real-data preparation; formation and validation runs only | Effort sweep, verifier annotation, adversarial suite; real-data replication subject to licensing |
| Feb 2027 | Pre-registered sealed reveals and result cards | Writing; first suitable deadline |
| Mar – Apr 2027 | Manuscript; a prospective test window accrues from the commitment date for a later version | An extended journal version after the real-data replication (not yet scheduled) |

Candidate venues, to be checked against each call for papers: ACM ICAIF, the KDD Applied Data Science track, the NeurIPS Datasets and Benchmarks track, FinNLP and finance workshops at ML conferences, ACL or EMNLP Findings, and the Journal of Financial Data Science or Quantitative Finance for extended versions.

## Risks

- **Cost and data.** Recorded runs are paid for once and replayed after that. Real-market claims wait for a point-in-time adjustment basis and a delisting-aware universe, and licensed data stay outside Git.
- **Null results.** Both plans pre-specify how a "not supported" result is reported. A well-measured null is still a contribution of the protocol and the benchmark.

## References

- Benjamini & Hochberg (1995). *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Chen et al. (2021). FinQA. *EMNLP 2021*.
- Glasserman & Lin (2023). Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis. arXiv:2309.17322. *(metadata to be verified before submission)*
- Hansen (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Harvey, Liu & Zhu (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Islam et al. (2023). FinanceBench. arXiv:2311.11944.
- Ji et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12).
- Kakushadze (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72–81.
- Koza (1992). *Genetic Programming*. MIT Press.
- Lopez-Lira & Tang (2023). Can ChatGPT forecast stock price movements? arXiv:2304.07619.
- Romano & Wolf (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237–1282.
- Romera-Paredes et al. (2024). Mathematical discoveries from program search with large language models. *Nature*, 625, 468–475.
- White (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
- Yao et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.
