# Offline verification

These commands validate the software and synthetic evidence without a market-data
subscription or LLM API key. Package installation needs internet access; the checks
below run offline after installation. Run from the repository root using Python
3.11 or 3.12 on a Unix-like shell.

## 1. Record the source and install

```bash
git rev-parse HEAD
python3 --version
python3 -m venv .venv
source .venv/bin/activate
./scripts/bootstrap.sh
python -m pip freeze
```

Keep the commit hash and dependency versions with your verification log.
[The constraints file](../configs/constraints.txt) pins the numeric versions used
for the committed factor-mining results. Other dependencies are not fully locked.
The optional model SDK is not installed by the bootstrap script.

## 2. Run the complete offline checks

```bash
./scripts/test-all.sh
python scripts/repository-audit.py
python projects/llm-factor-mining/scripts/render_paper_tables.py --check
python projects/marketdata-agent/scripts/render_paper_tables.py --check
```

Expected: every command exits with status zero. Pytest reports each component
separately; optional SDK tests may skip when the SDK is absent. A passing run
establishes the checked software behaviours, not an economic or scientific result.
The repository audit is a set of heuristic checks, not a complete security review.

## 3. Regenerate the scripted-agent benchmark

Use a new temporary directory so committed evidence and previous runs remain
untouched:

```bash
CHECK_DIR=$(mktemp -d)
python projects/marketdata-agent/scripts/run_benchmark.py --out "$CHECK_DIR/agent"
diff -r -x audit projects/marketdata-agent/results/benchmark "$CHECK_DIR/agent"
printf 'Generated evidence: %s\n' "$CHECK_DIR"
```

Expected on the recorded software stack (including Python 3.11.15): `diff`
exits zero with no output. On a different Python version, the summaries record
that version under `environment.python`; this metadata difference is expected.
Review all differences, and do not discard changes to results or audit heads.
The benchmark evaluates six scripted policies on 174 synthetic tasks. No language
model is called. The comparison excludes generated audit directories, which are
not committed; this command does not independently certify all audit logs.
If a difference appears, retain both outputs and compare versions and numeric
fields before interpreting it as a research change.

## 4. Regenerate the factor-search benchmark

This is the slower check. The [result card](../projects/llm-factor-mining/docs/result-card.md)
explains the market seeds, trial budgets, timing and interpretation.

```bash
FACTOR_DIR=$(mktemp -d)
python projects/llm-factor-mining/scripts/run_synthetic_benchmark.py \
  --out "$FACTOR_DIR/summary" --runs-dir "$FACTOR_DIR/runs"
printf 'Generated evidence: %s\n' "$FACTOR_DIR"
```

Compare `summary/summary.json` and `summary/summary.md` with the committed files
under `projects/llm-factor-mining/results/synthetic_benchmark/`. Wall-clock timings
and command-line output paths may differ. CPU-dependent floating-point rounding
can affect near-tied diagnostics, so do not promise byte-identical factor results.
This command runs the random and evolutionary baselines; it does not establish
LLM performance. Do not replace committed results merely to hide a mismatch.

## Evidence to retain

- Source commit, Python and package versions, OS/CPU details
- Commands, exit statuses, test failures and skips
- Generated summaries and any comparison differences
- Run ledgers and commitments needed to trace the reported outputs

Never include API keys, licensed market data or private run recordings in a
public verification log. Live experiments have additional requirements in
[CONTRIBUTING](../CONTRIBUTING.md) and the project research plans.
