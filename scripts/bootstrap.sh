#!/usr/bin/env bash
set -euo pipefail

# Development install of every suite component. The optional "llm" extra of the
# two LLM research projects (the Anthropic SDK) is deliberately not installed:
# their offline test suites must pass without it. Add it only for live runs,
# e.g. python -m pip install -c configs/constraints.txt -e "projects/llm-factor-mining[llm]".
#
# configs/constraints.txt pins the numeric stack recorded in the committed
# results, so that regenerated results can be compared with them.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONSTRAINTS="configs/constraints.txt"
python -m pip install -c "$CONSTRAINTS" -e "packages/quant-marketdata[dev]"
python -m pip install -c "$CONSTRAINTS" -e "projects/quant-research-platform[dev]"
python -m pip install -c "$CONSTRAINTS" -e "projects/equity-pairs-research[dev]"
python -m pip install -c "$CONSTRAINTS" -e "projects/index-rebalance-event-study[dev]"
python -m pip install -c "$CONSTRAINTS" -e "projects/llm-factor-mining[dev]"
python -m pip install -c "$CONSTRAINTS" -e "projects/marketdata-agent[dev]"
