#!/usr/bin/env bash
set -euo pipefail

# Development install of every suite component. The optional "llm" extra of the
# two LLM research projects (the Anthropic SDK) is deliberately not installed:
# their offline test suites must pass without it. Add it only for live runs,
# e.g. python -m pip install -e "projects/llm-factor-mining[llm]".
python -m pip install -e "packages/quant-marketdata[dev]"
python -m pip install -e "projects/quant-research-platform[dev]"
python -m pip install -e "projects/equity-pairs-research[dev]"
python -m pip install -e "projects/index-rebalance-event-study[dev]"
python -m pip install -e "projects/llm-factor-mining[dev]"
python -m pip install -e "projects/marketdata-agent[dev]"
