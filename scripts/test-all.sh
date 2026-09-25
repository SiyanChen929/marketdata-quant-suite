#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUITE_ROOT"
SUITE_SRC="$SUITE_ROOT/packages/quant-marketdata/src"
SUITE_SRC+=":$SUITE_ROOT/projects/quant-research-platform/src"
SUITE_SRC+=":$SUITE_ROOT/projects/equity-pairs-research/src"
SUITE_SRC+=":$SUITE_ROOT/projects/index-rebalance-event-study/src"
SUITE_SRC+=":$SUITE_ROOT/projects/llm-factor-mining/src"
SUITE_SRC+=":$SUITE_ROOT/projects/marketdata-agent/src"
export PYTHONPATH="$SUITE_SRC${PYTHONPATH:+:$PYTHONPATH}"

# One pytest process per component: each project keeps its own rootdir,
# configuration and test helpers.
python -m pytest -q -p no:cacheprovider packages/quant-marketdata/tests
python -m pytest -q -p no:cacheprovider projects/quant-research-platform/tests
python -m pytest -q -p no:cacheprovider projects/equity-pairs-research/tests
python -m pytest -q -p no:cacheprovider projects/index-rebalance-event-study/tests
python -m pytest -q -p no:cacheprovider projects/llm-factor-mining/tests
python -m pytest -q -p no:cacheprovider projects/marketdata-agent/tests
python -m pytest -q -p no:cacheprovider tests
