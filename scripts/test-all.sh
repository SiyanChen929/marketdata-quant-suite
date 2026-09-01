#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SUITE_ROOT/packages/quant-marketdata/src:$SUITE_ROOT/projects/quant-research-platform/src:$SUITE_ROOT/projects/equity-pairs-research/src:$SUITE_ROOT/projects/index-rebalance-event-study/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q -p no:cacheprovider packages/quant-marketdata/tests
python -m pytest -q -p no:cacheprovider projects/quant-research-platform/tests
python -m pytest -q -p no:cacheprovider projects/equity-pairs-research/tests
python -m pytest -q -p no:cacheprovider projects/index-rebalance-event-study/tests
python -m pytest -q -p no:cacheprovider tests
