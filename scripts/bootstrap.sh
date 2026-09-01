#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e "packages/quant-marketdata[dev]"
python -m pip install -e "projects/quant-research-platform[dev]"
python -m pip install -e "projects/equity-pairs-research[dev]"
python -m pip install -e "projects/index-rebalance-event-study[dev]"
