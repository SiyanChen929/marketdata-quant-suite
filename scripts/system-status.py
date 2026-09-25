#!/usr/bin/env python3
"""Display the local suite boundary without exposing credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    "shared_data_gateway": ROOT / "packages" / "quant-marketdata",
    "research_platform": ROOT / "projects" / "quant-research-platform",
    "equity_pairs": ROOT / "projects" / "equity-pairs-research",
    "index_event_study": ROOT / "projects" / "index-rebalance-event-study",
    "llm_factor_mining": ROOT / "projects" / "llm-factor-mining",
    "marketdata_agent": ROOT / "projects" / "marketdata-agent",
}


def status() -> dict[str, object]:
    data_home = os.getenv("QUANT_DATA_HOME")
    resolved_home = Path(data_home).expanduser().resolve() if data_home else None
    return {
        "provider": "MarketData",
        "credential_configured": bool(os.getenv("MARKETDATA_TOKEN")),
        "data_home": str(resolved_home) if resolved_home else None,
        "data_home_is_external": bool(resolved_home and ROOT not in resolved_home.parents and resolved_home != ROOT),
        "stores": {
            "confirmed": str(resolved_home / "lake" / "confirmed") if resolved_home else None,
            "provisional": str(resolved_home / "staging" / "provisional") if resolved_home else None,
        },
        "components": {name: path.is_dir() for name, path in COMPONENTS.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    result = status()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("MarketData Quant Suite")
    print(f"provider: {result['provider']}")
    print(f"credential configured: {result['credential_configured']}")
    print(f"external data home: {result['data_home'] or 'not configured'}")
    print(f"external boundary valid: {result['data_home_is_external']}")
    for name, ready in result["components"].items():
        print(f"{name}: {'ready' if ready else 'missing'}")


if __name__ == "__main__":
    main()
