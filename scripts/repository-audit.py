#!/usr/bin/env python3
"""Fail when the public suite contains artifacts, secrets, or provider drift."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
IGNORED_PARTS = {".git", ".venv"}
GENERATED_CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "cache",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".sh",
    ".example",
}
CODE_SUFFIXES = {".py", ".sh", ".toml", ".yml", ".yaml"}
SECRET_PATTERNS = {
    "embedded MarketData token": re.compile(
        r"MARKETDATA_(?:TOKEN|API_KEY)\s*=\s*(?!replace_me|[\"']?\$|os\.getenv)"
        r"[A-Za-z0-9+/=_-]{20,}"
    ),
    "embedded FRED key": re.compile(
        r"FRED_API_KEY\s*=\s*(?!replace_me|[\"']?\$|os\.getenv)[A-Za-z0-9_-]{20,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
FORBIDDEN_PRICE_IMPORTS = re.compile(
    r"^\s*(?:from\s+(?:yfinance|alpha_vantage|pandas_datareader)|"
    r"import\s+(?:yfinance|alpha_vantage|pandas_datareader))\b",
    re.MULTILINE,
)
DIRECT_MARKETDATA_ENDPOINTS = {
    "stock-candle": re.compile(r"(?:^|[^A-Za-z])/?stocks[/\\]+candles(?:[/\\]|\b)", re.I),
    "option-chain": re.compile(r"(?:^|[^A-Za-z])/?options[/\\]+chain(?:[/\\]|\b)", re.I),
}
DIRECT_HTTP_CALL = re.compile(
    r"\b(?:requests|session|self\.session)\.(?:get|post|put|patch|request)\s*\("
)
FORBIDDEN_FILENAMES = {
    "fred_api_key.txt",
    "marketdata_token.txt",
    "marketdata_api_key.txt",
}
SHARED_GATEWAY_ROOT = Path("packages/quant-marketdata")
PLATFORM_ADAPTER_CONTRACTS = {
    Path(
        "projects/quant-research-platform/src/quant_system/data/marketdata_provider.py"
    ): "get_daily_bars",
    Path(
        "projects/quant-research-platform/src/quant_system/data/intraday_provider.py"
    ): "get_stock_bars",
    Path(
        "projects/quant-research-platform/src/quant_system/options/chain.py"
    ): "get_option_chain",
}


def _standalone_tracked_files(root: Path) -> list[Path] | None:
    """Return tracked files only when ``root`` is itself a Git worktree root.

    CI runs the audit after tests, so untracked interpreter caches created by the
    test process must not be mistaken for committed repository content.
    """

    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top).resolve() != root.resolve():
            return None
        output = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def repository_files(root: Path = ROOT) -> list[Path]:
    tracked = _standalone_tracked_files(root)
    candidates = tracked if tracked is not None else list(root.rglob("*"))
    return [
        path
        for path in candidates
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
        and not GENERATED_CACHE_DIRS.intersection(path.relative_to(root).parts)
    ]


def generated_artifact_failures(root: Path = ROOT) -> list[str]:
    tracked = _standalone_tracked_files(root)
    candidates = tracked if tracked is not None else list(root.rglob("*"))
    cache_paths: set[Path] = set()
    failures: list[str] = []
    for path in candidates:
        relative = path.relative_to(root)
        if IGNORED_PARTS.intersection(relative.parts):
            continue
        inside_generated_cache = False
        for index, part in enumerate(relative.parts):
            if part in GENERATED_CACHE_DIRS:
                cache_paths.add(Path(*relative.parts[: index + 1]))
                inside_generated_cache = True
                break
        if path.is_file() and path.suffix.lower() == ".pyc" and not inside_generated_cache:
            failures.append(f"compiled Python artifact: {relative}")
        if path.is_file() and path.name == ".DS_Store":
            failures.append(f"Finder metadata artifact: {relative}")
    failures.extend(f"generated cache directory: {path}" for path in sorted(cache_paths))
    return failures


def direct_endpoint_failures(root: Path = ROOT) -> list[str]:
    failures: list[str] = []
    for path in repository_files(root):
        relative = path.relative_to(root)
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        if relative == Path("scripts/repository-audit.py"):
            continue
        if relative.is_relative_to(SHARED_GATEWAY_ROOT):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in DIRECT_MARKETDATA_ENDPOINTS.items():
            if pattern.search(text):
                failures.append(
                    f"direct MarketData {label} endpoint outside shared gateway: {relative}"
                )
    return failures


def platform_adapter_failures(root: Path = ROOT) -> list[str]:
    failures: list[str] = []
    for relative, required_method in PLATFORM_ADAPTER_CONTRACTS.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing shared-gateway platform adapter: {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not re.search(r"(?:from\s+quant_marketdata\s+import|import\s+quant_marketdata\b)", text):
            failures.append(f"platform adapter does not import quant_marketdata: {relative}")
        if "MarketDataClient" not in text or f".{required_method}(" not in text:
            failures.append(
                f"platform adapter does not use MarketDataClient.{required_method}: {relative}"
            )
        if DIRECT_HTTP_CALL.search(text):
            failures.append(f"platform adapter contains a direct HTTP call: {relative}")
    return failures


def audit_failures(root: Path = ROOT) -> tuple[list[str], int]:
    failures = generated_artifact_failures(root)
    files = repository_files(root)
    for path in files:
        relative = path.relative_to(root)
        if path.name.lower() in FORBIDDEN_FILENAMES:
            failures.append(f"credential file name: {relative}")
        if path.stat().st_size > MAX_BYTES:
            failures.append(
                f"large file: {relative} ({path.stat().st_size / 1024 / 1024:.1f} MiB)"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
        if "src" in relative.parts and FORBIDDEN_PRICE_IMPORTS.search(text):
            failures.append(f"unapproved market-price provider import: {relative}")
    failures.extend(direct_endpoint_failures(root))
    failures.extend(platform_adapter_failures(root))
    return sorted(set(failures)), len(files)


def main() -> int:
    failures, file_count = audit_failures(ROOT)
    if failures:
        print("Repository audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"Repository audit passed: {file_count} files, no generated residue, "
        "no file over 5 MiB, no embedded secrets, one shared price gateway."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
