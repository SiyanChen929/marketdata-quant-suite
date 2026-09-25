from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lfm_helpers import small_market
from llm_factor_mining.cli import benchmark_llm_factory, build_backend, build_parser, credentials_available, main


TINY = ["--n-symbols", "20", "--n-dates", "200", "--snr", "0.4"]


@pytest.fixture()
def no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def test_cli_evaluate_formation_and_validation(capsys) -> None:
    assert main(["evaluate", "-ts_mean(returns, 5)", *TINY]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["report"]["window"] == "formation"
    assert payload["report"]["expression"] == "neg(ts_mean(returns,5))"
    assert main(["evaluate", "-ts_mean(returns, 5)", "--window", "validation", *TINY]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["window"] == "validation"
    assert payload["report"]["window_end"] <= payload["splits"]["validation"]["end"] < payload["splits"]["test"]["start"]
    assert main(["evaluate", "ts_mean(close, 0)", *TINY]) == 1
    assert json.loads(capsys.readouterr().out)["codes"] == ["WINDOW_RANGE"]


def test_cli_evaluate_cannot_reach_the_test_window() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "close", "--window", "test"])
    with pytest.raises(SystemExit):  # windows cannot be moved: there is no --fractions / --start / --end
        parser.parse_args(["evaluate", "close", "--window", "validation", "--fractions", "0.6,0.35,0.05"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "close", "--end", "2033-12-31"])


def test_cli_evaluate_store_requires_a_registered_study(tmp_path) -> None:
    with pytest.raises(SystemExit, match="registry"):
        main(["evaluate", "close", "--data", "store"])
    with pytest.raises(SystemExit, match="not registered"):
        main(["evaluate", "close", "--data", "store", "--registry", str(tmp_path / "r.jsonl"), "--study", "u1"])


def test_cli_search_requires_a_run_directory() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["search", "--proposer", "random"])


def test_cli_search_random(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    code = main(
        ["search", "--proposer", "random", "--budget", "12", "--batch-size", "6", "--run-dir", str(run_dir), *TINY]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["n_trials"] == 12 and "proposer_provenance" in payload["summary"]
    assert (run_dir / "ledger.jsonl").is_file() and (run_dir / "reveal.json").is_file()
    with pytest.raises(SystemExit, match="already holds a ledger"):
        main(["search", "--proposer", "random", "--budget", "12", "--run-dir", str(run_dir), *TINY])


def test_cli_two_step_commit_then_reveal(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    args = ["search", "--proposer", "random", "--budget", "12", "--batch-size", "6", "--run-dir", str(run_dir)]
    assert main([*args, "--no-reveal", *TINY]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert not (run_dir / "reveal.json").exists()
    with pytest.raises(SystemExit, match="does not match"):
        main(["reveal", "--run-dir", str(run_dir), "--commitment", "0" * 64])
    assert main(["reveal", "--run-dir", str(run_dir), "--commitment", summary["commitment"]]) == 0
    revealed = json.loads(capsys.readouterr().out)
    assert revealed["reveal"]["commitment"] == summary["commitment"] and (run_dir / "reveal.json").is_file()
    with pytest.raises(SystemExit, match="second reveal"):
        main(["reveal", "--run-dir", str(run_dir), "--commitment", summary["commitment"]])
    # the refusal was logged and the recorded head follows the ledger
    from llm_factor_mining.protocol import verify_ledger_file

    head = json.loads((run_dir / "provenance.json").read_text())["ledger_head"]
    kinds = [record.kind for record in verify_ledger_file(run_dir / "ledger.jsonl", expected_head=head)]
    assert kinds[-2:] == ["seal_reveal", "seal_refused"]


def test_cli_store_study_flow(tmp_path, monkeypatch, capsys) -> None:
    from quant_marketdata import MarketDataStore

    market = small_market()
    home = tmp_path / "data"
    MarketDataStore(home).write_bars(market.bars, finality="confirmed")
    monkeypatch.setenv("QUANT_DATA_HOME", str(home))
    registry = str(tmp_path / "registry.jsonl")
    symbols = ",".join(market.panel.symbols)
    start, end = str(market.panel.dates[0].date()), str(market.panel.dates[-1].date())
    study = ["--registry", registry, "--study", "u1"]
    assert main(["register-study", *study, "--symbols", symbols, "--start", start, "--end", end]) == 0
    splits = json.loads(capsys.readouterr().out)["splits"]
    with pytest.raises(SystemExit, match="different terms"):
        main(["register-study", *study, "--symbols", symbols, "--start", start, "--end", end, "--fractions", "0.5,0.3,0.2"])

    assert main(["evaluate", "-ts_mean(returns, 5)", "--data", "store", *study, "--window", "validation"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_explorations"] == 1 and payload["report"]["window_end"] <= splits["validation"]["end"]

    run_dir = tmp_path / "run"
    assert main(["search", "--proposer", "random", "--budget", "12", "--run-dir", str(run_dir), "--data", "store", *study]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "only committed" in captured.err and "next_step" in result
    assert not (run_dir / "reveal.json").exists()  # store data are never revealed by search
    commitment = result["summary"]["commitment"]

    assert main(["reveal", "--run-dir", str(run_dir), "--commitment", commitment]) == 0
    capsys.readouterr()
    other = tmp_path / "run2"
    assert main(["search", "--proposer", "random", "--proposer-seed", "3", "--budget", "12", "--run-dir", str(other), "--data", "store", *study]) == 0
    second = json.loads(capsys.readouterr().out)["summary"]["commitment"]
    with pytest.raises(SystemExit, match="maximum of 1"):
        main(["reveal", "--run-dir", str(other), "--commitment", second])

    from llm_factor_mining.protocol import StudyRegistry

    counts = StudyRegistry(registry).summary("u1")
    assert counts["n_reveals"] == 1 and counts["n_commits"] == 2 and counts["n_explorations"] == 1


def test_cli_search_llm_needs_a_backend_and_credentials(tmp_path, no_credentials) -> None:
    run = ["--run-dir", str(tmp_path / "run"), "--budget", "4", *TINY]
    with pytest.raises(SystemExit):
        main(["search", "--proposer", "llm", *run])
    with pytest.raises(SystemExit):
        main(["search", "--proposer", "llm", "--backend", "replay", *run])
    with pytest.raises(FileNotFoundError):
        main(["search", "--proposer", "llm", "--backend", "replay", "--replay-file", str(tmp_path / "none.jsonl"), *run])
    with pytest.raises(SystemExit, match="credentials"):
        main(["search", "--proposer", "llm", "--backend", "anthropic", *run])
    assert not (tmp_path / "run").exists()  # fails before a partial run directory is created


def test_cli_backend_settings(tmp_path, monkeypatch) -> None:
    from llm_factor_mining.proposers import CachedBackend, FakeBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    parser = build_parser()
    base = ["search", "--proposer", "llm", "--run-dir", str(tmp_path / "r")]
    args = parser.parse_args([*base, "--backend", "anthropic", "--max-tokens", "50000"])
    with pytest.raises(SystemExit, match="stream"):
        build_backend(args, default_record=tmp_path / "rec.jsonl")
    args = parser.parse_args([*base, "--backend", "anthropic"])
    with pytest.raises(SystemExit, match="record"):
        build_backend(args)
    assert isinstance(build_backend(args, default_record=tmp_path / "rec.jsonl"), CachedBackend)
    # replay picks the identity from explicit flags (a file may hold several identities)
    path = tmp_path / "mixed.jsonl"
    CachedBackend(FakeBackend([{"proposals": []}], model="a"), path).complete_json("s", "u", {})
    CachedBackend(FakeBackend([{"proposals": []}], model="b"), path, reuse=True).complete_json("s", "u", {})
    replay_args = parser.parse_args([*base, "--backend", "replay", "--replay-file", str(path)])
    with pytest.raises(ValueError, match="identities"):
        build_backend(replay_args)
    explicit = parser.parse_args([*base, "--backend", "replay", "--replay-file", str(path), "--effort", "medium"])
    backend = build_backend(explicit)
    assert backend.identity()["effort"] == "medium" and backend.identity()["model"] == "claude-opus-5"


def test_benchmark_llm_arm_status_without_credentials(tmp_path, no_credentials) -> None:
    assert not credentials_available()
    common = ["benchmark", "--out", str(tmp_path / "o"), "--runs-dir", str(tmp_path / "r")]
    factory, status = benchmark_llm_factory(build_parser().parse_args(common))
    assert factory is None and status.startswith("not run")
    factory, status = benchmark_llm_factory(build_parser().parse_args([*common, "--backend", "anthropic"]))
    assert factory is None and "no credentials" in status
    args = build_parser().parse_args([*common, "--backend", "replay", "--replay-file", str(tmp_path / "x.jsonl")])
    assert benchmark_llm_factory(args)[0] is None
    with pytest.raises(SystemExit):  # output locations are explicit (no cwd-relative defaults)
        build_parser().parse_args(["benchmark"])


def test_cli_benchmark_tiny(tmp_path, capsys) -> None:
    out = tmp_path / "bench"
    code = main(
        [
            "benchmark", "--seeds", "0", "--snr", "0.4", "--null-seeds", "", "--n-symbols", "20", "--n-dates", "220",
            "--budget", "10", "--batch-size", "5", "--out", str(out), "--runs-dir", str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    paths = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(paths["summary_json"]).read_text())
    assert summary["proposers"]["llm"].startswith("not run")
    assert "SYNTHETIC DATA" in Path(paths["summary_md"]).read_text()
    assert (tmp_path / "runs" / "snr0.4_seed0_random" / "ledger.jsonl").is_file()


def test_benchmark_script_is_importable() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_synthetic_benchmark.py"
    assert script.is_file()
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_synthetic_benchmark", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines main() only; nothing runs on import
    assert callable(module.main)
    assert str(script.parents[1] / "src") in sys.path
