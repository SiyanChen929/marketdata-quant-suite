from __future__ import annotations

import json

from marketdata_agent import AuditLog, DEFAULT_TOOL_NAMES
from marketdata_agent.cli import main


def test_tools_command_prints_claude_format_tools(capsys):
    assert main(["tools", "--compact"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert [tool["name"] for tool in tools] == list(DEFAULT_TOOL_NAMES)
    assert all(tool["strict"] is True for tool in tools)


def test_verify_audit_command(tmp_path, capsys):
    path = tmp_path / "log.jsonl"
    log = AuditLog(path, fsync=False)
    log.append("episode_start", {"as_of": "2023-06-30"})
    log.append("final_answer", {"text": "ok"})
    head = log.head
    assert main(["verify-audit", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == 2
    assert main(["verify-audit", str(path), "--records", "2", "--head", head.head_hash]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(["verify-audit", str(path), "--records", "3", "--head", head.head_hash]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert main(["verify-audit", str(path), "--records", "2"]) == 2
    capsys.readouterr()
    path.write_text(path.read_text().replace('"ok"', '"no"'), encoding="utf-8")
    assert main(["verify-audit", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["line"] == 2


def _scripted_factory(args):
    from marketdata_agent import ScriptedBackend, ScriptedTurn

    def policy(state):
        if state.step == 0:
            return ScriptedTurn(tool_calls=(("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"}),))
        result = state.last_results[0]
        return ScriptedTurn(text=f"ANSWER: {result.payload['simple_return'] * 100:.2f}% [r:{result.result_id}]")

    return ScriptedBackend(policy, name="cli-test")


def test_ask_prints_answer_citations_grounding_and_a_verified_audit(tmp_path, capsys):
    code = main(
        ["ask", "What was SYN01's return in H1 2023?", "--as-of", "2023-06-30", "--run-dir", str(tmp_path)],
        backend_factory=_scripted_factory,
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.startswith("ANSWER: ") and "Citations:" in out and "period_return" in out
    assert "Grounding: 1/1 numeric claims supported (100.0%)" in out
    assert "synthetic panel (not real market prices)" in out and "verified" in out
    audits = list(tmp_path.glob("*.audit.jsonl"))
    assert len(audits) == 1 and list(tmp_path.glob("*.transcript.json"))


def test_ask_rejects_a_malformed_as_of(tmp_path, capsys):
    assert main(["ask", "q", "--as-of", "30/06/2023", "--run-dir", str(tmp_path)], backend_factory=_scripted_factory) == 2
    capsys.readouterr()


def test_ask_defaults_to_fallback_on_and_bench_to_off():
    from marketdata_agent.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["ask", "q", "--as-of", "2023-06-30"]).fallback is True
    assert parser.parse_args(["ask", "q", "--as-of", "2023-06-30", "--no-fallback"]).fallback is False
    assert parser.parse_args(["ask", "q", "--as-of", "2023-06-30", "--fallback"]).fallback is True
    assert parser.parse_args(["bench", "run", "--agent", "anthropic", "--fallback"]).fallback is True
    assert parser.parse_args(["bench", "run", "--agent", "anthropic"]).fallback is False
    assert parser.parse_args(["ask", "q", "--as-of", "2023-06-30"]).model == "claude-opus-5"


def test_bench_run_and_tasks_commands(tmp_path, capsys):
    out_dir = tmp_path / "oracle"
    assert main(["bench", "run", "--agent", "oracle", "--tasks", "12", "--out", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "oracle: accuracy 12/12 (100.0%)" in printed and "audit verified" in printed
    assert "harness-validation baselines on synthetic data" in printed
    assert json.loads((out_dir / "summary.json").read_text())["suite"]["n_tasks"] == 12
    suite_path = tmp_path / "suite.json"
    assert main(["bench", "tasks", "--tasks", "6", "--out", str(suite_path)]) == 0
    assert len(json.loads(suite_path.read_text())["tasks"]) == 6


def test_user_mistakes_are_one_line_errors_with_exit_status_2(tmp_path, capsys):
    out_dir = tmp_path / "oracle"
    assert main(["bench", "run", "--agent", "oracle", "--tasks", "6", "--out", str(out_dir)]) == 0
    capsys.readouterr()
    assert main(["bench", "run", "--agent", "oracle", "--tasks", "6", "--out", str(out_dir)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "--overwrite" in err and "Traceback" not in err
    missing = str(tmp_path / "absent.jsonl")
    assert main(["bench", "run", "--agent", "anthropic", "--tasks", "2", "--replay", missing, "--out", str(tmp_path / "r")]) == 2
    assert "no recording" in capsys.readouterr().err
    assert main(["bench", "run", "--agent", "oracle", "--no-clock", "--out", str(tmp_path / "x")]) == 2
    assert "apply to --agent anthropic only" in capsys.readouterr().err
    assert main(["bench", "run", "--agent", "anthropic", "--max-tokens", "0", "--out", str(tmp_path / "y")]) == 2
    assert "max_tokens" in capsys.readouterr().err


def test_verify_audit_reads_the_anchor_from_a_committed_summary(tmp_path, capsys):
    out_dir = tmp_path / "oracle"
    assert main(["bench", "run", "--agent", "oracle", "--tasks", "6", "--out", str(out_dir)]) == 0
    capsys.readouterr()
    log = str(out_dir / "audit" / "episodes.audit.jsonl")
    assert main(["verify-audit", log, "--summary", str(out_dir / "summary.json")]) == 0
    assert json.loads(capsys.readouterr().out)["anchor"]["records"] > 0
    lines = (out_dir / "audit" / "episodes.audit.jsonl").read_text().splitlines()
    (out_dir / "audit" / "episodes.audit.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert main(["verify-audit", log, "--summary", str(out_dir / "summary.json")]) == 1  # a dropped tail is caught
    capsys.readouterr()


def _failing_factory(kind):
    from marketdata_agent.backends import BackendError

    class Failing:
        name = "failing"

        def describe(self):
            return {"backend": "fake", "model": "fake-model", "fallback": False}

        def step(self, system, messages, tools):
            raise BackendError(kind, "simulated")

    return lambda args: Failing()


def test_a_failed_llm_benchmark_exits_non_zero_and_is_marked_invalid(tmp_path, capsys):
    out_dir = tmp_path / "anthropic"
    code = main(["bench", "run", "--agent", "anthropic", "--tasks", "4", "--out", str(out_dir)], backend_factory=_failing_factory("authentication"))
    printed = capsys.readouterr().out
    assert code == 3 and "INVALID RUN" in printed
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["valid"] is False and summary["tasks_scored"] == 0


def test_ask_writes_a_fresh_episode_per_call_and_hints_credentials_only_for_credential_errors(tmp_path, capsys):
    args = ["ask", "What was SYN01's return in H1 2023?", "--as-of", "2023-06-30", "--run-dir", str(tmp_path)]
    assert main(args, backend_factory=_scripted_factory) == 0
    assert main(args, backend_factory=_scripted_factory) == 0
    assert len(list(tmp_path.glob("*.transcript.json"))) == 2 and len(list(tmp_path.glob("*.audit.jsonl"))) == 2
    capsys.readouterr()
    assert main(args, backend_factory=_failing_factory("server_error")) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" not in err and "server_error" in err
    assert main(args, backend_factory=_failing_factory("missing_credentials")) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_eval_seed_builds_the_preregistered_evaluation_suite_and_records_it_before_the_run(tmp_path, capsys):
    """Regression: `bench run --seed` was documented as the evaluation command but ran on the development panel."""

    from marketdata_agent.bench import DEFAULT_SEED, generate_suite
    from marketdata_agent.bench.design import EVALUATION_COUNTS

    path = tmp_path / "eval.json"
    assert main(["bench", "tasks", "--eval-seed", "1001", "--out", str(path)]) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    development = generate_suite(DEFAULT_SEED)
    assert len(payload["tasks"]) == sum(EVALUATION_COUNTS.values()) == 234
    assert payload["seed"] == 1001 and payload["dataset"] != development.dataset.to_dict()
    assert not {t["metadata"]["item_key"] for t in payload["tasks"]} & set(development.item_keys())

    out = tmp_path / "run"
    assert main(["bench", "run", "--agent", "oracle", "--eval-seed", "1001", "--tasks", "6", "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"]["seed"] == 1001 and manifest["suite"]["n_tasks"] == 6
    capsys.readouterr()
    assert main(["bench", "run", "--agent", "oracle", "--seed", "7", "--tasks", "6", "--out", str(tmp_path / "dev")]) == 0
    assert "use --eval-seed for an evaluation suite" in capsys.readouterr().err
