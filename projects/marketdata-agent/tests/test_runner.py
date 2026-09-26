"""Runner tests: files, audit anchoring, reproducibility, validity, retries, resume, and baseline behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marketdata_agent import verify_chain
from marketdata_agent.audit import ChainHead
from marketdata_agent.backends import AnthropicConfig, BackendError, BackendTurn, ScriptedBackend, ScriptedTurn
from marketdata_agent.bench import (
    BASELINE_NAMES,
    DEFAULT_SEED,
    anthropic_agent,
    baseline_agent,
    generate_suite,
    run_agent,
    run_suite,
)
from marketdata_agent.bench.baselines import baseline_backend, ungrounded_mode
from marketdata_agent.bench.runner import AgentSpec


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_suite():
    return generate_suite(DEFAULT_SEED).subset(36)


@pytest.fixture(scope="module")
def baseline_runs(small_suite, tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    combined = run_suite([baseline_agent(n) for n in BASELINE_NAMES], small_suite, root)
    return root, combined


def test_runner_writes_verified_outputs_anchored_in_the_summary(baseline_runs, small_suite):
    root, combined = baseline_runs
    for name in BASELINE_NAMES:
        run_dir = root / name
        lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(small_suite)
        first = json.loads(lines[0])
        assert set(first) == {"agent", "task", "episode", "score"} and first["agent"] == name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["valid"] and summary["audit"]["ok"] and summary["suite"]["sha256"] == small_suite.sha256()
        assert summary["banner"].startswith("harness-validation baselines on synthetic data")
        assert summary["status"]["completed"] == len(small_suite) and summary["audit"]["timestamps"] == "fixed"
        anchor = ChainHead(summary["audit"]["records"], summary["audit"]["head_hash"])
        assert verify_chain(run_dir / "audit" / "episodes.audit.jsonl", expected_head=anchor).ok
        assert "## Episode status" in (run_dir / "summary.md").read_text(encoding="utf-8")
    markdown = (root / "summary.md").read_text(encoding="utf-8")
    assert "LLM agent results pending (requires ANTHROPIC_API_KEY)" in markdown and "| `oracle` |" in markdown
    assert set(combined["agents"]) == set(BASELINE_NAMES) and combined["valid"]


def test_oracle_validates_the_harness(baseline_runs):
    overall = baseline_runs[1]["agents"]["oracle"]["overall"]
    assert overall["accuracy"]["rate"] == 1.0
    assert overall["grounding_rate"]["rate"] == 1.0 and overall["citation_rate"]["rate"] == 1.0
    assert overall["lookahead_attempt_episodes"]["k"] == 0 and overall["leak_episodes"]["k"] == 0
    assert overall["denied_call_episodes"]["k"] == 0 and overall["hindsight_match"]["k"] == 0
    assert overall["false_abstention"]["k"] == 0 and overall["over_refusal"]["k"] == 0


def test_clock_blocks_the_naive_policy_and_the_ablation_leaks(baseline_runs):
    agents = baseline_runs[1]["agents"]
    naive, ablation = agents["lookahead_naive"]["overall"], agents["no_guard"]["overall"]
    assert naive["lookahead_attempt_episodes"]["k"] > 0 and naive["leak_episodes"]["k"] == 0
    assert naive["order_execution_attempt_episodes"]["k"] > 0
    # Look-ahead attempts are counted against the nominal as-of date also when the clock is off.
    assert ablation["leak_episodes"]["k"] > 0
    assert ablation["lookahead_attempt_episodes"]["k"] >= ablation["leak_episodes"]["k"]
    assert ablation["hindsight_match_numeric"]["rate"] == 1.0
    assert agents["no_guard"]["by_category"]["pit_trap"]["accuracy"]["k"] == 0
    assert naive["accuracy"]["rate"] > ablation["accuracy"]["rate"]


def test_the_trivial_policy_scores_on_traps_only(baseline_runs):
    agent = baseline_runs[1]["agents"]["abstain_or_refuse"]
    for category in ("pit_trap", "policy_trap", "unknown_symbol"):
        metrics = agent["by_category"][category]["accuracy"]
        assert metrics["k"] == metrics["n"]
    overall = agent["overall"]
    assert overall["false_abstention"]["rate"] == 1.0 and overall["mean_tool_calls"] == 0


def test_grounding_verifier_catches_every_ungrounded_answer(baseline_runs, small_suite, tmp_path):
    overall = baseline_runs[1]["agents"]["ungrounded"]["overall"]
    assert overall["grounding_rate"]["n"] > 0 and overall["grounding_rate"]["k"] == 0
    assert overall["unknown_citations"] > 0
    assert {"fabricated_citation", "uncited", "misreported", "near_miss"} <= {ungrounded_mode(t) for t in small_suite}
    # Every task with a reference plan ran it, so the verifier is checked against real results.
    records = [json.loads(line) for line in (baseline_runs[0] / "ungrounded" / "episodes.jsonl").read_text().splitlines()]
    planned = {t.id for t in small_suite if t.reference_plan}
    assert all(r["episode"]["result_ids"] for r in records if r["task"]["id"] in planned)


def test_episode_files_and_audit_logs_are_byte_reproducible(small_suite, tmp_path):
    subset = small_suite.subset(12)
    first = run_agent(baseline_agent("oracle"), subset, tmp_path / "a")
    second = run_agent(baseline_agent("oracle"), subset, tmp_path / "b")
    assert (tmp_path / "a" / "episodes.jsonl").read_bytes() == (tmp_path / "b" / "episodes.jsonl").read_bytes()
    assert (tmp_path / "a" / "audit" / "episodes.audit.jsonl").read_bytes() == (tmp_path / "b" / "audit" / "episodes.audit.jsonl").read_bytes()
    assert first.summary["metrics"] == second.summary["metrics"] and first.summary["audit"] == second.summary["audit"]
    with pytest.raises(FileExistsError):
        run_agent(baseline_agent("oracle"), subset, tmp_path / "a")
    assert run_agent(baseline_agent("oracle"), subset, tmp_path / "a", overwrite=True).summary["audit"]["ok"]


def test_the_oracle_is_unaffected_by_the_clock_ablation(tmp_path):
    """Regression for A1: a policy that never names a later date must not leak, be denied, or lose accuracy."""

    suite = generate_suite(DEFAULT_SEED)
    spec = baseline_agent("oracle_no_clock")
    assert spec.enforce_clock is False
    overall = run_agent(spec, suite, tmp_path / "a1").summary["metrics"]["overall"]
    assert overall["accuracy"]["k"] == len(suite)
    assert overall["leak_episodes"]["k"] == 0 and overall["denied_call_episodes"]["k"] == 0
    assert overall["lookahead_attempt_episodes"]["k"] == 0 and overall["grounding_rate"]["rate"] == 1.0
    committed = json.loads((PROJECT / "results" / "benchmark" / "summary.json").read_text(encoding="utf-8"))["agents"]
    assert committed["oracle_no_clock"]["overall"]["accuracy"] == committed["oracle"]["overall"]["accuracy"]
    assert committed["oracle_no_clock"]["overall"]["grounding_rate"] == committed["oracle"]["overall"]["grounding_rate"]


def test_the_committed_oracle_run_is_reproduced_exactly(tmp_path):
    committed = json.loads((PROJECT / "results" / "benchmark" / "summary.json").read_text(encoding="utf-8"))["agents"]["oracle"]
    rerun = run_agent(baseline_agent("oracle"), generate_suite(DEFAULT_SEED), tmp_path / "oracle").summary
    assert rerun["episodes_sha256"] == committed["episodes_sha256"]
    assert rerun["audit"]["head_hash"] == committed["audit"]["head_hash"]
    assert rerun["audit"]["records"] == committed["audit"]["records"]


class _AbstainingClient:
    """Fake Claude client that always abstains (exercises the LLM path offline)."""

    def __init__(self, model: str | None = None):
        self.messages = SimpleNamespace(create=self._create)
        self.calls = 0
        self.model = model

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ANSWER: INSUFFICIENT_DATA")],
            stop_reason="end_turn",
            model=self.model or kwargs["model"],
            usage=SimpleNamespace(input_tokens=50, output_tokens=5),
            stop_details=None,
            _request_id="req_fake",
        )


def test_llm_agent_path_records_and_replays_offline(small_suite, tmp_path):
    subset = small_suite.subset(8)
    client = _AbstainingClient()
    recorded = run_agent(
        anthropic_agent(AnthropicConfig(), record_path=tmp_path / "turns.jsonl", client=client), subset, tmp_path / "live"
    )
    assert client.calls == len(subset) and recorded.summary["is_llm"] and recorded.valid
    assert recorded.summary["banner"].startswith("language-model run on synthetic data")
    assert recorded.summary["audit"]["timestamps"] == "wall_clock"
    assert recorded.summary["metrics"]["overall"]["input_tokens"] == 50 * len(subset)
    replayed = run_agent(anthropic_agent(replay_path=tmp_path / "turns.jsonl"), subset, tmp_path / "replay")
    assert replayed.summary["metrics"]["overall"]["accuracy"] == recorded.summary["metrics"]["overall"]["accuracy"]
    assert [s.correct for s in replayed.scores] == [s.correct for s in recorded.scores]


def test_resume_reuses_recorded_turns_and_keeps_the_previous_outputs(small_suite, tmp_path):
    subset = small_suite.subset(6)
    client = _AbstainingClient()
    record = tmp_path / "run" / "recordings.jsonl"
    run_agent(anthropic_agent(AnthropicConfig(), record_path=record, client=client), subset, tmp_path / "run")
    assert client.calls == len(subset)
    again = _AbstainingClient()
    resumed = run_agent(anthropic_agent(AnthropicConfig(), record_path=record, client=again), subset, tmp_path / "run", resume=True)
    assert again.calls == 0 and resumed.valid
    assert resumed.summary["superseded"] == "superseded/run-1"
    assert (tmp_path / "run" / "superseded" / "run-1" / "episodes.jsonl").is_file()
    assert len(record.read_text().splitlines()) == len(subset)  # nothing re-recorded, nothing truncated


class _FailingBackend:
    """Raises a BackendError for the first ``failures`` calls, then abstains."""

    name = "failing"

    def __init__(self, kind: str, *, retryable: bool, failures: int):
        self.kind, self.retryable, self.failures, self.calls = kind, retryable, failures, 0

    def describe(self):
        return {"backend": "fake", "model": "fake-model", "fallback": False}

    def step(self, system, messages, tools):
        self.calls += 1
        if self.calls <= self.failures:
            raise BackendError(self.kind, "simulated", retryable=self.retryable)
        content = [{"type": "text", "text": "ANSWER: INSUFFICIENT_DATA"}]
        return BackendTurn.from_content(content, stop_reason="end_turn", served_model="fake-model", requested_model="fake-model")


def _spec(backend) -> AgentSpec:
    return AgentSpec("fake", lambda task: backend, "fake LLM", is_llm=True)


def test_an_unrecovered_backend_error_invalidates_the_run_and_is_never_scored(small_suite, tmp_path):
    """Regression: a run without credentials used to be written up as '0% accuracy' and exit 0."""

    backend = _FailingBackend("missing_credentials", retryable=False, failures=10**6)
    result = run_agent(_spec(backend), small_suite.subset(6), tmp_path / "bad")
    summary = result.summary
    assert not result.valid and summary["tasks_scored"] == 0 and summary["status"]["backend_error"] == 1
    assert summary["failure"]["error"]["kind"] == "missing_credentials" and backend.calls == 1
    assert summary["banner"].startswith("INVALID RUN") and summary["metrics"]["overall"]["n"] == 0
    markdown = (tmp_path / "bad" / "summary.md").read_text(encoding="utf-8")
    assert "INVALID RUN" in markdown and "| `backend_error` | 1 |" in markdown


def test_retryable_errors_are_retried_with_backoff(small_suite, tmp_path):
    waits: list[float] = []
    backend = _FailingBackend("rate_limit", retryable=True, failures=2)
    result = run_agent(_spec(backend), small_suite.subset(3), tmp_path / "retry", backoff_seconds=1.5, sleep=waits.append)
    assert result.valid and result.summary["retries"] == 2 and waits == [1.5, 3.0]
    assert result.summary["tasks_scored"] == 3
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "retry" / "audit" / "episodes.audit.jsonl").read_text().splitlines()]
    assert kinds.count("episode_retry") == 2


def test_a_served_model_other_than_the_requested_one_invalidates_a_run_without_fallback(small_suite, tmp_path):
    client = _AbstainingClient(model="claude-other")
    result = run_agent(anthropic_agent(AnthropicConfig(), client=client), small_suite.subset(3), tmp_path / "mismatch")
    # Regression: the run used to finish every task before being declared invalid; it now stops at the first mismatch.
    assert not result.valid and result.summary["metrics"]["overall"]["served_model_mismatch_episodes"] == 1
    assert result.summary["tasks_scored"] == 1 and result.summary["stopped"]["reason"] == "served_model_mismatch"
    assert "served by a model other than the requested one" in result.summary["banner"]
    assert "the run stopped after task" in result.summary["banner"]


def test_a_leak_with_the_clock_enforced_invalidates_and_stops_the_run(small_suite, tmp_path, monkeypatch):
    """H2a: one leak with the clock enforced is a harness defect. The view makes it impossible, so it is simulated."""

    import dataclasses

    import marketdata_agent.bench.runner as runner_module

    real = runner_module.score_episode
    monkeypatch.setattr(runner_module, "score_episode", lambda task, episode: dataclasses.replace(real(task, episode), leaked_results=1))
    result = run_agent(baseline_agent("oracle"), small_suite.subset(3), tmp_path / "leak")
    assert not result.valid and result.summary["tasks_scored"] == 1
    assert result.summary["stopped"]["reason"] == "leak_with_clock_enforced" and "H2a" in result.summary["banner"]
    ablated = run_agent(baseline_agent("no_guard"), small_suite.subset(3), tmp_path / "ablated")
    assert ablated.valid and ablated.summary["tasks_scored"] == 3  # leaks are the point of the ablation


def test_the_manifest_is_written_before_the_first_episode_and_guards_resume(small_suite, tmp_path):
    out = tmp_path / "run"
    seen: list[bool] = []

    def factory(task):
        seen.append((out / "manifest.json").is_file())
        return ScriptedBackend([ScriptedTurn(text="ANSWER: INSUFFICIENT_DATA")], name="probe")

    subset = small_suite.subset(2)
    run_agent(AgentSpec("probe", factory, "probe"), subset, out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert seen and all(seen) and manifest["suite"]["sha256"] == subset.sha256()
    with pytest.raises(ValueError, match="different suite"):
        run_agent(AgentSpec("probe", factory, "probe"), small_suite.subset(3), out, resume=True)
    assert (out / "episodes.jsonl").is_file() and not (out / "superseded").exists()  # refused before moving anything


def test_arms_select_the_prompt_the_tools_and_the_clock(small_suite, tmp_path):
    seen: list[tuple[str, list[str]]] = []

    class Spy(ScriptedBackend):
        def step(self, system, messages, tools):
            seen.append((system, [tool["name"] for tool in tools]))
            return super().step(system, messages, tools)

    def factory(task):
        return Spy([ScriptedTurn(text="ANSWER: INSUFFICIENT_DATA")], name="spy")

    subset = small_suite.subset(2)
    run_agent(AgentSpec("decoy", factory, "decoy arm", tools="decoy"), subset, tmp_path / "decoy")
    assert seen[-1][1][-1] == "execute_order"
    run_agent(
        AgentSpec("closed", factory, "closed book", tools="none", prompt_version="copilot_closed_book_v1"),
        subset,
        tmp_path / "closed",
    )
    assert seen[-1][1] == [] and "No tools are available" in seen[-1][0]
    summary = run_agent(AgentSpec("a1", factory, "A1", enforce_clock=False), subset, tmp_path / "a1").summary
    assert summary["arm"] == {"clock_enforced": False, "prompt_version": "copilot_system_v1", "tools": "default"}
