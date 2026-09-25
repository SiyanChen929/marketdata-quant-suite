from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from marketdata_agent import (
    PENDING_HUMAN_APPROVAL,
    Code,
    OrderProposal,
    ProposalBook,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
    default_registry,
    read_records,
    verify_chain,
)
from marketdata_agent.tools import PROPOSAL_NOTICE

from conftest import AS_OF, closes


def _tool_use(block_id: str, name: str, args: object) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def test_one_user_message_with_one_result_per_tool_use_in_order(audited_runtime):
    blocks = [
        SimpleNamespace(type="thinking", thinking="", signature="sig"),
        SimpleNamespace(type="text", text="Let me check."),
        _tool_use("toolu_a", "period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"}),
        {"type": "tool_use", "id": "toolu_b", "name": "period_return", "input": {"symbol": "SYN01", "start": "2023-01-02", "end": "2023-07-31"}},
        _tool_use("toolu_c", "get_news", {}),
    ]
    results = audited_runtime.handle_tool_uses(blocks)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b", "toolu_c"]
    assert [r["is_error"] for r in results] == [False, True, True]
    assert all(set(r) == {"type", "tool_use_id", "content", "is_error"} and r["type"] == "tool_result" for r in results)
    rid = audited_runtime.outcomes[0].result.result_id
    assert f"[r:{rid}]" in results[0]["content"]
    assert results[1]["content"].startswith("[error:lookahead_violation]")
    assert results[2]["content"].startswith("[error:unknown_tool]")
    assert set(audited_runtime.results()) == {rid}


def test_every_attempt_is_audited_and_the_chain_verifies(audited_runtime, audit):
    audited_runtime.call_tool("list_symbols", {}, tool_use_id="t1")
    audited_runtime.call_tool("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "2023-12-29"}, tool_use_id="t2")
    records = read_records(audit.path)
    kinds = [record["kind"] for record in records]
    assert kinds == ["policy_decision", "tool_call", "policy_decision", "tool_call"]
    assert records[2]["payload"]["decision"]["code"] == "lookahead_violation"
    assert records[3]["payload"]["is_error"] is True and records[3]["payload"]["provenance"] is None
    assert records[1]["payload"]["provenance"]["finality"] == "confirmed"
    assert all(record["episode_id"] == "ep-test" for record in records)
    assert verify_chain(audit.path).ok


def test_propose_order_records_a_pending_proposal_and_never_executes(audited_runtime, audit, panel):
    args = {"symbol": "syn01", "side": "buy", "quantity": 100, "rationale": "Momentum over H1 [r:abc123abc123]."}
    outcome = audited_runtime.call_tool("propose_order", args, tool_use_id="toolu_order")
    assert not outcome.is_error, outcome.content
    payload = outcome.result.payload
    assert payload["status"] == PENDING_HUMAN_APPROVAL
    assert payload["executed"] is False and payload["requires_human_confirmation"] is True
    assert payload["notice"] == PROPOSAL_NOTICE
    last_close = closes(panel, "SYN01", end=AS_OF.isoformat()).iloc[-1]
    assert payload["reference_close"] == pytest.approx(last_close)
    assert payload["reference_date"] == AS_OF.isoformat()
    assert "first session after 2023-06-30" in payload["earliest_execution"]

    proposals = audited_runtime.context.proposals.pending()
    assert len(proposals) == 1 and proposals[0].symbol == "SYN01"
    records = read_records(audit.path)
    proposal_records = [r for r in records if r["kind"] == "order_proposal"]
    assert len(proposal_records) == 1
    assert proposal_records[0]["payload"]["result_id"] == outcome.result.result_id
    assert proposal_records[0]["payload"]["status"] == PENDING_HUMAN_APPROVAL

    # No object on the path from the model to the proposal can transmit an order.
    for obj in (audited_runtime, audited_runtime.context, audited_runtime.context.proposals, proposals[0]):
        for name in ("execute", "submit", "place", "send", "route", "execute_order", "place_order", "submit_order"):
            assert not hasattr(obj, name), (type(obj).__name__, name)
    assert not hasattr(ProposalBook, "execute") and not hasattr(OrderProposal, "execute")


def test_order_rationale_limits(runtime):
    blank = runtime.call_tool("propose_order", {"symbol": "SYN01", "side": "sell", "quantity": 1, "rationale": "  "})
    assert blank.error_code == Code.INVALID_ARGUMENTS
    long = runtime.call_tool("propose_order", {"symbol": "SYN01", "side": "sell", "quantity": 1, "rationale": "x" * 2001})
    assert long.error_code == Code.INVALID_ARGUMENTS
    assert len(runtime.context.proposals) == 0


def test_metrics_count_lookahead_attempts_and_denials(runtime):
    runtime.call_tool("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "2023-07-03"})
    runtime.call_tool("realized_volatility", {"symbol": "SYN01", "window": 20, "end": "2023-09-29"})
    runtime.call_tool("compare_returns", {"symbols": [f"S{i}" for i in range(9)], "start": "2023-01-02", "end": "2024-01-02"})
    runtime.call_tool("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"})
    runtime.call_tool("period_return", {"symbol": "ZZZ", "start": "2023-01-02", "end": "latest"})
    metrics = runtime.metrics()
    assert metrics["tool_calls"] == 5
    assert metrics["lookahead_attempts"] == 3
    assert metrics["denied"] == 3
    assert metrics["succeeded"] == 1
    assert metrics["errors_by_code"] == {"unknown_symbol": 1}
    assert metrics["denials_by_code"] == {"lookahead_violation": 3}


def _probe_registry(handler) -> tuple[ToolRegistry, set[str]]:
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    probe = ToolSpec("probe", "Probe tool.", schema, handler)
    return ToolRegistry([probe]), {"probe"}


def test_handler_failures_are_contained(source):
    from marketdata_agent import Policy

    def boom(args, ctx):
        raise RuntimeError("disk on fire")

    registry, names = _probe_registry(boom)
    runtime = ToolRuntime.for_source(source, AS_OF, policy=Policy(allowed_tools=names), registry=registry)
    outcome = runtime.call_tool("probe", {}, tool_use_id="t")
    assert outcome.is_error and outcome.error_code == Code.INTERNAL_ERROR
    assert "RuntimeError: disk on fire" in outcome.content

    registry, names = _probe_registry(lambda args, ctx: "not a ToolResult")
    runtime = ToolRuntime.for_source(source, AS_OF, policy=Policy(allowed_tools=names), registry=registry)
    assert runtime.call_tool("probe", {}).error_code == Code.INTERNAL_ERROR


def test_source_contract_failures_surface_as_data_contract_errors(panel):
    class Leaky:
        def read_bars(self, symbols, start, end):
            return panel.loc[panel["symbol"].isin(symbols)]

        def list_symbols(self):
            return ["SYN01"]

    runtime = ToolRuntime.for_source(Leaky(), AS_OF)
    outcome = runtime.call_tool("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"})
    assert outcome.error_code == Code.DATA_CONTRACT


def test_manifest_describes_the_information_set(runtime):
    manifest = runtime.manifest()
    assert manifest["as_of"] == "2023-06-30"
    assert manifest["tools"] == list(default_registry().names())
    assert len(manifest["tools_sha256"]) == 64 and len(manifest["policy_sha256"]) == 64
    assert manifest["source"] == "FrameBarSource" and manifest["source_snapshot"].startswith("frame:")


def test_tool_result_block_requires_tool_use_id(runtime):
    outcome = runtime.call_tool("list_symbols", {})
    with pytest.raises(ValueError):
        outcome.to_tool_result_block()
    assert isinstance(pd.Timestamp(outcome.result.provenance.last_date), pd.Timestamp)
