from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from marketdata_agent import AuditIntegrityError, AuditLog, ChainHead, read_records, usage_to_dict, verify_chain
from marketdata_agent.audit import GENESIS_HASH, REDACTED, compute_record_hash


def _fixed_clock():
    start = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    counter = iter(range(10_000))
    return lambda: start + timedelta(seconds=next(counter))


def _filled_log(path, n: int = 5) -> AuditLog:
    log = AuditLog(path, now=_fixed_clock(), fsync=False)
    for index in range(n):
        log.append("tool_call", {"index": index, "tool": "period_return"}, episode_id="ep")
    return log


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def test_chain_verifies_and_links_records(tmp_path):
    path = tmp_path / "a.jsonl"
    log = _filled_log(path)
    result = verify_chain(path)
    assert result.ok and result.records == 5 and result.head == log.head
    records = read_records(path)
    assert records[0]["prev_hash"] == GENESIS_HASH
    for previous, current in zip(records, records[1:]):
        assert current["prev_hash"] == previous["record_hash"]
    assert all(compute_record_hash(r) == r["record_hash"] for r in records)
    assert [r["seq"] for r in records] == list(range(5))


def test_edit_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    _filled_log(path)
    lines = _lines(path)
    lines[2] = lines[2].replace('"index":2', '"index":7')
    path.write_text("".join(lines), encoding="utf-8")
    result = verify_chain(path)
    assert not result.ok and result.line == 3 and "edited record" in result.error


def test_edit_with_recomputed_record_hash_breaks_the_next_link(tmp_path):
    path = tmp_path / "a.jsonl"
    _filled_log(path)
    lines = _lines(path)
    record = json.loads(lines[1])
    record["payload"]["tool"] = "max_drawdown"
    record["record_hash"] = compute_record_hash(record)
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    result = verify_chain(path)
    assert not result.ok and result.line == 3 and "prev_hash" in result.error


@pytest.mark.parametrize("mutation", ["delete_middle", "reorder", "duplicate", "blank_line"])
def test_deletion_reorder_and_insertion_are_detected(tmp_path, mutation):
    path = tmp_path / "a.jsonl"
    _filled_log(path)
    lines = _lines(path)
    if mutation == "delete_middle":
        del lines[2]
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "duplicate":
        lines.insert(2, lines[1])
    else:
        lines.insert(2, "\n")
    path.write_text("".join(lines), encoding="utf-8")
    assert not verify_chain(path).ok


def test_tail_truncation_needs_an_external_anchor(tmp_path):
    path = tmp_path / "a.jsonl"
    log = _filled_log(path)
    anchor = log.head
    path.write_text("".join(_lines(path)[:-1]), encoding="utf-8")
    assert verify_chain(path).ok  # a self-contained chain cannot see a clean tail cut ...
    anchored = verify_chain(path, expected_head=anchor)  # ... an external head can
    assert not anchored.ok and "external anchor" in anchored.error
    assert verify_chain(path, expected_head=ChainHead(4, verify_chain(path).head_hash)).ok


def test_partial_final_write_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    _filled_log(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-10], encoding="utf-8")
    result = verify_chain(path)
    assert not result.ok and "truncated" in result.error


def test_reopening_continues_the_chain_and_refuses_a_broken_log(tmp_path):
    path = tmp_path / "a.jsonl"
    _filled_log(path, n=3)
    reopened = AuditLog(path, fsync=False)
    assert reopened.head.records == 3
    reopened.append("final_answer", {"text": "done"})
    assert verify_chain(path).records == 4
    lines = _lines(path)
    lines[0] = lines[0].replace("period_return", "period_returm")
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(AuditIntegrityError):
        AuditLog(path)
    with pytest.raises(AuditIntegrityError):
        read_records(path)


def test_identical_inputs_produce_identical_logs(tmp_path):
    _filled_log(tmp_path / "x.jsonl")
    _filled_log(tmp_path / "y.jsonl")
    assert (tmp_path / "x.jsonl").read_bytes() == (tmp_path / "y.jsonl").read_bytes()


def test_secrets_are_never_written(tmp_path, monkeypatch):
    secret = "live-secret-value-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    log = AuditLog(tmp_path / "s.jsonl", fsync=False)
    record = log.append(
        "model_call",
        {
            "api_key": "abc",
            "headers": {"x-api-key": "sk-ant-api03-abcdefghijklmnop", "Authorization": "Bearer abcdefghijklmnop"},
            "note": f"echo {secret} and sk-ant-api03-ZZZZZZZZZZZZZZZZ",
            "list": ["Bearer qwertyuiopasdf"],
            "usage": {"input_tokens": 1200, "output_tokens": 345, "cache_read_input_tokens": 0},
            "max_tokens": 16000,
        },
    )
    text = (tmp_path / "s.jsonl").read_text(encoding="utf-8")
    assert secret not in text and "sk-ant-" not in text and "abcdefghijklmnop" not in text and "qwertyuiop" not in text
    payload = record["payload"]
    assert payload["api_key"] == REDACTED
    assert payload["headers"] == {"x-api-key": REDACTED, "Authorization": REDACTED}
    assert payload["usage"] == {"input_tokens": 1200, "output_tokens": 345, "cache_read_input_tokens": 0}
    assert payload["max_tokens"] == 16000
    assert verify_chain(tmp_path / "s.jsonl").ok


class _PydanticLikeUsage:
    def model_dump(self):
        return {"input_tokens": 10, "output_tokens": 20}


def test_model_call_records_served_model_usage_and_stop_reason(tmp_path):
    log = AuditLog(tmp_path / "m.jsonl", fsync=False)
    usage = SimpleNamespace(input_tokens=900, output_tokens=120, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    same = log.log_model_call(
        "ep",
        requested_model="claude-opus-5",
        served_model="claude-opus-5",
        stop_reason="tool_use",
        usage=usage,
        fallback_enabled=False,
        turn=1,
    )["payload"]
    assert same["served_model"] == "claude-opus-5" and same["served_model_differs"] is False
    assert same["stop_reason"] == "tool_use"
    assert same["usage"] == {"input_tokens": 900, "output_tokens": 120, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    swapped = log.log_model_call(
        "ep",
        requested_model="claude-opus-5",
        served_model="some-fallback-model",
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 6},
        fallback_enabled=True,
    )["payload"]
    assert swapped["served_model_differs"] is True and swapped["fallback_enabled"] is True
    assert usage_to_dict(_PydanticLikeUsage()) == {"input_tokens": 10, "output_tokens": 20}
    assert usage_to_dict(None) is None
    final = log.log_final_answer("ep", text="Answer [r:b] [r:a]", cited_result_ids=["b", "a", "b"], stop_reason="end_turn")
    assert final["payload"]["cited_result_ids"] == ["a", "b"]
    assert [r["kind"] for r in read_records(log.path)] == ["model_call", "model_call", "final_answer"]
