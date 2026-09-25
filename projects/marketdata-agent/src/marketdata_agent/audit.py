"""Append-only JSONL audit log with a SHA-256 hash chain.

Each line is the canonical JSON of one record::

    {"episode_id", "kind", "payload", "prev_hash", "record_hash", "seq", "ts"}

``record_hash`` is the SHA-256 of the canonical JSON of the record without
``record_hash``; ``prev_hash`` is the previous record's ``record_hash`` (64
zeros for the first record).  :func:`verify_chain` therefore detects any
edited, deleted, inserted, or reordered record.  Two limits are inherent to a
self-contained chain and are stated plainly: removing records from the *end*
of the file, or rewriting the whole file with a recomputed chain, can only be
detected against an externally stored head (``AuditLog.head``), which callers
should copy into their run manifest.

Secrets are never written: keys that name credentials are replaced with
``[REDACTED]``, and string values are scrubbed of API-key patterns and of the
current values of credential environment variables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from .provenance import canonical_json, json_safe, sha256_text


GENESIS_HASH = "0" * 64
RECORD_KEYS = frozenset({"seq", "ts", "kind", "episode_id", "payload", "prev_hash", "record_hash"})
REDACTED = "[REDACTED]"
SENSITIVE_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MARKETDATA_TOKEN", "FRED_API_KEY")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_\-])(?:api[_\-]?key|apikey|secret|password|passwd|authorization|credentials?|"
    r"(?:auth|access|refresh|bearer|session)?[_\-]?token)$",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
)
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "service_tier",
)


class AuditIntegrityError(RuntimeError):
    """An existing audit file failed hash-chain verification."""


@dataclass(frozen=True)
class ChainHead:
    """External anchor for a log: record count and last record hash."""

    records: int
    head_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records, "head_hash": self.head_hash}


@dataclass(frozen=True)
class ChainVerification:
    """Result of :func:`verify_chain`; ``line`` is 1-based when an error is found."""

    ok: bool
    records: int
    head_hash: str
    error: str | None = None
    line: int | None = None

    @property
    def head(self) -> ChainHead:
        return ChainHead(self.records, self.head_hash)


def compute_record_hash(record: Mapping[str, Any]) -> str:
    """Hash a record's canonical JSON, excluding its ``record_hash`` field."""

    body = {key: value for key, value in record.items() if key != "record_hash"}
    return sha256_text(canonical_json(body))


def verify_chain(path: str | Path, *, expected_head: ChainHead | None = None) -> ChainVerification:
    """Verify every record of ``path``; optionally compare with an external head."""

    target = Path(path)
    previous = GENESIS_HASH
    count = 0
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return ChainVerification(False, 0, GENESIS_HASH, f"cannot read audit log: {exc}", None)
    if raw and not raw.endswith(b"\n"):
        lines = raw.split(b"\n")
        return ChainVerification(
            False, 0, GENESIS_HASH, "final record is truncated (no trailing newline)", len(lines)
        )
    for number, line in enumerate(raw.splitlines(), start=1):
        failure = _check_line(line, number, count, previous)
        if failure is not None:
            return ChainVerification(False, count, previous, failure, number)
        record = json.loads(line)
        previous = record["record_hash"]
        count += 1
    if expected_head is not None:
        if count != expected_head.records or previous != expected_head.head_hash:
            return ChainVerification(
                False,
                count,
                previous,
                f"log head ({count} records, {previous[:12]}) does not match the external anchor "
                f"({expected_head.records} records, {expected_head.head_hash[:12]})",
                None,
            )
    return ChainVerification(True, count, previous)


def _check_line(line: bytes, number: int, index: int, previous: str) -> str | None:
    try:
        text = line.decode("utf-8")
        record = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "record is not valid UTF-8 JSON"
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        return "record does not have the audit schema"
    if record["seq"] != index:
        return f"sequence number {record['seq']} where {index} was expected (deleted, inserted or reordered record)"
    if record["prev_hash"] != previous:
        return "prev_hash does not match the preceding record (deleted, inserted or reordered record)"
    if compute_record_hash(record) != record["record_hash"]:
        return "record_hash does not match the record contents (edited record)"
    if canonical_json(record) != text:
        return "record is not in canonical serialization (edited bytes)"
    return None


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Return all records after verifying the chain; raise if it is broken."""

    verification = verify_chain(path)
    if not verification.ok:
        raise AuditIntegrityError(f"{path}: line {verification.line}: {verification.error}")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Convert an SDK usage object, a mapping, or a simple namespace to JSON."""

    if usage is None:
        return None
    if isinstance(usage, Mapping):
        raw: Any = dict(usage)
    elif callable(getattr(usage, "to_dict", None)):
        raw = usage.to_dict()
    elif callable(getattr(usage, "model_dump", None)):
        raw = usage.model_dump()
    elif hasattr(usage, "__dict__"):
        raw = {key: value for key, value in vars(usage).items() if not key.startswith("_")}
    else:
        raw = {name: getattr(usage, name) for name in USAGE_FIELDS if hasattr(usage, name)}
    return json_safe(raw)


class AuditLog:
    """Thread-safe, append-only, hash-chained JSONL log.

    Opening an existing file verifies its chain first and continues it;
    a broken chain raises :class:`AuditIntegrityError` instead of appending.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
        fsync: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._fsync = bool(fsync)
        self._lock = threading.Lock()
        self._secret_values = tuple(
            value for name in SENSITIVE_ENV_VARS if len(value := os.getenv(name, "").strip()) >= 8
        )
        if self.path.exists() and self.path.stat().st_size > 0:
            verification = verify_chain(self.path)
            if not verification.ok:
                raise AuditIntegrityError(
                    f"refusing to append to a broken audit log {self.path}: "
                    f"line {verification.line}: {verification.error}"
                )
            self._seq, self._head = verification.records, verification.head_hash
        else:
            self._seq, self._head = 0, GENESIS_HASH

    @property
    def head(self) -> ChainHead:
        """Current record count and head hash; store it outside the log as an anchor."""

        with self._lock:
            return ChainHead(self._seq, self._head)

    def append(self, kind: str, payload: Mapping[str, Any], *, episode_id: str | None = None) -> dict[str, Any]:
        """Redact, hash-chain, and durably append one record; return it."""

        with self._lock:
            record: dict[str, Any] = {
                "seq": self._seq,
                "ts": self._now().astimezone(timezone.utc).isoformat(),
                "kind": str(kind),
                "episode_id": episode_id,
                "payload": self.redact(dict(payload)),
                "prev_hash": self._head,
            }
            record = json_safe(record)
            record["record_hash"] = compute_record_hash(record)
            line = canonical_json(record) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            self._seq += 1
            self._head = record["record_hash"]
            return record

    def redact(self, value: Any, key: str | None = None) -> Any:
        """Return ``value`` with credential-named keys and secret-looking strings removed."""

        if key is not None and _SENSITIVE_KEY.search(key):
            return REDACTED
        if isinstance(value, Mapping):
            return {str(k): self.redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            text = value
            for secret in self._secret_values:
                text = text.replace(secret, REDACTED)
            for pattern in _SECRET_VALUE_PATTERNS:
                text = pattern.sub(REDACTED, text)
            return text
        return value

    # Typed helpers -----------------------------------------------------------------

    def log_episode_start(self, episode_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Record the episode manifest (as-of date, policy, tool-set hash, model config)."""

        return self.append("episode_start", manifest, episode_id=episode_id)

    def log_model_call(
        self,
        episode_id: str | None,
        *,
        requested_model: str,
        served_model: str | None,
        stop_reason: str | None,
        usage: Any,
        fallback_enabled: bool,
        turn: int | None = None,
        request_id: str | None = None,
        latency_seconds: float | None = None,
        served_models: Iterable[str] | None = None,
        continuations: int = 0,
        stop_details: Any = None,
        fallback_events: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        """Record one model step, including every model that actually served it.

        ``served_models`` lists the model of each ``pause_turn`` segment; the
        step is flagged ``served_model_differs`` when any of them differs from
        the requested model.
        """

        segments = [str(model) for model in (served_models or ()) if model]
        if not segments and served_model is not None:
            segments = [served_model]
        payload = {
            "turn": turn,
            "requested_model": requested_model,
            "served_model": served_model,
            "served_models": segments,
            "served_model_differs": any(model != requested_model for model in segments),
            "fallback_enabled": bool(fallback_enabled),
            "stop_reason": stop_reason,
            "stop_details": json_safe(stop_details),
            "fallback_events": json_safe(list(fallback_events or ())),
            "continuations": int(continuations),
            "usage": usage_to_dict(usage),
            "request_id": request_id,
            "latency_seconds": latency_seconds,
        }
        return self.append("model_call", payload, episode_id=episode_id)

    def log_policy_decision(
        self,
        episode_id: str | None,
        *,
        tool: str,
        tool_use_id: str | None,
        args: Any,
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.append(
            "policy_decision",
            {"tool": tool, "tool_use_id": tool_use_id, "args": args, "decision": dict(decision)},
            episode_id=episode_id,
        )

    def log_tool_call(self, episode_id: str | None, outcome: Mapping[str, Any]) -> dict[str, Any]:
        return self.append("tool_call", outcome, episode_id=episode_id)

    def log_order_proposal(self, episode_id: str | None, proposal: Mapping[str, Any]) -> dict[str, Any]:
        return self.append("order_proposal", proposal, episode_id=episode_id)

    def log_final_answer(
        self,
        episode_id: str | None,
        *,
        text: str,
        cited_result_ids: Iterable[str],
        stop_reason: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "text": text,
            "cited_result_ids": sorted(set(cited_result_ids)),
            "stop_reason": stop_reason,
            **dict(extra or {}),
        }
        return self.append("final_answer", payload, episode_id=episode_id)
