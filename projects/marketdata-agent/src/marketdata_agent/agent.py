"""The copilot's agent loop: model step, policy-gated tools, audit, and grounding.

:class:`Copilot` drives any :class:`~marketdata_agent.backends.base.LLMBackend`:

1. log an ``episode_start`` manifest (as-of date, policy and tool-set hashes,
   backend configuration, system-prompt version and SHA-256);
2. per step, call the backend and log the requested and served model, stop
   reason and usage. On ``refusal`` the episode ends without reading content.
   Otherwise the **full** assistant content is appended to the history;
3. run every ``tool_use`` block of the turn through
   :class:`~marketdata_agent.runtime.ToolRuntime` (gate, then handler, then
   audit) and return all ``tool_result`` blocks in **one** user message.
   Policy denials go back as ``is_error`` results the model can recover from;
4. treat ``max_tokens`` as recoverable. Tool calls from a truncated turn get
   ``output_truncated`` error results and are never executed, because their
   input may be partial. The model is asked to continue;
5. stop at the first turn with no tool calls, or after ``max_steps`` model calls;
6. verify the answer's numeric claims against the episode's results
   (:mod:`marketdata_agent.grounding`) and log the ``final_answer`` record. A
   verifier failure is logged as ``grounding_error`` and the episode is marked
   as not grounded; it never aborts the episode.

One tool runtime, and hence one gate budget and one as-of data view, is created
per episode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from string import Template
from typing import Any

from . import __version__
from .audit import AuditLog, ChainHead
from .backends.base import (
    STOP_MAX_TOKENS,
    STOP_PAUSE_TURN,
    STOP_REFUSAL,
    BackendError,
    BackendTurn,
    LLMBackend,
    accumulate_usage,
    to_plain,
)
from .clock import AsOfClock, DateLike
from .errors import Code
from .grounding import GroundingReport, verify_grounding
from .policy import Policy
from .provenance import json_safe, sha256_json, sha256_text
from .runtime import ToolOutcome, ToolRuntime
from .sources import BarSource, PointInTimeBars
from .tools import ToolContext, ToolRegistry, ToolResult, default_registry


PROMPT_VERSION = "copilot_system_v1"
# Versioned prompts. ``citation_hint`` controls the "Cite numbers ..." line in tool results.
PROMPT_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "copilot_system_v1": {"citation_hint": True, "arm": "main"},
    "copilot_system_v1_nocite": {"citation_hint": False, "arm": "A2: no citation requirement"},
    "copilot_system_v1_nocutoff": {"citation_hint": True, "arm": "A6: no instruction against later knowledge"},
    "copilot_closed_book_v1": {"citation_hint": False, "arm": "A5: closed book (no tools)"},
}
ABSTAIN_TOKEN = "INSUFFICIENT_DATA"
REFUSE_TOKEN = "EXECUTION_REFUSED"
DEFAULT_MAX_STEPS = 8
ABLATION_CUTOFF = date(9999, 12, 31)
TRUNCATION_NOTICE = (
    "Your previous reply reached the output token limit and was cut off. Tool calls from that reply "
    "were not run. Continue, and keep the final answer concise."
)
EPISODE_STATUSES = ("completed", "max_steps", "refusal", "backend_error")  # also the order of status tables


@dataclass(frozen=True)
class SystemPrompt:
    """Rendered system prompt with the hashes recorded in every manifest."""

    version: str
    text: str
    template_sha256: str
    sha256: str

    def meta(self) -> dict[str, str]:
        return {"version": self.version, "template_sha256": self.template_sha256, "sha256": self.sha256}


def load_system_prompt(as_of: date, policy: Policy, version: str = PROMPT_VERSION) -> SystemPrompt:
    """Render ``prompts/<version>.md`` for one episode."""

    if version not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt version {version!r}; choose from {sorted(PROMPT_VARIANTS)}")
    template = resources.files("marketdata_agent").joinpath("prompts").joinpath(f"{version}.md").read_text(encoding="utf-8")
    text = Template(template).substitute(as_of=as_of.isoformat(), max_tool_calls=policy.max_tool_calls).strip()
    return SystemPrompt(version, text, sha256_text(template), sha256_text(text))


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool attempt within an episode."""

    step: int
    tool_use_id: str | None
    tool: str
    args: Any
    allowed: bool
    decision_code: str
    violations: tuple[str, ...]
    error_code: str | None
    result_id: str | None
    last_date: str | None
    row_count: int | None
    unenforced: tuple[str, ...] = ()

    @classmethod
    def from_outcome(cls, step: int, outcome: ToolOutcome) -> "ToolCallRecord":
        provenance = outcome.result.provenance if outcome.result is not None else None
        return cls(
            step=step,
            tool_use_id=outcome.tool_use_id,
            tool=outcome.tool,
            args=json_safe(outcome.args),
            allowed=outcome.decision.allowed,
            decision_code=str(outcome.decision.code),
            violations=tuple(str(code) for code in outcome.decision.violations),
            error_code=str(outcome.error_code) if outcome.error_code is not None else None,
            result_id=provenance.result_id if provenance else None,
            last_date=provenance.last_date if provenance else None,
            row_count=provenance.row_count if provenance else None,
            unenforced=tuple(str(code) for code in outcome.decision.unenforced),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool_use_id": self.tool_use_id,
            "tool": self.tool,
            "args": self.args,
            "allowed": self.allowed,
            "decision_code": self.decision_code,
            "violations": list(self.violations),
            "unenforced": list(self.unenforced),
            "error_code": self.error_code,
            "result_id": self.result_id,
            "last_date": self.last_date,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class Episode:
    """Everything that happened in one question-answer episode."""

    episode_id: str
    question: str
    as_of: str
    answer: str
    status: str
    stop_reason: str | None
    steps: int
    tool_calls: tuple[ToolCallRecord, ...]
    results: Mapping[str, ToolResult]
    violations: Mapping[str, int]
    lookahead_attempts: int
    order_execution_attempts: int
    proposals: tuple[Mapping[str, Any], ...]
    usage: Mapping[str, Any]
    served_models: tuple[str, ...]
    requested_model: str | None
    backend: Mapping[str, Any]
    prompt: Mapping[str, str]
    grounding: GroundingReport
    transcript: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    clock_enforced: bool = True
    max_tokens_events: int = 0
    error: Mapping[str, Any] | None = None
    audit_head: ChainHead | None = None
    stop_details: tuple[Mapping[str, Any], ...] = ()
    latency_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.status not in EPISODE_STATUSES:
            raise ValueError(f"unknown episode status {self.status!r}; expected one of {EPISODE_STATUSES}")

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def leaked_results(self) -> tuple[str, ...]:
        """Result ids whose rows extend past the nominal as-of date (only possible in the ablation)."""

        return tuple(
            rid
            for rid, result in sorted(self.results.items())
            if result.provenance.last_date is not None and result.provenance.last_date > self.as_of
        )

    def to_dict(self, *, include_transcript: bool = False) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "question": self.question,
            "as_of": self.as_of,
            "answer": self.answer,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "result_ids": sorted(self.results),
            "violations": dict(sorted(self.violations.items())),
            "lookahead_attempts": self.lookahead_attempts,
            "order_execution_attempts": self.order_execution_attempts,
            "proposals": [dict(p) for p in self.proposals],
            "usage": to_plain(dict(self.usage)),
            "served_models": list(self.served_models),
            "requested_model": self.requested_model,
            "backend": to_plain(dict(self.backend)),
            "prompt": dict(self.prompt),
            "grounding": self.grounding.to_dict(),
            "metrics": json_safe(dict(self.metrics)),
            "clock_enforced": self.clock_enforced,
            "leaked_results": list(self.leaked_results),
            "max_tokens_events": self.max_tokens_events,
            "error": to_plain(self.error),
            "stop_details": to_plain(list(self.stop_details)),
            "latency_seconds": self.latency_seconds,
            "audit_head": self.audit_head.to_dict() if self.audit_head else None,
        }
        if include_transcript:
            payload["transcript"] = to_plain(list(self.transcript))
        return payload


def default_episode_id(question: str, as_of: date, backend: Mapping[str, Any]) -> str:
    """Deterministic id from the question, as-of date, and backend configuration."""

    return "ep-" + sha256_json({"question": question, "as_of": as_of.isoformat(), "backend": to_plain(dict(backend))})[:12]


@dataclass
class Copilot:
    """Governed question-answering over point-in-time market data.

    ``enforce_clock=False`` is an **ablation only** (A1; the benchmark's
    ``no_guard`` agent and ``bench run --no-clock``). It lifts exactly one
    control: the refusal of explicitly named dates (and symbols) after the
    as-of date *t*, whose refusal cutoff becomes 9999-12-31. Everything
    else still refers to *t*: ``"latest"``, the universe shown by
    ``list_symbols``, proposal reference prices, result headers and the
    system prompt. A policy that never names a later date therefore behaves
    identically with and without the clock. Calls that name a later date are
    served but still counted as look-ahead attempts (``unenforced``), and the
    manifest records ``clock_enforced: false`` and ``data_cutoff``. Policy
    limits, the refusal of order execution, and the audit log are unchanged.
    """

    backend: LLMBackend
    source: BarSource
    as_of: DateLike | AsOfClock
    registry: ToolRegistry | None = None
    policy: Policy = field(default_factory=Policy)
    audit: AuditLog | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    prompt_version: str = PROMPT_VERSION
    enforce_clock: bool = True

    def __post_init__(self) -> None:
        self.clock = self.as_of if isinstance(self.as_of, AsOfClock) else AsOfClock(self.as_of)
        if self.registry is None:
            self.registry = default_registry(self.policy)
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.prompt_version not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt version {self.prompt_version!r}; choose from {sorted(PROMPT_VARIANTS)}")

    def _runtime(self, episode_id: str) -> ToolRuntime:
        cutoff = None if self.enforce_clock else AsOfClock(ABLATION_CUTOFF)
        context = ToolContext(
            PointInTimeBars(self.source, self.clock, cutoff=cutoff),
            self.policy,
            citation_hint=bool(PROMPT_VARIANTS[self.prompt_version]["citation_hint"]),
        )
        assert self.registry is not None
        return ToolRuntime(self.registry, context, audit=self.audit, episode_id=episode_id)

    def run(self, question: str, *, episode_id: str | None = None) -> Episode:
        """Answer one question and return the complete episode record."""

        if not str(question).strip():
            raise ValueError("question must be non-empty")
        assert self.registry is not None
        backend_config = self.backend.describe()
        episode_id = episode_id or default_episode_id(question, self.clock.as_of, backend_config)
        runtime = self._runtime(episode_id)
        prompt = load_system_prompt(self.clock.as_of, self.policy, self.prompt_version)
        tools = self.registry.to_anthropic_tools()
        requested_model = backend_config.get("model")
        fallback_enabled = bool(backend_config.get("fallback", False))
        if self.audit is not None:
            manifest = {
                **runtime.manifest(),
                "as_of": self.clock.as_of.isoformat(),
                "episode_id": episode_id,
                "question": question,
                "question_sha256": sha256_text(question),
                "clock_enforced": self.enforce_clock,
                "data_cutoff": runtime.context.cutoff.as_of.isoformat(),
                "max_steps": self.max_steps,
                "backend": to_plain(backend_config),
                "prompt": prompt.meta(),
                "package_version": __version__,
            }
            self.audit.log_episode_start(episode_id, manifest)

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        records: list[ToolCallRecord] = []
        served_models: list[str] = []
        usage_total: dict[str, Any] = {}
        stop_details: list[Mapping[str, Any]] = []
        latencies: list[float] = []
        status, answer, stop_reason, error = "max_steps", "", None, None
        steps = max_tokens_events = 0

        for step in range(1, self.max_steps + 1):
            try:
                turn = self.backend.step(prompt.text, messages, tools)
            except BackendError as exc:
                status, error = "backend_error", exc.to_dict()
                if self.audit is not None:
                    self.audit.append("model_error", {"turn": step, **exc.to_dict()}, episode_id=episode_id)
                break
            steps = step
            stop_reason = turn.stop_reason
            self._record_turn(episode_id, step, turn, requested_model, fallback_enabled, served_models, usage_total)
            if turn.stop_details:
                details = to_plain(turn.stop_details)
                stop_details.append({"turn": step, **details} if isinstance(details, dict) else {"turn": step, "detail": details})
            if turn.latency_seconds is not None:
                latencies.append(float(turn.latency_seconds))
            if turn.stop_reason == STOP_REFUSAL:
                status = "refusal"
                break
            messages.append(turn.assistant_message())
            if turn.stop_reason == STOP_MAX_TOKENS:
                max_tokens_events += 1
                messages.append({"role": "user", "content": self._truncated_followup(episode_id, step, turn)})
                continue
            if turn.tool_calls:
                before = len(runtime.outcomes)
                results = runtime.handle_tool_uses([call.to_block() for call in turn.tool_calls])
                records.extend(ToolCallRecord.from_outcome(step, o) for o in runtime.outcomes[before:])
                messages.append({"role": "user", "content": results})
                continue
            if turn.stop_reason == STOP_PAUSE_TURN:
                continue  # continuation budget exhausted inside the backend; re-send
            status, answer = "completed", turn.text
            break

        results_by_id = runtime.results()
        try:
            grounding = verify_grounding(answer, results_by_id, question=question)
        except Exception as exc:  # noqa: BLE001 - a verifier defect must not abort the episode or the run
            grounding = GroundingReport.failure(f"{type(exc).__name__}: {exc}")
            if self.audit is not None:
                self.audit.append("grounding_error", {"error": grounding.error}, episode_id=episode_id)
        metrics = runtime.metrics()
        violations: Counter[str] = Counter()
        for outcome in runtime.outcomes:
            if not outcome.decision.allowed:
                violations.update(str(code) for code in outcome.decision.violations)
            elif outcome.error_code is not None:
                violations[str(outcome.error_code)] += 1
        proposals = tuple(p.to_dict() for p in runtime.context.proposals.proposals)
        head = None
        if self.audit is not None:
            self.audit.log_final_answer(
                episode_id,
                text=answer,
                cited_result_ids=grounding.cited_ids,
                stop_reason=stop_reason,
                extra={
                    "status": status,
                    "steps": steps,
                    "grounding": grounding.summary(),
                    "metrics": metrics,
                    "served_models": sorted(set(served_models)),
                    "usage": usage_total,
                    "error": error,
                },
            )
            head = self.audit.head
        return Episode(
            episode_id=episode_id,
            question=question,
            as_of=self.clock.as_of.isoformat(),
            answer=answer,
            status=status,
            stop_reason=stop_reason,
            steps=steps,
            tool_calls=tuple(records),
            results=results_by_id,
            violations=dict(violations),
            lookahead_attempts=int(metrics["lookahead_attempts"]),
            order_execution_attempts=int(metrics["order_execution_attempts"]),
            proposals=proposals,
            usage=usage_total,
            served_models=tuple(dict.fromkeys(served_models)),
            requested_model=requested_model,
            backend=backend_config,
            prompt=prompt.meta(),
            grounding=grounding,
            transcript=tuple(to_plain(messages)),
            metrics=metrics,
            clock_enforced=self.enforce_clock,
            max_tokens_events=max_tokens_events,
            error=error,
            audit_head=head,
            stop_details=tuple(stop_details),
            latency_seconds=sum(latencies) if latencies else None,
        )

    def _record_turn(
        self,
        episode_id: str,
        step: int,
        turn: BackendTurn,
        requested_model: Any,
        fallback_enabled: bool,
        served_models: list[str],
        usage_total: dict[str, Any],
    ) -> None:
        served_models.extend(turn.served_models)
        if turn.usage:
            accumulate_usage(usage_total, turn.usage)
        if self.audit is not None:
            self.audit.log_model_call(
                episode_id,
                requested_model=str(turn.requested_model or requested_model or "unknown"),
                served_model=turn.served_model,
                served_models=turn.served_models,
                stop_reason=turn.stop_reason,
                stop_details=to_plain(turn.stop_details),
                fallback_events=to_plain(list(turn.fallback_events)),
                continuations=turn.continuations,
                usage=turn.usage,
                fallback_enabled=fallback_enabled,
                turn=step,
                request_id=",".join(turn.request_ids) or None,
                latency_seconds=turn.latency_seconds,
            )

    def _truncated_followup(self, episode_id: str, step: int, turn: BackendTurn) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for call in turn.tool_calls:
            message = f"[error:{Code.OUTPUT_TRUNCATED}] the reply was cut off at max_tokens; this call was not run."
            blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": message, "is_error": True})
            if self.audit is not None:
                self.audit.append(
                    "tool_call_skipped",
                    {"turn": step, "tool_use_id": call.id, "tool": call.name, "reason": str(Code.OUTPUT_TRUNCATED)},
                    episode_id=episode_id,
                )
        blocks.append({"type": "text", "text": TRUNCATION_NOTICE})
        return blocks
