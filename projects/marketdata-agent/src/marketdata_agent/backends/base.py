"""Backend-neutral types for one model step of the copilot's agent loop.

A backend turns ``(system, messages, tools)`` into one :class:`BackendTurn`:

* ``blocks``: the normalized text and ``tool_use`` parts the loop acts on;
* ``raw_content``: the assistant content exactly as returned (SDK block objects
  or plain dicts, including thinking blocks). The loop appends it to the history
  unchanged, so thinking blocks are passed back as the API requires;
* ``stop_reason``, ``served_model`` and ``usage``, which are recorded for every call.

Every backend used in this package satisfies :class:`LLMBackend`: the Claude
backend, the scripted baselines, and the record/replay wrappers. Evaluation is
therefore independent of the backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Literal, Protocol, runtime_checkable

from ..provenance import sha256_json


STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_PAUSE_TURN = "pause_turn"
STOP_REFUSAL = "refusal"


def to_plain(value: Any) -> Any:
    """Convert SDK models, dataclasses, namespaces and containers to plain JSON values.

    ``None`` fields are dropped, so an SDK block and the dict recorded from it
    serialize identically. Request fingerprints and replay files depend on this.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return to_plain(dump(mode="json", exclude_none=True))
    if is_dataclass(value) and not isinstance(value, type):
        return to_plain(asdict(value))
    if hasattr(value, "__dict__"):
        return to_plain({key: item for key, item in vars(value).items() if not key.startswith("_")})
    return str(value)


def block_field(block: Any, name: str) -> Any:
    """Read a field from an SDK block object or a dict block."""

    if isinstance(block, Mapping):
        return block.get(name)
    return getattr(block, name, None)


@dataclass(frozen=True)
class TextPart:
    """Normalized text block."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolCall:
    """Normalized ``tool_use`` block."""

    id: str
    name: str
    input: Any
    type: Literal["tool_use"] = "tool_use"

    def to_block(self) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


Part = TextPart | ToolCall


def normalize_blocks(content: Iterable[Any]) -> tuple[Part, ...]:
    """Extract text and tool_use parts; thinking, fallback and other blocks are skipped."""

    parts: list[Part] = []
    for block in content:
        kind = block_field(block, "type")
        if kind == "text":
            parts.append(TextPart(str(block_field(block, "text") or "")))
        elif kind == "tool_use":
            parts.append(
                ToolCall(
                    id=str(block_field(block, "id")),
                    name=str(block_field(block, "name")),
                    input=to_plain(block_field(block, "input")),
                )
            )
    return tuple(parts)


@dataclass(frozen=True)
class BackendTurn:
    """One assistant turn, normalized for the loop and kept raw for the history."""

    blocks: tuple[Part, ...]
    raw_content: tuple[Any, ...]
    stop_reason: str | None
    served_model: str | None
    requested_model: str | None = None
    usage: Mapping[str, Any] | None = None
    request_ids: tuple[str, ...] = ()
    latency_seconds: float | None = None
    continuations: int = 0
    stop_details: Mapping[str, Any] | None = None
    fallback_events: tuple[Mapping[str, Any], ...] = ()
    segment_models: tuple[str, ...] = ()

    @classmethod
    def from_content(cls, content: Sequence[Any], **kwargs: Any) -> "BackendTurn":
        raw = tuple(content)
        return cls(blocks=normalize_blocks(raw), raw_content=raw, **kwargs)

    @property
    def text(self) -> str:
        return "\n".join(part.text for part in self.blocks if isinstance(part, TextPart)).strip()

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(part for part in self.blocks if isinstance(part, ToolCall))

    @property
    def served_models(self) -> tuple[str, ...]:
        """Every model that served a segment of this turn (``pause_turn`` continuations included)."""

        models = list(self.segment_models) or ([self.served_model] if self.served_model else [])
        return tuple(dict.fromkeys(model for model in models if model))

    def assistant_message(self) -> dict[str, Any]:
        """The full assistant content, unchanged, as one history message."""

        return {"role": "assistant", "content": list(self.raw_content)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_content": to_plain(list(self.raw_content)),
            "stop_reason": self.stop_reason,
            "served_model": self.served_model,
            "requested_model": self.requested_model,
            "usage": to_plain(self.usage),
            "request_ids": list(self.request_ids),
            "latency_seconds": self.latency_seconds,
            "continuations": self.continuations,
            "stop_details": to_plain(self.stop_details),
            "fallback_events": to_plain(list(self.fallback_events)),
            "segment_models": list(self.segment_models),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendTurn":
        return cls.from_content(
            list(payload.get("raw_content") or []),
            stop_reason=payload.get("stop_reason"),
            served_model=payload.get("served_model"),
            requested_model=payload.get("requested_model"),
            usage=payload.get("usage"),
            request_ids=tuple(payload.get("request_ids") or ()),
            latency_seconds=payload.get("latency_seconds"),
            continuations=int(payload.get("continuations") or 0),
            stop_details=payload.get("stop_details"),
            fallback_events=tuple(payload.get("fallback_events") or ()),
            segment_models=tuple(payload.get("segment_models") or ()),
        )


class BackendError(RuntimeError):
    """A model call failed. The episode ends with status ``backend_error``."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "retryable": self.retryable,
            "status_code": self.status_code,
            "request_id": self.request_id,
        }


@runtime_checkable
class LLMBackend(Protocol):
    """One model step: given the conversation so far, return the next assistant turn."""

    name: str

    def step(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> BackendTurn:
        """Return the next assistant turn; raise :class:`BackendError` on failure."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the JSON-safe configuration recorded in the episode manifest."""
        ...


def request_fingerprint(
    system: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    """SHA-256 of the canonical request: backend config, system prompt, history and tools."""

    return sha256_json(
        {
            "config": to_plain(dict(config)),
            "system": system,
            "messages": to_plain(list(messages)),
            "tools": to_plain(list(tools)),
        }
    )


@dataclass
class UsageTotals:
    """Sum of integer usage fields across calls (nested dicts are summed recursively)."""

    totals: dict[str, Any] = field(default_factory=dict)

    def add(self, usage: Mapping[str, Any] | None) -> None:
        if usage:
            accumulate_usage(self.totals, usage)


def accumulate_usage(total: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Add ``usage`` into ``total`` in place (integers summed, other values replaced)."""

    for key, value in usage.items():
        if isinstance(value, bool) or value is None:
            total[key] = value
        elif isinstance(value, int):
            current = total.get(key)
            total[key] = (current if isinstance(current, int) and not isinstance(current, bool) else 0) + value
        elif isinstance(value, Mapping):
            nested = total.get(key)
            if not isinstance(nested, dict):
                nested = {}
                total[key] = nested
            accumulate_usage(nested, value)
        else:
            total[key] = value
