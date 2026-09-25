"""Scripted backend: predetermined turns for tests and harness-validation baselines.

A script is a sequence of :class:`ScriptedTurn` values, or of callables that
build the next turn from a :class:`ScriptState`. The callables see the
conversation so far, including the parsed tool results. Scripted agents read
result ids and values from tool results the same way a model would, so a
scripted oracle exercises the whole harness: gate, tools, provenance, citations,
grounding and scoring. Scripted runs contain no language model and their
scores are not LLM results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .base import STOP_END_TURN, STOP_TOOL_USE, BackendError, BackendTurn, block_field


_RESULT_HEADER = re.compile(r"^\[r:(?P<rid>[0-9a-f]+)\] (?P<tool>[A-Za-z0-9_-]+) \|")
_ERROR_HEADER = re.compile(r"^\[error:(?P<code>[a-z_]+)\]\s*(?P<message>.*)$", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolResult:
    """A ``tool_result`` block parsed back into its result id, tool and payload."""

    tool_use_id: str
    is_error: bool
    content: str
    result_id: str | None = None
    tool: str | None = None
    payload: Mapping[str, Any] | None = None
    error_code: str | None = None


def parse_tool_result(block: Mapping[str, Any]) -> ParsedToolResult:
    """Parse the runtime's ``[r:<id>] tool | ...`` / ``[error:<code>] ...`` result text."""

    content = str(block_field(block, "content") or "")
    tool_use_id = str(block_field(block, "tool_use_id") or "")
    is_error = bool(block_field(block, "is_error"))
    error = _ERROR_HEADER.match(content)
    if is_error or error:
        return ParsedToolResult(tool_use_id, True, content, error_code=error.group("code") if error else None)
    lines = content.split("\n")
    header = _RESULT_HEADER.match(lines[0]) if lines else None
    payload: Mapping[str, Any] | None = None
    if len(lines) > 1:
        try:
            payload = json.loads(lines[1])
        except json.JSONDecodeError:
            payload = None
    return ParsedToolResult(
        tool_use_id,
        False,
        content,
        result_id=header.group("rid") if header else None,
        tool=header.group("tool") if header else None,
        payload=payload,
    )


def tool_results_in(message: Mapping[str, Any]) -> list[ParsedToolResult]:
    """Parse every ``tool_result`` block of one user message."""

    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return []
    return [parse_tool_result(block) for block in content if block_field(block, "type") == "tool_result"]


@dataclass(frozen=True)
class ScriptState:
    """What a scripted policy may inspect when producing its next turn."""

    step: int
    messages: Sequence[Mapping[str, Any]]

    @property
    def question(self) -> str:
        first = self.messages[0]["content"] if self.messages else ""
        return first if isinstance(first, str) else ""

    @property
    def last_results(self) -> list[ParsedToolResult]:
        return tool_results_in(self.messages[-1]) if self.messages else []

    @property
    def all_results(self) -> list[ParsedToolResult]:
        found: list[ParsedToolResult] = []
        for message in self.messages:
            found.extend(tool_results_in(message))
        return found


@dataclass(frozen=True)
class ScriptedTurn:
    """One scripted assistant turn: optional text, tool calls, and a stop reason."""

    text: str | None = None
    tool_calls: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    stop_reason: str | None = None
    usage: Mapping[str, Any] | None = None


TurnSource = ScriptedTurn | Callable[[ScriptState], ScriptedTurn]


class ScriptExhausted(BackendError):
    """The script has no turn for the requested step."""

    def __init__(self, step: int) -> None:
        super().__init__("script_exhausted", f"scripted backend has no turn for step {step}")


class ScriptedBackend:
    """Replays a fixed list of turns, or asks a policy callable for every turn.

    Tool-use ids are deterministic (``toolu_s<step>_<index>``), which keeps
    transcripts and request fingerprints reproducible.
    """

    def __init__(
        self,
        script: Sequence[TurnSource] | Callable[[ScriptState], ScriptedTurn],
        *,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._policy = script if callable(script) else None
        self._turns: tuple[TurnSource, ...] = () if callable(script) else tuple(script)
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"backend": "scripted", "name": self.name, "model": f"scripted:{self.name}", "fallback": False}

    def step(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> BackendTurn:
        index = self.calls
        self.calls += 1
        state = ScriptState(step=index, messages=tuple(messages))
        if self._policy is not None:
            turn = self._policy(state)
        else:
            if index >= len(self._turns):
                raise ScriptExhausted(index)
            source = self._turns[index]
            turn = source(state) if callable(source) else source
        content: list[dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        for position, (tool_name, tool_input) in enumerate(turn.tool_calls):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_s{index:02d}_{position:02d}",
                    "name": tool_name,
                    "input": dict(tool_input),
                }
            )
        stop_reason = turn.stop_reason or (STOP_TOOL_USE if turn.tool_calls else STOP_END_TURN)
        return BackendTurn.from_content(
            content,
            stop_reason=stop_reason,
            served_model=f"scripted:{self.name}",
            requested_model=f"scripted:{self.name}",
            usage=turn.usage,
        )
