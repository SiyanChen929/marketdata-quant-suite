"""LLM backends behind one protocol: Claude, scripted baselines, and record/replay."""

from __future__ import annotations

from .anthropic_backend import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    FALLBACK_BETA,
    AnthropicBackend,
    AnthropicConfig,
)
from .base import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_PAUSE_TURN,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    BackendError,
    BackendTurn,
    LLMBackend,
    TextPart,
    ToolCall,
    normalize_blocks,
    request_fingerprint,
    to_plain,
)
from .replay import RecordingBackend, ReplayBackend, ReplayMissError
from .scripted import (
    ParsedToolResult,
    ScriptExhausted,
    ScriptState,
    ScriptedBackend,
    ScriptedTurn,
    parse_tool_result,
    tool_results_in,
)

__all__ = [
    "AnthropicBackend",
    "AnthropicConfig",
    "BackendError",
    "BackendTurn",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "EFFORT_LEVELS",
    "FALLBACK_BETA",
    "LLMBackend",
    "ParsedToolResult",
    "RecordingBackend",
    "ReplayBackend",
    "ReplayMissError",
    "STOP_END_TURN",
    "STOP_MAX_TOKENS",
    "STOP_PAUSE_TURN",
    "STOP_REFUSAL",
    "STOP_TOOL_USE",
    "ScriptExhausted",
    "ScriptState",
    "ScriptedBackend",
    "ScriptedTurn",
    "TextPart",
    "ToolCall",
    "normalize_blocks",
    "parse_tool_result",
    "request_fingerprint",
    "to_plain",
    "tool_results_in",
]
