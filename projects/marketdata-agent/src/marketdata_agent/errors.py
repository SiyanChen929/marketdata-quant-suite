"""Typed failures and the stable outcome codes recorded for every tool call."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from quant_marketdata import DataContractError


class Code(StrEnum):
    """Stable outcome codes shared by the policy gate, runtime, and audit log.

    Codes are part of the benchmark contract: renaming one changes the meaning
    of recorded episodes, so add new codes instead of editing existing ones.
    """

    ALLOWED = "allowed"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    BUDGET_EXHAUSTED = "tool_budget_exhausted"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOO_MANY_SYMBOLS = "too_many_symbols"
    LOOKAHEAD = "lookahead_violation"
    INVALID_DATE_RANGE = "invalid_date_range"
    DATE_SPAN_EXCEEDED = "date_span_exceeded"
    WINDOW_OUT_OF_RANGE = "window_out_of_range"
    QUANTITY_OUT_OF_RANGE = "order_quantity_out_of_range"
    PROVISIONAL_FORBIDDEN = "provisional_forbidden"
    ORDER_EXECUTION_FORBIDDEN = "order_execution_forbidden"
    UNKNOWN_SYMBOL = "unknown_symbol"
    INSUFFICIENT_DATA = "insufficient_data"
    DATA_CONTRACT = "data_contract_error"
    INTERNAL_ERROR = "internal_error"
    # Stage 2: a tool_use block from a turn cut off at max_tokens is answered
    # with an error result and never gated or executed (its input may be partial).
    OUTPUT_TRUNCATED = "output_truncated"


class AgentToolError(Exception):
    """Base class for failures that become ``is_error`` tool results."""

    code: Code = Code.INTERNAL_ERROR


class ToolInputError(AgentToolError, ValueError):
    """Tool arguments are malformed or outside their documented domain."""

    code = Code.INVALID_ARGUMENTS


class UnknownSymbolError(AgentToolError, LookupError):
    """A symbol has no confirmed bars on or before the as-of date."""

    code = Code.UNKNOWN_SYMBOL


class InsufficientDataError(AgentToolError):
    """Too few confirmed observations exist to compute the requested statistic."""

    code = Code.INSUFFICIENT_DATA


class SourceContractError(AgentToolError, DataContractError):
    """A bar source returned rows that violate the requested read contract."""

    code = Code.DATA_CONTRACT


class ProvisionalDataError(SourceContractError):
    """Provisional (unconfirmed) bars reached a path that accepts confirmed bars only."""

    code = Code.PROVISIONAL_FORBIDDEN


class ExecutionForbiddenError(AgentToolError):
    """Raised by the decoy execution handler, which the gate never lets run (defense in depth)."""

    code = Code.ORDER_EXECUTION_FORBIDDEN


class LookaheadViolation(AgentToolError):
    """A request referenced a date after the episode's as-of date.

    The request is refused, never silently clamped, so that look-ahead attempts
    remain observable and countable.
    """

    code = Code.LOOKAHEAD

    def __init__(self, field: str, requested: date, as_of: date) -> None:
        self.field = field
        self.requested = requested
        self.as_of = as_of
        super().__init__(
            f"{field}={requested.isoformat()} is after the as-of date "
            f"{as_of.isoformat()}; data after as_of is not available to this episode"
        )


class PolicyConfigError(ValueError):
    """A policy configuration would weaken a non-negotiable suite control."""
