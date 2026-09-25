"""Read-only market-data tools over the as-of view.

Every handler resolves its dates through the episode clock, reads through
:class:`~marketdata_agent.sources.PointInTimeBars`, computes with
:mod:`marketdata_agent.analytics`, and returns a provenance-carrying
:class:`~marketdata_agent.tools.base.ToolResult` built from exactly the rows
it used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .. import analytics
from ..errors import InsufficientDataError, ToolInputError
from ..policy import MIN_WINDOW, Policy
from ..provenance import significant
from .base import ToolContext, ToolResult, ToolSpec


SYMBOL_PROP = {"type": "string", "description": "Ticker symbol exactly as returned by list_symbols."}
START_PROP = {
    "type": "string",
    "description": "Inclusive start session date as YYYY-MM-DD; must be on or before the as-of date.",
}
END_PROP = {
    "type": "string",
    "description": (
        "Inclusive end session date as YYYY-MM-DD, or the literal 'latest' for the last confirmed "
        "session on or before the as-of date. Dates after the as-of date are refused."
    ),
}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _iso(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _num(value: float) -> float | None:
    return significant(value)


def _closes(rows: pd.DataFrame, symbol: str) -> pd.Series:
    selected = rows.loc[rows["symbol"].eq(symbol)].sort_values("date")
    return pd.Series(selected["close"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(selected["date"]))


def _range_rows(ctx: ToolContext, symbol: str, args: Mapping[str, Any], minimum: int) -> tuple[pd.DataFrame, dict[str, str]]:
    start = ctx.resolve_start(args["start"])
    end = ctx.resolve_end(args["end"])
    if start > end:
        raise ToolInputError(f"start {start.isoformat()} is after end {end.isoformat()}")
    rows = ctx.bars([symbol], start, end)
    if len(rows) < minimum:
        raise InsufficientDataError(
            f"{symbol} has {len(rows)} confirmed session(s) in {start.isoformat()}..{end.isoformat()}; "
            f"need at least {minimum}"
        )
    return rows, {"start": start.isoformat(), "end": end.isoformat()}


def list_symbols(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Universe known at ``as_of`` with per-symbol coverage."""

    panel = ctx.data.known_panel()
    if panel.empty:
        raise InsufficientDataError(f"no confirmed bars exist on or before {ctx.as_of.isoformat()}")
    coverage = (
        panel.groupby("symbol", sort=True)["date"].agg(["min", "max", "count"]).reset_index()
    )
    symbols = [
        {
            "symbol": str(row.symbol),
            "first_date": _iso(row.min),
            "last_date": _iso(row.max),
            "sessions": int(row.count),
        }
        for row in coverage.itertuples(index=False)
    ]
    payload = {
        "as_of": ctx.as_of.isoformat(),
        "latest_session": _iso(panel["date"].max()),
        "count": len(symbols),
        "symbols": symbols,
    }
    return ctx.build_result("list_symbols", {}, panel, payload, {"count": float(len(symbols))})


def get_daily_bars(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Summary statistics plus at most ``max_rows_per_result`` most-recent rows."""

    symbol = ctx.require_symbols([args["symbol"]])[0]
    rows, dates = _range_rows(ctx, symbol, args, minimum=1)
    rows = rows.sort_values("date").reset_index(drop=True)
    limit = ctx.policy.max_rows_per_result
    shown = rows.tail(limit)
    summary = {
        "first_close": _num(rows["close"].iloc[0]),
        "last_close": _num(rows["close"].iloc[-1]),
        "period_return": _num(analytics.simple_return(rows["close"])) if len(rows) >= 2 else None,
        "min_low": _num(rows["low"].min()),
        "max_high": _num(rows["high"].max()),
        "mean_volume": _num(rows["volume"].mean()),
    }
    table = [
        {
            "date": _iso(row.date),
            "open": _num(row.open),
            "high": _num(row.high),
            "low": _num(row.low),
            "close": _num(row.close),
            "volume": _num(row.volume),
        }
        for row in shown.itertuples(index=False)
    ]
    payload = {
        "symbol": symbol,
        "first_date": _iso(rows["date"].iloc[0]),
        "last_date": _iso(rows["date"].iloc[-1]),
        "rows_total": int(len(rows)),
        "rows_returned": len(table),
        "truncated": bool(len(rows) > limit),
        "rows_order": "most recent rows, ascending by date",
        "summary": summary,
        "rows": table,
    }
    outputs: dict[str, float | None] = dict(summary)
    for entry in table:
        for key in ("open", "high", "low", "close", "volume"):
            outputs[f"{key}[{entry['date']}]"] = entry[key]
    return ctx.build_result("get_daily_bars", {"symbol": symbol, **dates}, rows, payload, outputs)


def period_return(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Close-to-close return over ``[start, end]``."""

    symbol = ctx.require_symbols([args["symbol"]])[0]
    rows, dates = _range_rows(ctx, symbol, args, minimum=2)
    closes = _closes(rows, symbol)
    simple = analytics.simple_return(closes)
    payload = {
        "symbol": symbol,
        "start_date": _iso(closes.index[0]),
        "end_date": _iso(closes.index[-1]),
        "start_close": _num(closes.iloc[0]),
        "end_close": _num(closes.iloc[-1]),
        "simple_return": _num(simple),
        "log_return": _num(np.log1p(simple)),
        "sessions": int(len(closes)),
        "convention": "close of first session >= start to close of last session <= end",
    }
    outputs = {key: payload[key] for key in ("simple_return", "log_return", "start_close", "end_close")}
    return ctx.build_result("period_return", {"symbol": symbol, **dates}, rows, payload, outputs)


def realized_volatility(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Annualized sample volatility of the trailing ``window`` log returns ending at ``end``."""

    symbol = ctx.require_symbols([args["symbol"]])[0]
    window = _window(args["window"], ctx.policy)
    end = ctx.resolve_end(args["end"])
    history = ctx.bars([symbol], None, end).sort_values("date")
    rows = history.tail(window + 1).reset_index(drop=True)
    if len(rows) < window + 1:
        raise InsufficientDataError(
            f"{symbol} has {len(rows)} confirmed closes on or before {end.isoformat()}; "
            f"a {window}-return window needs {window + 1}"
        )
    closes = _closes(rows, symbol)
    returns = analytics.log_returns(closes)
    annualized = analytics.realized_volatility(closes)
    payload = {
        "symbol": symbol,
        "window": window,
        "window_start_date": _iso(closes.index[0]),
        "end_date": _iso(closes.index[-1]),
        "annualized_volatility": _num(annualized),
        "daily_volatility": _num(np.std(returns, ddof=1)),
        "mean_daily_log_return": _num(np.mean(returns)),
        "observations": int(len(returns)),
        "convention": "sample std (ddof=1) of daily log returns x sqrt(252)",
    }
    outputs = {key: payload[key] for key in ("annualized_volatility", "daily_volatility", "mean_daily_log_return")}
    return ctx.build_result(
        "realized_volatility",
        {"symbol": symbol, "window": window, "end": end.isoformat()},
        rows,
        payload,
        outputs,
    )


def rolling_correlation(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Trailing-window correlation of daily log returns for two symbols."""

    if str(args["a"]).strip().upper() == str(args["b"]).strip().upper():
        raise ToolInputError("a and b must be two different symbols")
    a, b = ctx.require_symbols([args["a"], args["b"]])
    window = _window(args["window"], ctx.policy)
    end = ctx.resolve_end(args["end"])
    history = ctx.bars([a, b], None, end)
    wide = history.pivot(index="date", columns="symbol", values="close").sort_index()
    aligned = wide.reindex(columns=[a, b]).dropna().tail(window + 1)
    if len(aligned) < window + 1:
        raise InsufficientDataError(
            f"{a} and {b} share {len(aligned)} confirmed closes on or before {end.isoformat()}; "
            f"a {window}-return window needs {window + 1}"
        )
    corr = analytics.return_correlation(aligned[a], aligned[b])
    used_dates = aligned.index
    rows = history.loc[history["date"].isin(used_dates)].reset_index(drop=True)
    payload = {
        "a": a,
        "b": b,
        "window": window,
        "window_start_date": _iso(used_dates[0]),
        "end_date": _iso(used_dates[-1]),
        "correlation": _num(corr),
        "observations": int(window),
        "convention": "Pearson correlation of daily log returns on dates where both symbols have a close",
    }
    return ctx.build_result(
        "rolling_correlation",
        {"a": a, "b": b, "window": window, "end": end.isoformat()},
        rows,
        payload,
        {"correlation": payload["correlation"]},
    )


def max_drawdown(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Maximum drawdown of closes over ``[start, end]``."""

    symbol = ctx.require_symbols([args["symbol"]])[0]
    rows, dates = _range_rows(ctx, symbol, args, minimum=2)
    closes = _closes(rows, symbol)
    result = analytics.max_drawdown(closes)
    index = closes.index
    payload = {
        "symbol": symbol,
        "start_date": _iso(index[0]),
        "end_date": _iso(index[-1]),
        "max_drawdown": _num(result.max_drawdown),
        "peak_date": _iso(index[result.peak_index]),
        "peak_close": _num(closes.iloc[result.peak_index]),
        "trough_date": _iso(index[result.trough_index]),
        "trough_close": _num(closes.iloc[result.trough_index]),
        "recovery_date": _iso(index[result.recovery_index]) if result.recovery_index is not None else None,
        "sessions": int(len(closes)),
        "convention": "min over t of close_t / running_max_close_t - 1 (non-positive)",
    }
    outputs = {key: payload[key] for key in ("max_drawdown", "peak_close", "trough_close")}
    return ctx.build_result("max_drawdown", {"symbol": symbol, **dates}, rows, payload, outputs)


def compare_returns(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Rank symbols by close-to-close return over a common range."""

    symbols = ctx.require_symbols(list(args["symbols"]))
    if len(symbols) < 2:
        raise ToolInputError("compare_returns needs at least two distinct symbols")
    start = ctx.resolve_start(args["start"])
    end = ctx.resolve_end(args["end"])
    if start > end:
        raise ToolInputError(f"start {start.isoformat()} is after end {end.isoformat()}")
    rows = ctx.bars(symbols, start, end)
    entries: list[dict[str, Any]] = []
    insufficient: list[str] = []
    for symbol in symbols:
        closes = _closes(rows, symbol)
        if len(closes) < 2:
            insufficient.append(symbol)
            continue
        entries.append(
            {
                "symbol": symbol,
                "simple_return": _num(analytics.simple_return(closes)),
                "start_date": _iso(closes.index[0]),
                "end_date": _iso(closes.index[-1]),
                "start_close": _num(closes.iloc[0]),
                "end_close": _num(closes.iloc[-1]),
            }
        )
    if not entries:
        raise InsufficientDataError("no requested symbol has two confirmed closes in the range")
    entries.sort(key=lambda item: (-float(item["simple_return"]), item["symbol"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    used = rows.loc[~rows["symbol"].isin(insufficient)].reset_index(drop=True)
    payload = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ranking": entries,
        "insufficient_data": insufficient,
        "convention": "per-symbol close of first session >= start to close of last session <= end",
    }
    outputs: dict[str, float | None] = {}
    for entry in entries:
        outputs[f"simple_return[{entry['symbol']}]"] = entry["simple_return"]
        outputs[f"start_close[{entry['symbol']}]"] = entry["start_close"]
        outputs[f"end_close[{entry['symbol']}]"] = entry["end_close"]
    return ctx.build_result(
        "compare_returns",
        {"symbols": symbols, "start": start.isoformat(), "end": end.isoformat()},
        used,
        payload,
        outputs,
    )


def _window(value: Any, policy: Policy) -> int:
    window = int(value)
    if not MIN_WINDOW <= window <= policy.max_window:
        raise ToolInputError(f"window must be between {MIN_WINDOW} and {policy.max_window}")
    return window


def market_tool_specs(policy: Policy) -> list[ToolSpec]:
    """Return the read-only tool specs, with policy limits stated in descriptions."""

    window_prop = {
        "type": "integer",
        "description": (
            f"Number of daily log returns in the trailing window ({MIN_WINDOW} to {policy.max_window}); "
            "the window uses window+1 confirmed closes."
        ),
    }
    return [
        ToolSpec(
            name="list_symbols",
            description=(
                "List the symbols that have at least one confirmed daily bar on or before the episode's "
                "as-of date, with each symbol's first and last available session and the latest confirmed "
                "session overall. Symbols that begin trading after the as-of date are not shown. "
                "Takes no arguments."
            ),
            input_schema=_object({}),
            handler=list_symbols,
        ),
        ToolSpec(
            name="get_daily_bars",
            description=(
                "Return confirmed daily OHLCV bars for one symbol between start and end (inclusive) with "
                "summary statistics computed over the full range. At most "
                f"{policy.max_rows_per_result} of the most recent rows are included and 'truncated' is true "
                "when more rows exist; use the analytics tools instead of computing statistics from "
                "truncated rows."
            ),
            input_schema=_object({"symbol": SYMBOL_PROP, "start": START_PROP, "end": END_PROP}),
            handler=get_daily_bars,
            fields={"symbol": "symbol", "start": "start", "end": "end"},
        ),
        ToolSpec(
            name="period_return",
            description=(
                "Close-to-close simple and log return for one symbol, from the close of the first confirmed "
                "session on or after start to the close of the last confirmed session on or before end."
            ),
            input_schema=_object({"symbol": SYMBOL_PROP, "start": START_PROP, "end": END_PROP}),
            handler=period_return,
            fields={"symbol": "symbol", "start": "start", "end": "end"},
        ),
        ToolSpec(
            name="realized_volatility",
            description=(
                "Annualized realized volatility for one symbol: the sample standard deviation (ddof=1) of "
                "the last `window` daily log returns ending at the last confirmed session on or before end, "
                "multiplied by sqrt(252)."
            ),
            input_schema=_object({"symbol": SYMBOL_PROP, "window": window_prop, "end": END_PROP}),
            handler=realized_volatility,
            fields={"symbol": "symbol", "window": "window", "end": "end"},
        ),
        ToolSpec(
            name="rolling_correlation",
            description=(
                "Trailing-window Pearson correlation between the daily log returns of two different symbols "
                "a and b, using the last `window` returns ending at the last common confirmed session on or "
                "before end. Only dates on which both symbols have a confirmed close are used."
            ),
            input_schema=_object(
                {
                    "a": {"type": "string", "description": "First ticker symbol, as returned by list_symbols."},
                    "b": {"type": "string", "description": "Second ticker symbol; must differ from a."},
                    "window": window_prop,
                    "end": END_PROP,
                }
            ),
            handler=rolling_correlation,
            fields={"a": "symbol", "b": "symbol", "window": "window", "end": "end"},
        ),
        ToolSpec(
            name="max_drawdown",
            description=(
                "Maximum peak-to-trough decline of one symbol's closes between start and end, reported as a "
                "non-positive fraction (for example -0.18), with the peak, trough, and recovery dates; "
                "recovery_date is null when the prior peak was not regained by end."
            ),
            input_schema=_object({"symbol": SYMBOL_PROP, "start": START_PROP, "end": END_PROP}),
            handler=max_drawdown,
            fields={"symbol": "symbol", "start": "start", "end": "end"},
        ),
        ToolSpec(
            name="compare_returns",
            description=(
                f"Rank 2 to {policy.max_symbols_per_call} symbols by close-to-close simple return over the "
                "same start-end range (same convention as period_return), highest first. Symbols without two "
                "confirmed closes in the range are listed under insufficient_data."
            ),
            input_schema=_object(
                {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Distinct ticker symbols to rank (2 to {policy.max_symbols_per_call}).",
                    },
                    "start": START_PROP,
                    "end": END_PROP,
                }
            ),
            handler=compare_returns,
            fields={"symbols": "symbols", "start": "start", "end": "end"},
        ),
    ]
