"""Claude backend: one Messages API step of a manually driven agent loop.

The copilot runs its own loop rather than the SDK's tool runner. The research
design depends on four things the loop has to own:

* **policy gating**: every ``tool_use`` block passes :class:`~marketdata_agent.policy.PolicyGate`
  before any handler runs, and denials go back to the model as ``is_error`` results;
* **audit logging**: every model call (requested and served model, stop reason,
  usage) and every tool attempt is written to the hash-chained audit log;
* **deterministic replay**: requests are fingerprinted so that recorded turns can
  be replayed offline (:mod:`marketdata_agent.backends.replay`);
* **backend-agnostic evaluation**: the same loop drives Claude, the scripted
  baselines and replayed runs, so the benchmark measures agents under an identical
  harness.

This module therefore implements only :meth:`AnthropicBackend.step`, a single
request/response exchange (including ``pause_turn`` continuation). The loop is
:class:`marketdata_agent.agent.Copilot`.

Request shape (current API): ``client.messages.create(model, max_tokens, system,
messages, tools, thinking={"type": "adaptive"}, output_config={"effort": ...})``.
No sampling parameters are sent, and no forced ``tool_choice`` or assistant
prefill is used. The SDK refuses non-streaming requests whose ``max_tokens``
could take longer than ten minutes, so above :data:`NONSTREAMING_MAX_TOKENS`
the same request is sent with ``messages.stream(...)`` and read with
``get_final_message()`` (``stream=True`` forces this for any size). Server-side fallback (``fallbacks="default"``) can change which
model serves a request, so it is an explicit flag. It is off by default and off
in benchmark configurations, and ``response.model`` is recorded for every call.
The ``anthropic`` package is imported lazily; the rest of the package works
without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

from ..audit import usage_to_dict
from .base import (
    STOP_PAUSE_TURN,
    STOP_REFUSAL,
    BackendError,
    BackendTurn,
    UsageTotals,
    block_field,
    to_plain,
)


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "high"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
THINKING = {"type": "adaptive"}
# The SDK raises before sending a non-streaming request whose expected duration,
# 3600 s x max_tokens / 128000, exceeds 600 s: above 21,333 tokens.
NONSTREAMING_MAX_TOKENS = 21_333


@dataclass(frozen=True)
class AnthropicConfig:
    """Request configuration; recorded verbatim in every episode manifest."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str = DEFAULT_EFFORT
    fallback: bool = False
    max_continuations: int = 4
    stream: bool | None = None

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("model must be a non-empty model id")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.max_continuations < 0:
            raise ValueError("max_continuations must be non-negative")

    @property
    def streaming(self) -> bool:
        """Whether requests go through ``messages.stream`` (automatic above the non-streaming limit)."""

        return bool(self.stream) if self.stream is not None else self.max_tokens > NONSTREAMING_MAX_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "thinking": dict(THINKING),
            "fallback": self.fallback,
            "fallback_beta": FALLBACK_BETA if self.fallback else None,
            "max_continuations": self.max_continuations,
            "transport": "stream" if self.streaming else "create",
        }


def _sdk() -> Any | None:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - exercised only without the optional extra
        return None
    return anthropic


class AnthropicBackend:
    """:class:`~marketdata_agent.backends.base.LLMBackend` over the Claude Messages API.

    ``client`` may be injected (a duck-typed fake in tests). Otherwise
    ``anthropic.Anthropic()`` is created on first use, which reads credentials
    from the environment or an ``ant auth login`` profile.
    """

    name = "anthropic"

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        *,
        client: Any | None = None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or AnthropicConfig()
        self._client = client
        self._timer = timer

    @property
    def client(self) -> Any:
        if self._client is None:
            sdk = _sdk()
            if sdk is None:
                raise BackendError(
                    "missing_dependency",
                    "the Claude backend needs the optional 'anthropic' package: pip install 'marketdata-agent[llm]'",
                )
            self._client = sdk.Anthropic()
        return self._client

    def describe(self) -> dict[str, Any]:
        return self.config.to_dict()

    def request_params(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Keyword arguments for ``messages.create`` (without the fallback beta fields)."""

        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": list(messages),
            "thinking": dict(THINKING),
            "output_config": {"effort": self.config.effort},
        }
        if tools:
            params["tools"] = list(tools)
        return params

    def step(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> BackendTurn:
        """Send one request; continue a ``pause_turn`` up to ``max_continuations`` times.

        On ``refusal`` the content is not read, so the turn carries no blocks.
        Content from ``pause_turn`` segments is concatenated into one assistant
        turn, which is equivalent to consecutive assistant messages. A
        ``max_tokens`` stop is returned as is, and the loop treats it as recoverable.
        """

        history = list(messages)
        content: list[Any] = []
        usage = UsageTotals()
        request_ids: list[str] = []
        fallback_events: list[Mapping[str, Any]] = []
        segment_models: list[str] = []
        continuations = 0
        started = self._timer()
        while True:
            response, request_id = self._create(system, history, tools)
            if request_id:
                request_ids.append(str(request_id))
            usage.add(usage_to_dict(getattr(response, "usage", None)))
            served = getattr(response, "model", None)
            if served:
                segment_models.append(str(served))
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == STOP_REFUSAL:  # checked before content is read
                return BackendTurn(
                    blocks=(),
                    raw_content=(),
                    stop_reason=STOP_REFUSAL,
                    served_model=served,
                    requested_model=self.config.model,
                    usage=usage.totals or None,
                    request_ids=tuple(request_ids),
                    latency_seconds=self._timer() - started,
                    continuations=continuations,
                    stop_details=to_plain(getattr(response, "stop_details", None)),
                    segment_models=tuple(segment_models),
                )
            segment = list(response.content)
            content.extend(segment)
            fallback_events.extend(to_plain(b) for b in segment if block_field(b, "type") == "fallback")
            if stop_reason == STOP_PAUSE_TURN and continuations < self.config.max_continuations:
                continuations += 1
                history = [*history, {"role": "assistant", "content": segment}]
                continue
            return BackendTurn.from_content(
                content,
                stop_reason=stop_reason,
                served_model=served,
                requested_model=self.config.model,
                usage=usage.totals or None,
                request_ids=tuple(request_ids),
                latency_seconds=self._timer() - started,
                continuations=continuations,
                stop_details=to_plain(getattr(response, "stop_details", None)),
                fallback_events=tuple(fallback_events),
                segment_models=tuple(segment_models),
            )

    def _send(
        self, system: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, str | None]:
        """Send one request; return ``(message, request_id)``."""

        params = self.request_params(system, messages, tools)
        api = self.client.beta.messages if self.config.fallback else self.client.messages
        extra: dict[str, Any] = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.config.fallback else {}
        if self.config.streaming:
            with api.stream(**params, **extra) as stream:
                message = stream.get_final_message()
                request_id = getattr(stream, "request_id", None)
            return message, request_id or getattr(message, "_request_id", None)
        message = api.create(**params, **extra)
        return message, getattr(message, "_request_id", None)

    def _create(
        self, system: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, str | None]:
        sdk = _sdk()
        if sdk is None:  # pragma: no cover - a fake client without the SDK installed
            return self._send(system, messages, tools)
        # Most specific first. The SDK already retries 408/409/429/5xx and
        # connection errors (max_retries); anything that reaches here is final.
        try:
            return self._send(system, messages, tools)
        except sdk.RateLimitError as exc:
            raise BackendError("rate_limit", str(exc), retryable=True, status_code=429, request_id=_rid(exc)) from exc
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as exc:
            raise BackendError("authentication", str(exc), status_code=exc.status_code, request_id=_rid(exc)) from exc
        except sdk.NotFoundError as exc:
            raise BackendError("not_found", str(exc), status_code=404, request_id=_rid(exc)) from exc
        except sdk.BadRequestError as exc:
            raise BackendError("bad_request", str(exc), status_code=400, request_id=_rid(exc)) from exc
        except sdk.APIStatusError as exc:
            server = exc.status_code >= 500
            raise BackendError(
                "server_error" if server else "api_error",
                str(exc),
                retryable=server,
                status_code=exc.status_code,
                request_id=_rid(exc),
            ) from exc
        except sdk.APIConnectionError as exc:  # includes APITimeoutError
            raise BackendError("connection", str(exc), retryable=True) from exc
        except sdk.AnthropicError as exc:
            raise BackendError("sdk_error", f"{type(exc).__name__}: {exc}") from exc
        except TypeError as exc:
            # The SDK raises TypeError before any request when no credentials resolve.
            if "authentication method" not in str(exc):
                raise
            raise BackendError("missing_credentials", str(exc)) from exc
        except ValueError as exc:
            # The SDK refuses a request it would have to stream; the config should have
            # selected streaming, so this is a configuration error, not a crash.
            if "Streaming is required" not in str(exc):
                raise
            raise BackendError("bad_request", str(exc)) from exc


def _rid(exc: Any) -> str | None:
    value = getattr(exc, "request_id", None)
    return str(value) if value else None
