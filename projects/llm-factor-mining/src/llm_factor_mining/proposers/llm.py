"""LLM-guided proposer with auditable, replayable backends.

Components:

* :class:`LLMBackend` - protocol ``complete_json(system, user, schema) -> LLMResponse``;
* :class:`AnthropicBackend` - Claude Messages API with structured JSON output
  (``output_config.format``), adaptive thinking and a configurable effort.
  The ``anthropic`` package is imported lazily.  No sampling parameters are
  sent.  Server-side fallback is an explicit flag that is **off** by default,
  because a fallback silently changes the model under study; the served model
  id, stop reason and token usage of every call are recorded either way.
  Transient failures (429, 5xx/529, connection errors) are retried with
  bounded exponential backoff; configuration errors (400/401/403/404/413/422)
  raise :class:`LLMConfigurationError`, which aborts the run;
* :class:`CachedBackend` - wraps any backend and appends every request and
  outcome (responses *and* failures, fatal ones included, in order) to a
  JSON Lines file keyed by the SHA-256 of the canonical request.  By default
  the file is write-only (a live run is a new sample); ``reuse=True`` answers
  repeated requests from it;
* :class:`ReplayBackend` - offline, exact replay of such a file: the ``n``-th
  request with a given key receives the ``n``-th recorded outcome for that
  key, failures included, so a replayed search follows the live run's path; a
  cache miss raises :class:`ReplayMissError` (never silently falls through);
* :class:`FakeBackend` - scripted responses for tests;
* :class:`LLMProposer` - renders versioned prompt templates from
  ``llm_factor_mining/prompts`` (their SHA-256 is recorded with every call).

Prompts are anonymized by construction: the context carries no dates, asset
identifiers or date counts (the search sends degenerate and failed trials back
with fixed, count-free texts).  The formation statistics themselves still
imply the approximate formation sample size (``t / ICIR`` grows like the
square root of the number of IC dates).  Harness-written text (templates and
the grammar card) is checked at render time by
:func:`find_anonymization_violations` and a hit is fatal; rejected-proposal
text (the model's own expression plus a validator or harness message) is
checked for dates, month names and identifiers and redacted instead of
aborting the run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
import json
from pathlib import Path
import re
import string
import time
from typing import Any, Protocol, runtime_checkable

from ..jsonutil import canonical_json, json_safe, sha256_json, sha256_text
from .base import EvaluatedFeedback, Proposal, ProposalContext, ProposerError, RejectedFeedback


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16000
# The installed SDK refuses non-streaming requests whose max_tokens implies more
# than ten minutes of generation (3600 s * max_tokens / 128000 > 600 s).  This
# backend does not stream, so larger budgets are rejected up front.
MAX_NONSTREAMING_TOKENS = 21_333
DEFAULT_PROMPT_VERSION = "v2"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE_FORMAT = "llm-factor-mining/llm-cache-v1"
PROMPT_PACKAGE = "llm_factor_mining"
# HTTP statuses that signal a misconfigured run (bad request, key, permission,
# model id, size or schema): retrying cannot help and the run must stop.
FATAL_STATUS_CODES = frozenset({400, 401, 403, 404, 413, 422})
# Statuses worth retrying: timeout, conflict, rate limit and every 5xx (incl. 529).
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})
REDACTED_TEXT = "<redacted: contains a date, month name or asset identifier>"

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "rationale": {"type": "string"},
                    "economic_mechanism": {"type": "string"},
                },
                "required": ["expression", "rationale", "economic_mechanism"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class LLMError(ProposerError):
    """Recoverable LLM failure; carries the call metadata recorded so far."""

    kind = "llm_error"

    def __init__(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class LLMRefusalError(LLMError):
    kind = "refusal"


class LLMTruncatedError(LLMError):
    kind = "max_tokens"


class LLMFormatError(LLMError):
    kind = "format"


class LLMTransientError(LLMError):
    kind = "transient"


class LLMAPIError(LLMError):
    kind = "api_status"


class LLMConfigurationError(RuntimeError):
    """A non-retryable API error (bad request, key, permission, model id): fatal.

    Deliberately *not* a :class:`ProposerError`: a misconfigured run must stop
    instead of turning into an apparent scientific null result.
    """

    kind = "configuration"

    def __init__(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class ReplayMissError(RuntimeError):
    """A replayed run asked for a request that was never recorded (fatal)."""


class RecordedFailureError(RuntimeError):
    """Replay of a call whose live run failed with an unexpected, non-LLM exception (fatal)."""


# record kind of an exception that is neither an LLMError nor a configuration error
UNEXPECTED_KIND = "unexpected"


class PromptLeakError(RuntimeError):
    """Harness-written prompt text contains a date, year or forbidden identifier (fatal)."""


_ERROR_CLASSES: dict[str, type[LLMError]] = {
    cls.kind: cls
    for cls in (LLMRefusalError, LLMTruncatedError, LLMFormatError, LLMTransientError, LLMAPIError)
}
# outcomes that a served response determines (a replay may repeat them)
DETERMINISTIC_KINDS = frozenset({LLMRefusalError.kind, LLMTruncatedError.kind, LLMFormatError.kind})


# --------------------------------------------------------------------------
# backend protocol and request fingerprints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResponse:
    """Parsed JSON object plus call metadata (served model, usage, stop reason...)."""

    data: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        request_tag: str = "",
    ) -> LLMResponse: ...

    def identity(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RequestFingerprint:
    request_sha256: str
    identity_sha256: str
    key: str


def request_fingerprint(
    system: str,
    user: str,
    schema: Mapping[str, Any],
    identity: Mapping[str, Any],
    request_tag: str = "",
) -> RequestFingerprint:
    """Cache key = SHA-256 over the canonical request and the backend identity.

    ``request_tag`` (e.g. a replicate id) distinguishes otherwise identical
    requests; it is part of the key but is never sent to the model.
    """

    request_sha = sha256_json({"system": system, "user": user, "schema": schema, "tag": request_tag})
    identity_sha = sha256_json(identity)
    return RequestFingerprint(request_sha, identity_sha, sha256_text(f"{request_sha}:{identity_sha}"))


# --------------------------------------------------------------------------
# Anthropic backend
# --------------------------------------------------------------------------


def _optional_anthropic() -> Any | None:
    try:
        import anthropic  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        return None
    return anthropic


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return json_safe(dict(usage))
    for method in ("to_dict", "model_dump"):
        convert = getattr(usage, method, None)
        if callable(convert):
            try:
                return json_safe(convert())
            except TypeError:  # pragma: no cover - defensive
                continue
    names = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {name: json_safe(getattr(usage, name)) for name in names if hasattr(usage, name)}


def _stop_details(response: Any) -> dict[str, Any] | None:
    details = getattr(response, "stop_details", None)
    if details is None:
        return None
    return {
        "category": json_safe(getattr(details, "category", None)),
        "explanation": json_safe(getattr(details, "explanation", None)),
    }


class AnthropicBackend:
    """Claude Messages API backend returning schema-constrained JSON.

    ``client`` may be any object exposing ``messages.create`` (and
    ``beta.messages.create`` when ``use_fallback``); by default an
    ``anthropic.Anthropic()`` client is created on first use, resolving
    credentials from the environment.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_fallback: bool = False,
        client: Any | None = None,
        max_pause_continuations: int = 3,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError(f"unknown effort level {effort!r}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_tokens > MAX_NONSTREAMING_TOKENS:
            raise ValueError(
                f"max_tokens={max_tokens} exceeds {MAX_NONSTREAMING_TOKENS}, the largest budget the SDK "
                "accepts without streaming; this backend does not stream"
            )
        if max_retries < 0 or retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("retry settings must be non-negative")
        self.model = str(model)
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.use_fallback = bool(use_fallback)
        self.max_pause_continuations = int(max_pause_continuations)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self._sleep = sleep
        self._client = client

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "thinking": "adaptive",
            "server_side_fallback": self.use_fallback,
        }

    def _get_client(self) -> Any:
        if self._client is None:
            sdk = _optional_anthropic()
            if sdk is None:
                raise ImportError(
                    "AnthropicBackend needs the optional 'anthropic' package: "
                    "pip install 'llm-factor-mining[llm]'"
                )
            self._client = sdk.Anthropic()
        return self._client

    def build_request(self, system: str, user: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Keyword arguments for ``messages.create`` (no sampling parameters)."""

        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": json.loads(canonical_json(schema))},
            },
        }

    def _dispatch(self, client: Any, request: Mapping[str, Any]) -> Any:
        if self.use_fallback:
            return client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        return client.messages.create(**request)

    def _send(self, request: Mapping[str, Any]) -> Any:
        client = self._get_client()
        sdk = _optional_anthropic()
        connection_errors: tuple[type[BaseException], ...] = () if sdk is None else (sdk.APIConnectionError,)
        try:
            return self._dispatch(client, request)
        except connection_errors as exc:
            raise LLMTransientError(f"connection error: {exc}", {"error_type": type(exc).__name__}) from exc
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if not isinstance(status, int) or isinstance(status, bool):
                raise
            raise classify_status_error(status, exc) from exc

    def _send_with_retries(self, request: Mapping[str, Any], log: list[dict[str, Any]]) -> Any:
        """``_send`` with bounded exponential backoff on transient failures."""

        attempt = 0
        while True:
            try:
                return self._send(request)
            except LLMTransientError as exc:
                if attempt >= self.max_retries:
                    exc.metadata.update({"attempts": attempt + 1, "retry_log": list(log)})
                    raise
                delay = min(self.retry_max_delay, self.retry_base_delay * (2.0**attempt))
                log.append({"attempt": attempt + 1, "error": str(exc)[:300], "delay_seconds": delay})
                self._sleep(delay)
                attempt += 1

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        request_tag: str = "",
    ) -> LLMResponse:
        request = self.build_request(system, user, schema)
        started = time.perf_counter()
        retry_log: list[dict[str, Any]] = []
        response = self._send_with_retries(request, retry_log)
        continuations = 0
        while getattr(response, "stop_reason", None) == "pause_turn":
            if continuations >= self.max_pause_continuations:
                break
            continuations += 1
            messages = [*request["messages"], {"role": "assistant", "content": response.content}]
            request = {**request, "messages": messages}
            response = self._send_with_retries(request, retry_log)
        stop_reason = getattr(response, "stop_reason", None)
        metadata: dict[str, Any] = {
            "backend": "anthropic",
            "requested_model": self.model,
            "served_model": json_safe(getattr(response, "model", None)),
            "stop_reason": stop_reason,
            "stop_details": _stop_details(response),
            "usage": _usage_dict(getattr(response, "usage", None)),
            "request_id": json_safe(getattr(response, "_request_id", None)),
            "server_side_fallback": self.use_fallback,
            "pause_continuations": continuations,
            "transient_retries": len(retry_log),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        if retry_log:
            metadata["retry_log"] = retry_log
        if stop_reason == "refusal":  # checked before reading any content
            raise LLMRefusalError("the model declined the request (stop_reason=refusal)", metadata)
        if stop_reason == "max_tokens":
            raise LLMTruncatedError(
                f"response truncated at max_tokens={self.max_tokens}; retry with a larger budget", metadata
            )
        if stop_reason == "pause_turn":
            raise LLMTruncatedError("response still paused after the continuation limit", metadata)
        texts = [block.text for block in getattr(response, "content", []) if getattr(block, "type", None) == "text"]
        if not texts:
            raise LLMFormatError("response contains no text block", metadata)
        metadata["raw_text"] = texts[0]
        try:
            data = json.loads(texts[0])
        except json.JSONDecodeError as exc:
            raise LLMFormatError(f"response is not valid JSON: {exc}", metadata) from exc
        if not isinstance(data, dict):
            raise LLMFormatError("response JSON is not an object", metadata)
        return LLMResponse(data, metadata)


def classify_status_error(status: int, exc: BaseException) -> Exception:
    """Map an HTTP status error to a fatal, transient or recoverable backend error."""

    metadata = {"status_code": status, "error_type": type(exc).__name__}
    if status in FATAL_STATUS_CODES:
        return LLMConfigurationError(
            f"API returned status {status} ({type(exc).__name__}): the request, credentials, "
            f"permissions or model id are wrong; retrying cannot help: {exc}",
            metadata,
        )
    if status in TRANSIENT_STATUS_CODES or status >= 500:
        return LLMTransientError(f"API returned transient status {status}: {exc}", metadata)
    return LLMAPIError(f"API returned status {status}: {exc}", metadata)


# --------------------------------------------------------------------------
# cache / replay / fake
# --------------------------------------------------------------------------


def _load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("format") != CACHE_FORMAT:
            raise ValueError(f"{path}:{number} is not an llm-factor-mining cache record")
        records.append(record)
    return records


def _record_kind(record: Mapping[str, Any]) -> str:
    error = record["response"].get("error")
    return "ok" if error is None else str(error.get("kind"))


def _is_deterministic(record: Mapping[str, Any]) -> bool:
    kind = _record_kind(record)
    return kind == "ok" or kind in DETERMINISTIC_KINDS


def _response_from_record(record: Mapping[str, Any], source: str) -> LLMResponse:
    response = record["response"]
    metadata = {
        **response.get("metadata", {}),
        "cache": source,
        "request_key": record["key"],
        "occurrence": record.get("occurrence", 0),
    }
    error = response.get("error")
    if error is not None:
        kind = error.get("kind")
        message = str(error.get("message", "recorded LLM error"))
        if kind == LLMConfigurationError.kind:
            raise LLMConfigurationError(message, metadata)  # fatal on replay, as it was live
        if kind == UNEXPECTED_KIND:
            raise RecordedFailureError(f"the recorded run stopped here with {message}")
        cls = _ERROR_CLASSES.get(kind, LLMError)
        raise cls(message, metadata)
    return LLMResponse(json.loads(canonical_json(response["data"])), metadata)


def _group_by_key(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    return grouped


class CachedBackend:
    """Recording wrapper around a live backend.

    Every outcome of every request is appended in call order: parsed
    responses, deterministic failures of a served response (refusal,
    truncation, invalid JSON), failures that survived the backend's own
    retries (transient or other API errors) *and* fatal ones (configuration
    errors and any other exception, recorded before it is re-raised; a replay
    raises them again).  Each record carries its
    occurrence index for its request key, so :class:`ReplayBackend` can follow
    the live run's exact path, failures included.

    ``reuse=False`` (default) makes the file write-only: a live run is a new
    sample, so the wrapper refuses to start on an existing, non-empty file.
    ``reuse=True`` answers a repeated request from its latest deterministic
    record instead of calling the live backend again.
    """

    def __init__(self, inner: LLMBackend, path: str | Path, *, reuse: bool = False) -> None:
        self.inner = inner
        self.path = Path(path)
        self.reuse = bool(reuse)
        records = _load_cache(self.path)
        if records and not self.reuse:
            raise FileExistsError(
                f"{self.path} already holds {len(records)} recorded responses; record files are write-only "
                "(a live run is a new sample): use a fresh file, --backend replay, or reuse=True"
            )
        self._by_key = _group_by_key(records)

    def identity(self) -> dict[str, Any]:
        return self.inner.identity()

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
        self._by_key.setdefault(record["key"], []).append(record)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        request_tag: str = "",
    ) -> LLMResponse:
        identity = self.identity()
        fingerprint = request_fingerprint(system, user, schema, identity, request_tag)
        previous = self._by_key.get(fingerprint.key, [])
        if self.reuse:
            deterministic = [record for record in previous if _is_deterministic(record)]
            if deterministic:
                return _response_from_record(deterministic[-1], "hit")
        base = {
            "format": CACHE_FORMAT,
            "key": fingerprint.key,
            "occurrence": len(previous),
            "request_sha256": fingerprint.request_sha256,
            "identity_sha256": fingerprint.identity_sha256,
            "identity": identity,
            "request": {"system": system, "user": user, "schema": schema, "tag": request_tag},
        }
        try:
            response = self.inner.complete_json(system, user, schema, request_tag=request_tag)
        except (LLMError, LLMConfigurationError) as exc:
            self._append(
                {
                    **base,
                    "response": {"error": {"kind": exc.kind, "message": str(exc)}, "metadata": json_safe(exc.metadata)},
                }
            )
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised unchanged
            message = f"{type(exc).__name__}: {exc}"[:2000]
            self._append({**base, "response": {"error": {"kind": UNEXPECTED_KIND, "message": message}, "metadata": {}}})
            raise
        self._append({**base, "response": {"data": response.data, "metadata": json_safe(response.metadata)}})
        return LLMResponse(
            response.data,
            {**response.metadata, "cache": "miss", "request_key": fingerprint.key, "occurrence": base["occurrence"]},
        )


class ReplayBackend:
    """Offline backend answering only from a :class:`CachedBackend` file."""

    def __init__(self, path: str | Path, *, identity: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"replay file {self.path} does not exist")
        records = _load_cache(self.path)
        self._by_key = _group_by_key(records)
        self._cursor: Counter[str] = Counter()
        if identity is None:
            identities = {canonical_json(record["identity"]): record["identity"] for record in records}
            if len(identities) > 1:
                raise ValueError("replay file mixes backend identities; pass identity= explicitly")
            identity = next(iter(identities.values()), {"backend": "replay-empty"})
        self._identity = dict(identity)

    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        request_tag: str = "",
    ) -> LLMResponse:
        fingerprint = request_fingerprint(system, user, schema, self._identity, request_tag)
        records = self._by_key.get(fingerprint.key)
        if not records:
            raise ReplayMissError(
                f"no recorded response for request {fingerprint.key[:16]} in {self.path}; replay is "
                "exact, so the prompt text, schema, backend settings and request tag must all match"
            )
        position = self._cursor[fingerprint.key]
        if position < len(records):
            record = records[position]
            self._cursor[fingerprint.key] += 1
        elif _is_deterministic(records[-1]):
            record = records[-1]  # a served response determines the outcome of an identical repeat
        else:
            raise ReplayMissError(
                f"request {fingerprint.key[:16]} was sent {position + 1} times but only {len(records)} "
                f"outcome(s) are recorded in {self.path}, the last of which was not deterministic"
            )
        return _response_from_record(record, "replay")


class FakeBackend:
    """Scripted backend for tests: a list of JSON objects or a callable."""

    def __init__(
        self,
        responses: Sequence[Mapping[str, Any]] | Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]],
        *,
        model: str = "fake-model",
    ) -> None:
        self._responses = responses
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def identity(self) -> dict[str, Any]:
        return {"backend": "fake", "model": self.model}

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        request_tag: str = "",
    ) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": schema, "tag": request_tag})
        if callable(self._responses):
            data = self._responses(system, user, schema)
        else:
            index = len(self.calls) - 1
            if index >= len(self._responses):
                raise LLMFormatError("FakeBackend ran out of scripted responses", {"served_model": self.model})
            data = self._responses[index]
        metadata = {
            "backend": "fake",
            "requested_model": self.model,
            "served_model": self.model,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": len(user) // 4, "output_tokens": 0},
        }
        return LLMResponse(json.loads(canonical_json(data)), metadata)


# --------------------------------------------------------------------------
# prompts and anonymization
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt file; ``sha256`` is over the exact file bytes."""

    name: str
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    def render(self, **values: Any) -> str:
        return string.Template(self.text).substitute({key: str(value) for key, value in values.items()})


def load_prompt(name: str) -> PromptTemplate:
    """Load ``prompts/<name>.md`` from the installed package."""

    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"invalid prompt name {name!r}")
    resource = resources.files(PROMPT_PACKAGE).joinpath("prompts", f"{name}.md")
    return PromptTemplate(name, resource.read_text(encoding="utf-8"))


_ISO_DATE = re.compile(r"\b(?:18|19|20|21)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b")
# A year may end a sentence ("in 2008."), so only a following digit or word
# character (or a decimal fraction such as "2008.5") disqualifies it.
_YEAR = re.compile(r"(?<![\w.])(?:18|19|20|21)\d{2}(?!\w|\.\d)")
# Unambiguous month names in any case; "March" and "May" (also ordinary English
# words) and the usual abbreviations only when capitalized.
_MONTHS = re.compile(
    r"\b(?:january|february|april|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_CAPITALIZED_MONTHS = re.compile(
    r"\b(?:March|May|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b\.?"
)


def find_anonymization_violations(
    text: str,
    *,
    forbidden_tokens: Iterable[str] = (),
    check_years: bool = True,
) -> list[str]:
    """Return human-readable reasons why ``text`` could reveal the sample.

    Flags ISO-style dates, month names (full names in any case; "March",
    "May" and abbreviations such as "Sept." when capitalized), (optionally)
    four-digit year-like numbers and any whole-word occurrence of
    ``forbidden_tokens`` (e.g. the panel's symbols).  The check is a
    heuristic: it cannot detect distributional fingerprints of a period.
    """

    problems = [f"date {match.group(0)!r}" for match in _ISO_DATE.finditer(text)]
    problems += [f"month name {match.group(0)!r}" for match in _MONTHS.finditer(text)]
    problems += [f"month name {match.group(0)!r}" for match in _CAPITALIZED_MONTHS.finditer(text)]
    if check_years:
        problems += [f"year-like number {match.group(0)!r}" for match in _YEAR.finditer(text)]
    for token in forbidden_tokens:
        token = str(token).strip()
        if token and re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", text):
            problems.append(f"forbidden identifier {token!r}")
    return problems


def _format_feedback(items: Sequence[EvaluatedFeedback]) -> str:
    if not items:
        return "- (none yet)"
    lines = []
    for item in items:
        lines.append(
            f"- {item.expression} | IC={item.formation_ic:+.4f} ICIR={item.formation_icir:+.3f} "
            f"t={item.formation_tstat:+.2f} turnover={item.formation_turnover:.2f} "
            f"complexity={item.complexity:g}"
        )
    return "\n".join(lines)


def _format_rejected(items: Sequence[RejectedFeedback]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item.expression} -> {', '.join(item.codes)}: {item.message}" for item in items)


def redact_rejected(
    items: Sequence[RejectedFeedback], *, forbidden_tokens: Iterable[str] = ()
) -> tuple[list[RejectedFeedback], int]:
    """Replace model-written rejected text that looks like a date or identifier.

    Rejected proposals are the model's own text; echoing a month name or a
    ticker back is not a leak from the sample, but it would trip the prompt
    check.  Such items keep their error codes and lose their text.
    """

    tokens = tuple(forbidden_tokens)
    out: list[RejectedFeedback] = []
    redacted = 0
    for item in items:
        text = f"{item.expression}\n{item.message}"
        if find_anonymization_violations(text, forbidden_tokens=tokens, check_years=False):
            out.append(RejectedFeedback(REDACTED_TEXT, item.codes, REDACTED_TEXT))
            redacted += 1
        else:
            out.append(item)
    return out, redacted


# --------------------------------------------------------------------------
# proposer
# --------------------------------------------------------------------------


class LLMProposer:
    """Asks an :class:`LLMBackend` for proposals given formation-only feedback."""

    name = "llm"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        replicate: int = 0,
        run_tag: str = "",
        forbidden_tokens: Iterable[str] = (),
        max_feedback: int = 10,
        max_rejected: int = 8,
    ) -> None:
        if run_tag and (";" in run_tag or "=" not in run_tag):
            raise ValueError("run_tag must be 'key=value' pairs joined by ',' (no ';')")
        self.backend = backend
        self.prompt_version = prompt_version
        self.replicate = int(replicate)
        self.run_tag = str(run_tag)
        self.forbidden_tokens = tuple(str(token) for token in forbidden_tokens)
        self.max_feedback = int(max_feedback)
        self.max_rejected = int(max_rejected)
        self.system_template = load_prompt(f"system_{prompt_version}")
        self.user_template = load_prompt(f"propose_{prompt_version}")
        self.calls: list[dict[str, Any]] = []

    @property
    def request_tag(self) -> str:
        """Cache-key tag (never sent to the model): replicate id plus the run tag."""

        tag = f"replicate={self.replicate}"
        return tag if not self.run_tag else f"{tag};{self.run_tag}"

    def _blocks(self, context: ProposalContext) -> tuple[dict[str, str], int]:
        rejected, n_redacted = redact_rejected(
            context.rejected[: self.max_rejected], forbidden_tokens=self.forbidden_tokens
        )
        blocks = {
            "top_block": _format_feedback(context.top[: self.max_feedback]),
            "bottom_block": _format_feedback(context.bottom[: self.max_feedback]),
            "rejected_block": _format_rejected(rejected),
        }
        if "$last_block" in self.user_template.text or "${last_block}" in self.user_template.text:
            blocks["last_block"] = _format_feedback(context.last_round[: self.max_feedback])
        return blocks, n_redacted

    def render(self, context: ProposalContext) -> tuple[str, str]:
        """Render and anonymization-check the system and user prompts.

        Harness-written text (templates and grammar card) must pass the check,
        otherwise :class:`PromptLeakError` aborts the run.  The integer counters
        (round, requested, trials, budget) are excluded from the year check, and
        model-written rejected text is redacted by :func:`redact_rejected`.
        Feedback statistics are formation-window numbers, so they are checked
        for dates, month names and identifiers but not for year-like values.
        """

        system = self.system_template.render()
        blocks, _ = self._blocks(context)
        counters = {
            "n_requested": context.n_requested,
            "round_index": context.round_index,
            "n_trials_so_far": context.n_trials_so_far,
            "budget_remaining": context.budget_remaining,
        }
        skeleton = self.user_template.render(
            grammar=context.grammar, **{key: "0" for key in counters}, **{key: "" for key in blocks}
        )
        problems = find_anonymization_violations(system + "\n" + skeleton, forbidden_tokens=self.forbidden_tokens)
        problems += find_anonymization_violations(
            "\n".join(blocks.values()), forbidden_tokens=self.forbidden_tokens, check_years=False
        )
        if problems:
            raise PromptLeakError("prompt is not anonymized: " + "; ".join(sorted(set(problems))))
        return system, self.user_template.render(grammar=context.grammar, **counters, **blocks)

    def propose(self, context: ProposalContext) -> list[Proposal]:
        system, user = self.render(context)
        _, n_redacted = self._blocks(context)
        call: dict[str, Any] = {
            "round_index": context.round_index,
            "prompt_version": self.prompt_version,
            "system_template_sha256": self.system_template.sha256,
            "user_template_sha256": self.user_template.sha256,
            "system_sha256": sha256_text(system),
            "user_sha256": sha256_text(user),
            "request_tag": self.request_tag,
            "n_redacted": n_redacted,
        }
        try:
            response = self.backend.complete_json(system, user, PROPOSAL_SCHEMA, request_tag=self.request_tag)
        except LLMError as exc:
            self.calls.append({**call, "status": exc.kind, "error": str(exc), **_call_meta(exc.metadata)})
            raise
        items = response.data.get("proposals")
        if not isinstance(items, list):
            self.calls.append({**call, "status": "format", **_call_meta(response.metadata)})
            raise LLMFormatError("response has no 'proposals' list", response.metadata)
        proposals: list[Proposal] = []
        for position, item in enumerate(items[: context.n_requested]):
            if not isinstance(item, Mapping) or not isinstance(item.get("expression"), str):
                continue
            proposals.append(
                Proposal(
                    expression=item["expression"],
                    rationale=str(item.get("rationale", "")),
                    metadata={
                        "generator": "llm",
                        "economic_mechanism": str(item.get("economic_mechanism", "")),
                        "position": position,
                        "served_model": response.metadata.get("served_model"),
                        "request_key": response.metadata.get("request_key"),
                        "user_sha256": call["user_sha256"],
                        "user_template_sha256": self.user_template.sha256,
                        "system_template_sha256": self.system_template.sha256,
                        "prompt_version": self.prompt_version,
                    },
                )
            )
        self.calls.append(
            {
                **call,
                "status": "ok",
                "n_returned": len(items),
                "n_used": len(proposals),
                **_call_meta(response.metadata),
            }
        )
        return proposals

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_version": self.prompt_version,
            "system_template_sha256": self.system_template.sha256,
            "user_template_sha256": self.user_template.sha256,
            "schema_sha256": sha256_json(PROPOSAL_SCHEMA),
            "backend": self.backend.identity(),
            "replicate": self.replicate,
            "run_tag": self.run_tag,
            "request_tag": self.request_tag,
            "uses_feedback": True,
        }

    def provenance(self) -> dict[str, Any]:
        """Aggregate call log: served models, stop reasons, cache use and token usage."""

        usage: Counter[str] = Counter()
        for call in self.calls:
            for key, value in (call.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
        return {
            "n_calls": len(self.calls),
            "served_models": sorted({str(call["served_model"]) for call in self.calls if call.get("served_model")}),
            "status_counts": dict(Counter(call["status"] for call in self.calls)),
            "stop_reasons": dict(Counter(str(call.get("stop_reason")) for call in self.calls)),
            "cache_counts": dict(Counter(str(call.get("cache")) for call in self.calls if call.get("cache"))),
            "usage_totals": dict(usage),
            "n_redacted_rejections": int(sum(int(call.get("n_redacted", 0)) for call in self.calls)),
            "transient_retries": int(sum(int(call.get("transient_retries") or 0) for call in self.calls)),
            "prompt_hashes": sorted(
                {call["system_template_sha256"] for call in self.calls}
                | {call["user_template_sha256"] for call in self.calls}
            ),
            "calls": self.calls,
        }


def _call_meta(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "served_model",
        "requested_model",
        "stop_reason",
        "usage",
        "cache",
        "request_key",
        "occurrence",
        "stop_details",
        "transient_retries",
        "status_code",
    )
    return {key: json_safe(metadata.get(key)) for key in keys if key in metadata}


__all__ = [
    "AnthropicBackend",
    "CachedBackend",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_VERSION",
    "DETERMINISTIC_KINDS",
    "FALLBACK_BETA",
    "FATAL_STATUS_CODES",
    "FakeBackend",
    "LLMAPIError",
    "LLMBackend",
    "LLMConfigurationError",
    "LLMError",
    "LLMFormatError",
    "LLMProposer",
    "LLMRefusalError",
    "LLMResponse",
    "LLMTransientError",
    "LLMTruncatedError",
    "MAX_NONSTREAMING_TOKENS",
    "PROPOSAL_SCHEMA",
    "PromptLeakError",
    "PromptTemplate",
    "ReplayBackend",
    "RecordedFailureError",
    "ReplayMissError",
    "classify_status_error",
    "find_anonymization_violations",
    "load_prompt",
    "redact_rejected",
    "request_fingerprint",
]
