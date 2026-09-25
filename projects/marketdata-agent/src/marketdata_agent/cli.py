"""Command-line entry point.

``marketdata-agent tools``
    Print the strict tool definitions exactly as they are sent to the Messages API.
``marketdata-agent verify-audit PATH [--records N --head SHA | --summary summary.json]``
    Verify an audit log's hash chain, optionally against an anchor: an explicit
    record count and head hash, or the head recorded in a committed summary.
``marketdata-agent ask "QUESTION" --as-of YYYY-MM-DD``
    Interactive copilot backed by Claude. Prints the answer, its citations, the
    grounding report, the served model, and the verified audit path. Each call
    writes files under a fresh, timestamped episode id.
    Server-side fallback is on by default here (``--no-fallback`` turns it off),
    and the served model is always shown.
``marketdata-agent bench run --agent NAME``
    Run one agent over the generated suite. The scripted baselines are offline.
    ``anthropic`` needs credentials and keeps fallback off unless ``--fallback``
    is given. ``--no-clock`` (ablation A1), ``--prompt-version`` (A2, A6),
    ``--tools none`` (closed book, A5) and ``--tools decoy`` (an offered but
    refused ``execute_order``) select the arm. Every turn is recorded; an
    interrupted run is continued with ``--resume``, which reuses recorded turns.
    Exit codes: 0 valid run, 1 audit verification failed, 2 usage error,
    3 invalid run (unrecovered backend error or served-model mismatch).
``marketdata-agent bench tasks``
    Write the generated suite to JSON.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from . import __version__
from .agent import DEFAULT_MAX_STEPS, PROMPT_VARIANTS, PROMPT_VERSION, Copilot, Episode, default_episode_id
from .audit import AuditLog, ChainHead, verify_chain
from .backends import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    AnthropicBackend,
    AnthropicConfig,
    LLMBackend,
)
from .clock import parse_date
from .errors import AgentToolError
from .policy import Policy
from .sources import BarSource, FrameBarSource, StoreBarSource
from .tools import default_registry


BackendFactory = Callable[[argparse.Namespace], LLMBackend]
CREDENTIAL_ERRORS = frozenset({"authentication", "missing_credentials", "missing_dependency"})
EXIT_INVALID_RUN = 3
CLOSED_BOOK_PROMPT = "copilot_closed_book_v1"


class UsageError(ValueError):
    """A command-line mistake reported as one line with exit status 2."""


def build_parser() -> argparse.ArgumentParser:
    from .bench import BASELINE_NAMES

    parser = argparse.ArgumentParser(
        prog="marketdata-agent",
        description="Governed, point-in-time-safe trading copilot over the MarketData gateway.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    tools = commands.add_parser("tools", help="print the strict tool definitions (Claude API format)")
    tools.add_argument("--compact", action="store_true", help="print compact JSON")

    verify = commands.add_parser("verify-audit", help="verify an audit log's SHA-256 hash chain")
    verify.add_argument("path", help="audit JSONL file")
    verify.add_argument("--records", type=int, default=None, help="expected record count (external anchor)")
    verify.add_argument("--head", default=None, help="expected head hash (external anchor)")
    verify.add_argument("--summary", default=None, help="read the anchor from a benchmark summary.json")
    verify.add_argument("--agent", default=None, help="agent whose head to use from a combined summary.json")

    ask = commands.add_parser("ask", help="ask the copilot one question (Claude backend)")
    ask.add_argument("question", help="the question, in quotes")
    ask.add_argument("--as-of", required=True, help="information cutoff, YYYY-MM-DD")
    ask.add_argument("--source", choices=("synthetic", "store"), default="synthetic", help="bar source (default synthetic)")
    ask.add_argument("--symbols", default=None, help="comma-separated universe restriction for --source store")
    ask.add_argument("--data-home", default=None, help="store root (default: QUANT_DATA_HOME)")
    _model_arguments(ask)
    fallback = ask.add_mutually_exclusive_group()
    fallback.add_argument("--fallback", dest="fallback", action="store_true", help="server-side fallback on (default)")
    fallback.add_argument("--no-fallback", dest="fallback", action="store_false", help="disable server-side fallback")
    ask.set_defaults(fallback=True)
    ask.add_argument("--run-dir", default="runs/ask", help="where the audit log and transcript are written")
    ask.add_argument("--json", action="store_true", help="print the episode as JSON")

    bench = commands.add_parser("bench", help="benchmark commands")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    run = bench_commands.add_parser("run", help="run one agent over the generated suite")
    run.add_argument("--agent", required=True, choices=(*BASELINE_NAMES, "anthropic"))
    run.add_argument("--tasks", type=int, default=None, help="stratified subset size (default: all)")
    run.add_argument("--seed", type=int, default=None, help="task-generation seed (default: the suite default)")
    run.add_argument("--out", default=None, help="output directory (default: runs/bench/<agent>)")
    _model_arguments(run)
    run.add_argument("--fallback", action="store_true", help="enable server-side fallback (off for benchmarks)")
    run.add_argument("--no-clock", action="store_true", help="ablation A1: serve explicitly named dates after the cutoff")
    run.add_argument("--prompt-version", default=None, choices=sorted(PROMPT_VARIANTS), help="system prompt (ablations A2, A6)")
    run.add_argument(
        "--tools",
        default="default",
        choices=("default", "decoy", "none"),
        help="tool set: default; decoy adds a refused execute_order (RQ3); none is the closed-book arm (A5)",
    )
    run.add_argument("--record", default=None, help="record Claude turns to this JSONL (default: <out>/recordings.jsonl)")
    run.add_argument("--replay", default=None, help="replay recorded Claude turns instead of calling the API")
    run.add_argument("--resume", action="store_true", help="rerun an interrupted run, reusing its recorded turns")
    run.add_argument("--overwrite", action="store_true", help="replace an existing run directory (recordings are kept)")
    run.add_argument("--max-attempts", type=int, default=3, help="attempts per episode on retryable API errors")
    run.add_argument("--backoff", type=float, default=30.0, help="initial retry wait in seconds (doubles per attempt)")

    tasks = bench_commands.add_parser("tasks", help="write the generated suite as JSON")
    tasks.add_argument("--seed", type=int, default=None)
    tasks.add_argument("--tasks", type=int, default=None)
    tasks.add_argument("--out", default="-", help="output path, or - for stdout")
    return parser


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model id (default {DEFAULT_MODEL})")
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORT_LEVELS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)


def _config(args: argparse.Namespace) -> AnthropicConfig:
    return AnthropicConfig(model=args.model, max_tokens=args.max_tokens, effort=args.effort, fallback=bool(args.fallback))


def _anthropic_backend(args: argparse.Namespace) -> LLMBackend:
    return AnthropicBackend(_config(args))


def _source(args: argparse.Namespace) -> BarSource:
    if args.source == "store":
        universe = [s for s in args.symbols.split(",") if s.strip()] if args.symbols else None
        return StoreBarSource.from_environment(data_home=args.data_home, universe=universe)
    from .bench.generator import dataset_frame

    return FrameBarSource(dataset_frame())


def _print_episode(episode: Episode, audit_path: Path, verified: bool, source: str) -> None:
    print(episode.answer or f"(no answer; status={episode.status})")
    print()
    if episode.status != "completed":
        print(f"Status: {episode.status}" + (f" ({episode.error['kind']}: {episode.error['message']})" if episode.error else ""))
    cited = episode.grounding.cited_ids
    if cited:
        print("Citations:")
        for rid in cited:
            result = episode.results.get(rid)
            if result is None:
                print(f"  [r:{rid}] UNKNOWN: no result with this id exists in the episode")
            else:
                args = json.dumps(dict(result.provenance.args), sort_keys=True)
                print(f"  [r:{rid}] {result.tool} {args} rows={result.provenance.row_count} last_date={result.provenance.last_date}")
    g = episode.grounding
    if g.error:
        print(f"Grounding: the verifier failed ({g.error}); the answer is treated as not grounded")
    elif g.n_claims:
        print(
            f"Grounding: {g.n_supported}/{g.n_claims} numeric claims supported "
            f"({100 * (g.grounding_rate or 0):.1f}%), citation rate {100 * (g.citation_rate or 0):.1f}%"
        )
        for check in g.unsupported:
            note = f", {check.note}" if check.note else ""
            print(f"  UNSUPPORTED {check.claim.text!r} ({check.status}{note}; cited {list(check.cited_ids) or 'nothing'})")
    else:
        print("Grounding: the answer makes no numeric claims")
    print(
        f"Tool calls: {len(episode.tool_calls)} (denied {sum(not c.allowed for c in episode.tool_calls)}, "
        f"look-ahead attempts {episode.lookahead_attempts}); proposals pending human approval: {len(episode.proposals)}"
    )
    fallback = "on" if episode.backend.get("fallback") else "off"
    print(f"Model: requested {episode.requested_model}; served {', '.join(episode.served_models) or 'n/a'} (fallback {fallback})")
    if episode.usage:
        tokens = {k: v for k, v in episode.usage.items() if k.endswith("_tokens") and isinstance(v, int)}
        print("Usage: " + ", ".join(f"{k}={v}" for k, v in sorted(tokens.items())))
    head = episode.audit_head
    print(f"Data: {source}")
    print(
        f"Audit: {audit_path} ({head.records if head else 0} records, head {head.head_hash[:16] if head else 'n/a'}) "
        f"{'verified' if verified else 'VERIFICATION FAILED'}"
    )


def _ask(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    try:
        as_of = parse_date(args.as_of, field="--as-of")
    except AgentToolError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    source = _source(args)
    backend = backend_factory(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    episode_id = f"{default_episode_id(args.question, as_of, backend.describe())}-{stamp}"
    audit_path = run_dir / f"{episode_id}.audit.jsonl"
    if audit_path.exists():  # pragma: no cover - a microsecond collision
        raise UsageError(f"{audit_path} already exists; retry")
    audit = AuditLog(audit_path)
    copilot = Copilot(backend, source=source, as_of=as_of, policy=policy, audit=audit, max_steps=args.max_steps)
    episode = copilot.run(args.question, episode_id=episode_id)
    head = audit.head
    verified = verify_chain(audit_path, expected_head=head).ok
    (run_dir / f"{episode_id}.head.json").write_text(json.dumps(head.to_dict()) + "\n", encoding="utf-8")
    (run_dir / f"{episode_id}.transcript.json").write_text(
        json.dumps(episode.to_dict(include_transcript=True), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    label = "synthetic panel (not real market prices)" if args.source == "synthetic" else "MarketDataStore (confirmed)"
    if args.json:
        print(json.dumps(episode.to_dict(), indent=2, sort_keys=True))
    else:
        _print_episode(episode, audit_path, verified, label)
    if episode.status == "backend_error":
        kind = (episode.error or {}).get("kind")
        if kind in CREDENTIAL_ERRORS:
            print(
                "The Claude backend needs credentials: set ANTHROPIC_API_KEY (or log in with `ant auth login`) and "
                "install the optional extra: pip install 'marketdata-agent[llm]'.",
                file=sys.stderr,
            )
        else:
            print(f"The model call failed ({kind}); see the audit log for details.", file=sys.stderr)
        return 2
    return 0 if episode.completed and verified else 1


def _backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.stem}.superseded-{stamp}{path.suffix}")
    path.rename(target)
    return target


def _bench(args: argparse.Namespace, backend_factory: BackendFactory | None) -> int:
    from .bench import DEFAULT_SEED, anthropic_agent, baseline_agent, generate_suite, run_agent
    from .bench.runner import AgentSpec

    seed = DEFAULT_SEED if args.seed is None else args.seed
    suite = generate_suite(seed)
    if args.tasks is not None:
        suite = suite.subset(args.tasks)
    if args.bench_command == "tasks":
        text = json.dumps(suite.to_dict(), indent=1)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        return 0
    if args.resume and args.overwrite:
        raise UsageError("--resume and --overwrite cannot be combined")
    out_dir = Path(args.out or f"runs/bench/{args.agent}")
    if args.agent == "anthropic":
        tools = args.tools
        prompt = args.prompt_version or (CLOSED_BOOK_PROMPT if tools == "none" else PROMPT_VERSION)
        arm: dict[str, Any] = {"enforce_clock": not args.no_clock, "prompt_version": prompt, "tools": tools}
        if backend_factory is not None:
            backend = backend_factory(args)
            agent = AgentSpec("anthropic", lambda task: backend, "injected backend", is_llm=True, **arm)
        else:
            record = None if args.replay else Path(args.record or out_dir / "recordings.jsonl")
            if record is not None and record.exists() and record.stat().st_size > 0 and not args.resume:
                if not args.overwrite:
                    raise UsageError(
                        f"{record} already exists; pass --resume to continue that run (recorded turns are reused) "
                        "or --overwrite to start a new recording (the old one is kept as a backup)"
                    )
                print(f"kept the previous recording as {_backup(record)}", file=sys.stderr)
            agent = anthropic_agent(_config(args), record_path=record, replay_path=args.replay, **arm)
    else:
        if args.no_clock or args.prompt_version or args.tools != "default":
            raise UsageError("--no-clock, --prompt-version and --tools apply to --agent anthropic only")
        agent = baseline_agent(args.agent)
    command = "marketdata-agent " + " ".join(shlex.quote(a) for a in sys.argv[1:]) if sys.argv[1:2] == ["bench"] else None
    result = run_agent(
        agent,
        suite,
        out_dir,
        max_steps=args.max_steps,
        command=command,
        overwrite=args.overwrite,
        resume=args.resume,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff,
    )
    overall = result.summary["metrics"]["overall"]
    accuracy = overall["accuracy"]
    grounding = overall["grounding_rate"]
    print(
        f"{agent.name}: accuracy {accuracy['k']}/{accuracy['n']}"
        + (f" ({100 * accuracy['rate']:.1f}%)" if accuracy["rate"] is not None else "")
        + f"; grounding {grounding['k']}/{grounding['n']}"
        + f"; look-ahead attempt episodes {overall['lookahead_attempt_episodes']['k']}"
        + f"; leak episodes {overall['leak_episodes']['k']}"
        + f"; audit {'verified' if result.summary['audit']['ok'] else 'FAILED'}"
    )
    print("Status: " + ", ".join(f"{k} {v}" for k, v in result.summary["status"].items()))
    print(f"Results: {out_dir / 'summary.md'}")
    if result.summary.get("banner"):
        print(result.summary["banner"])
    if not result.valid:
        return EXIT_INVALID_RUN
    return 0 if result.summary["audit"]["ok"] else 1


def _summary_anchor(path: str, agent: str | None) -> ChainHead:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "agents" in payload:
        if agent is None:
            raise UsageError(f"{path} is a combined summary; pass --agent with one of {sorted(payload['agents'])}")
        if agent not in payload["agents"]:
            raise UsageError(f"{path} has no agent {agent!r}; choose from {sorted(payload['agents'])}")
        audit = payload["agents"][agent]["audit"]
    else:
        audit = payload["audit"]
    if "head_hash" not in audit:
        raise UsageError(f"{path} records no audit head")
    return ChainHead(int(audit["records"]), str(audit["head_hash"]))


def _verify(args: argparse.Namespace) -> int:
    anchor = None
    if (args.records is None) != (args.head is None):
        raise UsageError("--records and --head must be given together")
    if args.summary is not None and args.records is not None:
        raise UsageError("use either --summary or --records/--head")
    if args.records is not None:
        anchor = ChainHead(int(args.records), str(args.head))
    elif args.summary is not None:
        anchor = _summary_anchor(args.summary, args.agent)
    result = verify_chain(args.path, expected_head=anchor)
    report: dict[str, Any] = {
        "ok": result.ok,
        "records": result.records,
        "head_hash": result.head_hash,
        "anchor": anchor.to_dict() if anchor else None,
        "error": result.error,
        "line": result.line,
    }
    print(json.dumps(report, indent=2))
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None, *, backend_factory: BackendFactory | None = None) -> int:
    """Entry point. ``backend_factory`` replaces the Claude backend (tests, other providers)."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "tools":
            payload = default_registry(Policy()).to_anthropic_tools()
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=False))
            return 0
        if args.command == "verify-audit":
            return _verify(args)
        if args.command == "ask":
            return _ask(args, backend_factory or _anthropic_backend)
        if args.command == "bench":
            return _bench(args, backend_factory)
    except (UsageError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:  # configuration values rejected by the package (model, effort, max tokens, ...)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2  # pragma: no cover - argparse enforces a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
