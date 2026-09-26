"""Command-line entry point.

Subcommands:

* ``validate EXPR`` / ``library`` / ``grammar`` - offline DSL utilities;
* ``register-study`` - fix the formation / validation / test windows of a
  store data set in a study registry *before* anything is evaluated on it;
* ``evaluate EXPR`` - formation- or validation-window metrics of one
  expression.  On a synthetic market the windows are the fixed default split;
  on store data they come from the registered study, the panel is never
  loaded past the registered validation end, and every call is appended to
  the registry's exploration log.  The test window is not reachable here;
* ``search`` - one full protocol run (random, evolutionary or LLM proposer)
  into a required run directory.  On synthetic data it reveals the test
  window by default; on store data it only *commits* (records the commitment
  in the registry) and prints the commitment hash to publish;
* ``reveal`` - the separate, single reveal of a committed run: it checks the
  run's ledger, the data hash and the commitment hash you published, asks the
  registry (store data), and only then scores the test window;
* ``benchmark`` - the synthetic planted-alpha benchmark.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import pandas as pd

from .data import Panel, align_symbols, load_confirmed_panel
from .dsl import LIBRARY, canonical_string, complexity, describe_language, structural_hash, validate
from .dsl.validate import DSLLimits
from .evaluate.engine import Evaluator, ExpressionError
from .evaluate.metrics import MetricConfig, evaluate_signal
from .jsonutil import json_safe, write_json


PROPOSERS = ("random", "evolutionary", "llm")
DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _add_synthetic_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument("--data", choices=("synthetic", "store"), default="synthetic")
    group.add_argument("--seed", type=int, default=0, help="synthetic market seed")
    group.add_argument("--snr", type=float, default=0.10, help="synthetic signal-to-noise ratio")
    group.add_argument("--n-symbols", type=int, default=60)
    group.add_argument("--n-dates", type=int, default=750)
    group.add_argument("--horizon", type=int, default=1, help="synthetic data only (a study fixes its own)")
    group.add_argument("--lag", type=int, default=1, help="synthetic data only (a study fixes its own)")
    group.add_argument("--study", default=None, help="registered study name (--data store)")
    group.add_argument("--registry", default=None, help="study registry JSONL file (--data store)")


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("anthropic", "replay"), default=None, help="LLM backend")
    parser.add_argument("--model", default=None, help="Claude model id (default claude-opus-5)")
    parser.add_argument("--effort", default=None, choices=EFFORTS, help="effort level (default high)")
    parser.add_argument("--max-tokens", type=int, default=None, help="max output tokens (default 16000)")
    parser.add_argument(
        "--fallback", action="store_true", help="enable server-side model fallback (changes the model under test)"
    )
    parser.add_argument("--replay-file", default=None, help="JSONL response file for --backend replay")
    parser.add_argument("--record-file", default=None, help="fresh JSONL file recording anthropic responses")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-factor-mining",
        description="LLM-guided formulaic factor mining with a sealed-holdout protocol.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("validate", help="parse and statically validate an expression")
    check.add_argument("expression", help="factor expression, e.g. 'cs_rank(-ts_std(returns, 20))'")
    commands.add_parser("library", help="list the reference factor catalog as JSON lines")
    commands.add_parser("grammar", help="print the operator and terminal reference card")

    register = commands.add_parser("register-study", help="fix a store data set's windows in a registry")
    register.add_argument("--registry", required=True)
    register.add_argument("--study", required=True)
    register.add_argument("--symbols", required=True, help="comma-separated symbols")
    register.add_argument("--start", required=True)
    register.add_argument("--end", required=True)
    register.add_argument("--fractions", default="0.6,0.2,0.2", help="formation,validation,test fractions")
    register.add_argument("--lag", type=int, default=1)
    register.add_argument("--horizon", type=int, default=1)
    register.add_argument("--max-reveals", type=int, default=1, help="pre-registered number of test reveals")

    evaluate = commands.add_parser("evaluate", help="score one expression on formation or validation")
    evaluate.add_argument("expression")
    evaluate.add_argument("--window", choices=("formation", "validation"), default="formation")
    _add_synthetic_arguments(evaluate)

    search = commands.add_parser("search", help="run one propose/select/commit(/reveal) search")
    search.add_argument("--proposer", choices=PROPOSERS, required=True)
    _add_llm_arguments(search)
    search.add_argument("--proposer-seed", type=int, default=10_000)
    search.add_argument("--budget", type=int, default=200)
    search.add_argument("--batch-size", type=int, default=20)
    search.add_argument("--fdr-alpha", type=float, default=0.10)
    search.add_argument("--confirm-alpha", type=float, default=0.10)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--run-dir", required=True, help="fresh directory for the ledger and run artifacts")
    search.add_argument("--no-reveal", action="store_true", help="commit the frozen set but do not reveal test")
    _add_synthetic_arguments(search)

    reveal = commands.add_parser("reveal", help="reveal the test window of a committed run, once")
    reveal.add_argument("--run-dir", required=True)
    reveal.add_argument("--commitment", required=True, help="the commitment hash you published after search")
    reveal.add_argument("--registry", default=None, help="study registry (runs on store data)")

    bench = commands.add_parser("benchmark", help="synthetic planted-alpha benchmark")
    bench.add_argument("--seeds", default="0,1,2")
    bench.add_argument("--snr", default="0.05,0.15", help="comma-separated planted SNR levels (> 0)")
    bench.add_argument("--null-seeds", default=",".join(str(seed) for seed in range(10)), help="seeds of snr=0 markets")
    bench.add_argument("--n-symbols", type=int, default=60)
    bench.add_argument("--n-dates", type=int, default=750)
    bench.add_argument("--budget", type=int, default=200)
    bench.add_argument("--batch-size", type=int, default=20)
    bench.add_argument("--proposers", default="random,evolutionary,llm", help="arms: name or name@budget")
    _add_llm_arguments(bench)
    bench.add_argument("--out", required=True, help="directory for summary.json / summary.md")
    bench.add_argument("--runs-dir", required=True, help="directory for per-run ledgers (keep out of Git)")
    return parser


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _fractions(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 3:
        raise SystemExit("--fractions needs three comma-separated numbers")
    return values  # type: ignore[return-value]


def _csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def credentials_available() -> bool:
    """Best-effort check for Anthropic credentials (env vars or an ``ant`` profile)."""

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    return Path(os.path.expanduser("~/.config/anthropic")).is_dir()


def _explicit_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    """Backend identity built from --model/--effort/--max-tokens/--fallback, if any was given."""

    from .proposers.llm import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, AnthropicBackend

    given = [args.model, args.effort, args.max_tokens] + ([True] if args.fallback else [])
    if not any(value is not None for value in given):
        return None
    try:
        return AnthropicBackend(
            model=args.model or DEFAULT_MODEL,
            effort=args.effort or DEFAULT_EFFORT,
            max_tokens=args.max_tokens or DEFAULT_MAX_TOKENS,
            use_fallback=bool(args.fallback),
        ).identity()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_backend(args: argparse.Namespace, *, default_record: Path | None = None) -> Any:
    """The LLM backend named by ``--backend``; live responses are always recorded."""

    from .proposers.llm import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, AnthropicBackend, CachedBackend, ReplayBackend

    if args.backend == "replay":
        if not args.replay_file:
            raise SystemExit("--backend replay needs --replay-file")
        return ReplayBackend(args.replay_file, identity=_explicit_identity(args))
    if args.backend == "anthropic":
        if not credentials_available():
            raise SystemExit(
                "--backend anthropic needs credentials (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN); none found"
            )
        try:
            backend = AnthropicBackend(
                model=args.model or DEFAULT_MODEL,
                effort=args.effort or DEFAULT_EFFORT,
                max_tokens=args.max_tokens or DEFAULT_MAX_TOKENS,
                use_fallback=bool(args.fallback),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        record = args.record_file or default_record
        if record is None:
            raise SystemExit("--backend anthropic needs --record-file: every live response must be recorded")
        try:
            return CachedBackend(backend, record)
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
    raise SystemExit("--proposer llm needs --backend anthropic or --backend replay")


def _registry(path: str | None) -> Any:
    from .protocol.registry import StudyRegistry

    if not path:
        raise SystemExit("--data store needs --registry (the study registry file) and --study")
    return StudyRegistry(path)


def store_loader(spec: Any) -> Any:
    """``load(end)`` for a registered store study: confirmed bars, columns aligned to the study."""

    from quant_marketdata import MarketDataStore

    store = MarketDataStore()

    def load(end: pd.Timestamp | None) -> Panel:
        stop = pd.Timestamp(spec.end) if end is None else min(pd.Timestamp(end), pd.Timestamp(spec.end))
        return align_symbols(load_confirmed_panel(store, list(spec.symbols), spec.start, stop), spec.symbols)

    return load


def synthetic_market(meta: dict[str, Any]) -> Any:
    from .benchmark.synthetic import SyntheticMarketConfig, simulate_market

    return simulate_market(
        SyntheticMarketConfig(
            n_symbols=int(meta["n_symbols"]),
            n_dates=int(meta["n_dates"]),
            snr=float(meta["snr"]),
            seed=int(meta["seed"]),
            lag=int(meta["lag"]),
        )
    )


def _synthetic_meta(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": "synthetic",
        "seed": args.seed,
        "snr": args.snr,
        "n_symbols": args.n_symbols,
        "n_dates": args.n_dates,
        "lag": args.lag,
        "horizon": args.horizon,
    }


def load_panel(args: argparse.Namespace) -> Panel:
    """The synthetic market of ``args`` (store data go through a registered study)."""

    if args.data != "synthetic":
        raise SystemExit("store data are read through a registered study (--study, --registry)")
    return synthetic_market(_synthetic_meta(args)).panel


def _metric(lag: int, horizon: int) -> MetricConfig:
    return MetricConfig(horizon=horizon, lag=lag)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _cmd_register(args: argparse.Namespace) -> int:
    from .protocol.registry import RegistryError, StudySpec, study_symbols
    from .protocol.splits import chronological_splits
    from .search import SearchConfig, window_capacity_problems

    registry = _registry(args.registry)
    symbols = study_symbols(_csv(args.symbols))
    try:
        spec = StudySpec(
            name=args.study,
            source="store",
            symbols=symbols,
            start=str(pd.Timestamp(args.start).date()),
            end=str(pd.Timestamp(args.end).date()),
            fractions=_fractions(args.fractions),
            lag=args.lag,
            max_horizon=args.horizon,
            max_reveals=args.max_reveals,
        )
        # only the trading calendar is used; no statistic of the bars is computed or printed
        calendar = store_loader(spec)(None).dates
        splits = chronological_splits(calendar, fractions=spec.fractions, lag=spec.lag, max_horizon=spec.max_horizon)
        # refuse, before the windows are fixed for good, a study on which `search` could never select
        problems = window_capacity_problems(
            calendar,
            splits,
            SearchConfig(metric=_metric(spec.lag, spec.max_horizon)),
            n_symbols=len(spec.symbols),
        )
        if problems:
            raise SystemExit(
                "not registered: these windows could never select a factor (use a longer date range or more "
                "symbols): " + "; ".join(problems)
            )
        study = registry.register_study(spec, splits)
    except RegistryError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"study": spec.to_dict(), "splits": study.splits.to_dict(), "record": study.record_hash}, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .protocol.splits import chronological_splits, truncate_panel

    registry = None
    if args.data == "store":
        registry = _registry(args.registry)
        if not args.study:
            raise SystemExit("--data store needs --study")
        study = registry.require(args.study)
        splits, metric = study.splits, _metric(study.spec.lag, study.spec.max_horizon)
        window = splits.formation if args.window == "formation" else splits.validation
        visible = store_loader(study.spec)(window.end)  # never loaded past the registered window end
    else:
        panel = load_panel(args)
        metric = _metric(args.lag, args.horizon)
        splits = chronological_splits(panel.dates, fractions=DEFAULT_FRACTIONS, lag=metric.lag, max_horizon=metric.horizon)
        window = splits.formation if args.window == "formation" else splits.validation
        visible = truncate_panel(panel, window.end)
    try:
        signal = Evaluator(visible).evaluate(args.expression)
    except ExpressionError as exc:
        print(json.dumps({"ok": False, "codes": list(exc.codes), "error": str(exc)}, sort_keys=True))
        return 1
    node = validate(args.expression).node
    assert node is not None
    report = evaluate_signal(
        signal,
        visible.close,
        config=metric,
        window=window,
        expression=canonical_string(node),
        expression_hash=structural_hash(node),
        panel_sha256=visible.content_sha256,
    ).report
    payload: dict[str, Any] = {"ok": True, "splits": splits.to_dict(), "report": report.to_dict()}
    if registry is not None:
        payload["n_explorations"] = registry.record_exploration(
            args.study,
            command="evaluate",
            window=args.window,
            canonical=canonical_string(node),
            expression_hash=structural_hash(node),
            panel_sha256=visible.content_sha256,
        )
    print(json.dumps(json_safe(payload), sort_keys=True))
    return 0


def _make_search_proposer(args: argparse.Namespace, symbols: Sequence[str], limits: DSLLimits) -> Any:
    from .proposers.evolutionary import EvolutionaryProposer
    from .proposers.llm import LLMProposer
    from .proposers.random_grammar import RandomGrammarProposer

    if args.proposer == "random":
        return RandomGrammarProposer(args.proposer_seed, limits=limits)
    if args.proposer == "evolutionary":
        return EvolutionaryProposer(args.proposer_seed, limits=limits)
    backend = build_backend(args, default_record=Path(args.run_dir) / "llm_responses.jsonl")
    return LLMProposer(backend, replicate=args.proposer_seed, forbidden_tokens=symbols)


def _cmd_search(args: argparse.Namespace) -> int:
    from .protocol.splits import chronological_splits
    from .search import SearchConfig, run_search

    run_dir = Path(args.run_dir)
    if (run_dir / "ledger.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a ledger; use a fresh run directory")
    registry = None
    reveal = not args.no_reveal
    if args.data == "store":
        registry = _registry(args.registry)
        if not args.study:
            raise SystemExit("--data store needs --study")
        study = registry.require(args.study)
        splits, metric = study.splits, _metric(study.spec.lag, study.spec.max_horizon)
        panel: Any = store_loader(study.spec)
        symbols: Sequence[str] = study.spec.symbols
        meta: dict[str, Any] = {"data": "store", "study": args.study, "registry": str(Path(args.registry).resolve())}
        if reveal:
            print(
                "note: store data are only committed by `search`; reveal them with `reveal` after publishing "
                "the commitment hash",
                file=sys.stderr,
            )
        reveal = False
    else:
        market_panel = load_panel(args)
        metric = _metric(args.lag, args.horizon)
        splits = chronological_splits(
            market_panel.dates, fractions=DEFAULT_FRACTIONS, lag=metric.lag, max_horizon=metric.horizon
        )
        panel, symbols, meta = market_panel, market_panel.symbols, _synthetic_meta(args)
    limits = DSLLimits()
    proposer = _make_search_proposer(args, symbols, limits)
    config = SearchConfig(
        budget=args.budget,
        batch_size=args.batch_size,
        metric=metric,
        limits=limits,
        fdr_alpha=args.fdr_alpha,
        confirm_alpha=args.confirm_alpha,
        top_k=args.top_k,
    )
    result = run_search(proposer, panel, splits, config, run_dir=run_dir, reveal=reveal, run_metadata={"cli": meta})
    payload: dict[str, Any] = {"summary": result.summary(), "run_dir": str(run_dir)}
    if registry is not None and result.commitment is not None:
        registry.record_commit(
            args.study,
            run=str(run_dir.resolve()),
            commitment=result.commitment,
            data_sha256=result.data_sha256,
            test_window=splits.to_dict()["test"],
            ledger_head=result.ledger_head,
        )
        payload["next_step"] = (
            "publish the commitment hash outside this repository (for example in a pushed commit or a "
            "public timestamp), then run: llm-factor-mining reveal --run-dir "
            f"{run_dir} --commitment {result.commitment} --registry {args.registry}"
        )
    if registry is not None:
        # exploration evaluations are forking paths outside this run's ledger: report their count with it
        payload["registry"] = _record_registry_summary(run_dir, registry, args.study, "registry_at_commit")
    for warning in result.warnings():
        print(f"warning: {warning}", file=sys.stderr)
    if result.reveal is not None:
        payload["test"] = {
            "composite": result.reveal.payload.get("composite"),
            "factors": [
                {"expression": row["expression"], "orientation": row["orientation"], "report": row["report"]}
                for row in result.reveal.payload.get("factors", [])
            ],
        }
    print(json.dumps(json_safe(payload), sort_keys=True))
    return 0 if not result.aborted else 3


def _record_registry_summary(run_dir: Path, registry: Any, study: str, key: str) -> dict[str, Any]:
    """Add the study's registry counts (explorations, commits, reveals) to the run's provenance.json."""

    summary = json_safe(registry.summary(study))
    path = run_dir / "provenance.json"
    if path.exists():
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance[key] = summary
        write_json(path, provenance)
    return summary


def _cmd_reveal(args: argparse.Namespace) -> int:
    from .protocol.ledger import TrialLedger, verify_ledger_file
    from .protocol.seal import FrozenFactor, FrozenFactorSet, SealError, SealedHoldout
    from .protocol.splits import splits_from_dict
    from .search import SearchConfig

    run_dir = Path(args.run_dir)
    try:
        record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        selected = json.loads((run_dir / "selected.json").read_text(encoding="utf-8"))
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{run_dir} is not a committed run directory: {exc}") from exc
    if args.commitment != selected["commitment"]:
        raise SystemExit("the commitment hash does not match the run's commitment; nothing was revealed")
    verify_ledger_file(run_dir / "ledger.jsonl", expected_head=provenance["ledger_head"])
    config = SearchConfig.from_dict(record["config"])
    splits = splits_from_dict(record["splits"])
    meta = record["run_metadata"]["cli"]
    authorize = None
    registry = None
    if meta["data"] == "store":
        registry = _registry(args.registry or meta.get("registry"))
        panel = store_loader(registry.require(meta["study"]).spec)(None)
        authorize = registry.reveal_authorizer(meta["study"], run=str(run_dir.resolve()))
    else:
        panel = synthetic_market(meta).panel
    if panel.content_sha256 != selected["data_sha256"]:
        raise SystemExit("the reloaded data do not match the committed data hash; nothing was revealed")
    frozen = FrozenFactorSet(
        tuple(
            FrozenFactor(item["expression"], item["expression_hash"], int(item["orientation"]))
            for item in selected["frozen"]["factors"]
        ),
        label=selected["frozen"]["label"],
    )
    ledger = TrialLedger(run_dir / "ledger.jsonl", timestamps=config.record_timestamps)
    holdout = SealedHoldout(panel, splits.test, config.metric, ledger, limits=config.limits)
    if holdout.commitment != args.commitment:
        raise SystemExit("the run's ledger holds no matching commitment; nothing was revealed")
    try:
        revealed = holdout.reveal(frozen, authorize=authorize)
    except SealError as exc:
        # the refusal is now in the ledger; keep the run's recorded head in step with it
        provenance["ledger_head"] = ledger.head_hash
        write_json(run_dir / "provenance.json", provenance)
        raise SystemExit(f"reveal refused: {exc}") from exc
    write_json(run_dir / "reveal.json", revealed.payload)
    provenance["reveal_ledger_head"] = ledger.head_hash
    provenance["ledger_head"] = ledger.head_hash
    output: dict[str, Any] = {"reveal": revealed.payload, "ledger_head": ledger.head_hash}
    if registry is not None:
        provenance["registry_at_reveal"] = output["registry"] = json_safe(registry.summary(meta["study"]))
    write_json(run_dir / "provenance.json", provenance)
    print(json.dumps(json_safe(output), sort_keys=True))
    return 0


def benchmark_llm_factory(args: argparse.Namespace) -> tuple[Any, str]:
    """``(factory or None, status text)`` for the benchmark's LLM arm."""

    from .proposers.llm import LLMProposer

    if args.backend is None:
        return None, "not run: no --backend given (no credentials or replay cache supplied)"
    if args.backend == "anthropic" and not credentials_available():
        return None, "not run: no credentials (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN unset)"
    if args.backend == "replay" and not (args.replay_file and Path(args.replay_file).exists()):
        return None, "not run: replay file missing"
    backend = build_backend(args, default_record=Path(args.out) / "llm_responses.jsonl")

    def factory(seed: int, *, run_tag: str = "", forbidden_tokens: Sequence[str] = ()) -> LLMProposer:
        return LLMProposer(backend, replicate=seed, run_tag=run_tag, forbidden_tokens=forbidden_tokens)

    return factory, "run"


def benchmark_command_line(argv: Sequence[str]) -> str:
    return "python -m llm_factor_mining.cli " + " ".join(shlex.quote(part) for part in argv)


def run_benchmark_from_args(args: argparse.Namespace, *, command: str) -> tuple[Path, Path, dict[str, Any], Path]:
    """Run the benchmark; returns ``(summary.json, summary.md, summary, per-run ledger directory)``.

    An explicitly requested LLM backend that cannot run stops the command
    before anything is run or written (it would otherwise re-run every
    baseline and overwrite the summary without the LLM arm).  Per-run ledgers
    go to a fresh subdirectory of ``--runs-dir`` for every invocation.
    """

    from .benchmark.report import write_summary
    from .benchmark.runner import BenchmarkConfig, fresh_runs_dir, run_benchmark

    if args.backend == "anthropic" and not credentials_available():
        raise SystemExit(
            "--backend anthropic needs credentials (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN); none found. "
            "Nothing was run and no summary was written."
        )
    if args.backend == "replay" and not (args.replay_file and Path(args.replay_file).exists()):
        raise SystemExit("--backend replay needs an existing --replay-file. Nothing was run and no summary was written.")
    arms = _csv(args.proposers)
    if args.backend is not None and not any(arm.partition("@")[0] == "llm" for arm in arms):
        arms.append("llm")
    try:
        config = BenchmarkConfig(
            seeds=tuple(int(part) for part in _csv(args.seeds)),
            snr_levels=tuple(float(part) for part in _csv(args.snr)),
            null_seeds=tuple(int(part) for part in _csv(args.null_seeds)),
            n_symbols=args.n_symbols,
            n_dates=args.n_dates,
            budget=args.budget,
            batch_size=args.batch_size,
            proposers=tuple(arms),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    factory, status = benchmark_llm_factory(args)
    runs_path = fresh_runs_dir(args.runs_dir)
    print(f"per-run ledgers: {runs_path}", file=sys.stderr)
    summary = run_benchmark(
        config,
        llm_factory=factory,
        llm_status=status,
        runs_dir=runs_path,
        progress=lambda message: print(message, file=sys.stderr),
    )
    json_path, md_path = write_summary(summary, args.out, command=command)
    return json_path, md_path, summary, runs_path


def _cmd_benchmark(args: argparse.Namespace, argv: Sequence[str]) -> int:
    json_path, md_path, _, runs_path = run_benchmark_from_args(args, command=benchmark_command_line(argv))
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "runs_dir": str(runs_path)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from quant_marketdata import QuantMarketDataError

    from .data import PanelError
    from .protocol.registry import RegistryError
    from .protocol.splits import SplitError

    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    try:
        return _dispatch(args, raw)
    except RegistryError as exc:
        raise SystemExit(str(exc)) from exc
    except (SplitError, PanelError, FileNotFoundError, FileExistsError, QuantMarketDataError) as exc:
        # usage and data problems end with one line, not a traceback
        raise SystemExit(f"error: {exc}") from exc


def _dispatch(args: argparse.Namespace, raw: Sequence[str]) -> int:
    if args.command == "validate":
        result = validate(args.expression)
        payload = result.to_dict()
        if result.ok and result.node is not None:
            payload["canonical"] = canonical_string(result.node)
            payload["hash"] = structural_hash(result.node)
            payload["complexity"] = complexity(result.node)
        print(json.dumps(payload, sort_keys=True))
        return 0 if result.ok else 1
    if args.command == "library":
        for factor in LIBRARY:
            print(
                json.dumps(
                    {
                        "name": factor.name,
                        "family": factor.family,
                        "source": factor.source,
                        "expression": factor.expression,
                        "canonical": canonical_string(factor.node),
                        "hash": factor.hash,
                    },
                    sort_keys=True,
                )
            )
        return 0
    if args.command == "grammar":
        print(describe_language())
        return 0
    if args.command == "register-study":
        return _cmd_register(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "reveal":
        return _cmd_reveal(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args, raw)
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
