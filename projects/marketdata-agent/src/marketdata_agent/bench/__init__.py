"""Benchmark: deterministic tasks, independent ground truth, baselines, scoring, runner and analysis."""

from __future__ import annotations

from .analysis import (
    all_repetitions_success,
    benjamini_hochberg,
    cluster_bootstrap_ci,
    holm,
    mcnemar_exact,
    threshold_decision,
)
from .baselines import (
    BASELINE_DESCRIPTIONS,
    BASELINE_NAMES,
    UNGROUNDED_MODES,
    baseline_backend,
    compose_answer,
    format_value,
    ungrounded_mode,
)
from .generator import (
    DEFAULT_COUNTS,
    DEFAULT_DATASET,
    DEFAULT_SEED,
    GENERATOR_VERSION,
    dataset_frame,
    evaluation_dataset,
    generate_evaluation_suites,
    generate_suite,
    item_key,
)
from .runner import HARNESS_BANNER, AgentSpec, RunResult, anthropic_agent, baseline_agent, run_agent, run_suite
from .scoring import TaskScore, aggregate, execution_claimed, judge, score_episode, summarize, wilson_interval
from .tasks import CATEGORIES, AnswerForm, Expected, Task, TaskSuite, ToolStep, load_suite

__all__ = [
    "AgentSpec",
    "AnswerForm",
    "BASELINE_DESCRIPTIONS",
    "BASELINE_NAMES",
    "CATEGORIES",
    "DEFAULT_COUNTS",
    "DEFAULT_DATASET",
    "DEFAULT_SEED",
    "Expected",
    "GENERATOR_VERSION",
    "HARNESS_BANNER",
    "RunResult",
    "Task",
    "TaskScore",
    "TaskSuite",
    "ToolStep",
    "UNGROUNDED_MODES",
    "aggregate",
    "all_repetitions_success",
    "anthropic_agent",
    "baseline_agent",
    "baseline_backend",
    "benjamini_hochberg",
    "cluster_bootstrap_ci",
    "compose_answer",
    "dataset_frame",
    "evaluation_dataset",
    "execution_claimed",
    "format_value",
    "generate_evaluation_suites",
    "generate_suite",
    "holm",
    "item_key",
    "judge",
    "load_suite",
    "mcnemar_exact",
    "run_agent",
    "run_suite",
    "score_episode",
    "summarize",
    "threshold_decision",
    "ungrounded_mode",
    "wilson_interval",
]
