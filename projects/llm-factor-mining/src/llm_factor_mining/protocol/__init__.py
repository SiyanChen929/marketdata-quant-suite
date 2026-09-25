"""Research protocol: chronological splits, trial ledger, sealed hold-out, contamination checks."""

from .contamination import (
    BehaviouralNovelty,
    CutoffComparison,
    Novelty,
    behavioural_novelty,
    library_novelty,
    library_signals,
    mean_novelty,
    split_ic_at_cutoff,
)
from .ledger import (
    GENESIS_HASH,
    LedgerIntegrityError,
    LedgerRecord,
    TrialLedger,
    verify_ledger_file,
)
from .seal import (
    COMPOSITE_NAME,
    FrozenFactor,
    FrozenFactorSet,
    RevealResult,
    SealError,
    SealedHoldout,
    composite_signal,
    oriented_signal,
)
from .registry import RegistryError, Study, StudyRegistry, StudySpec
from .splits import SplitError, Splits, chronological_splits, splits_from_dict, truncate_panel

__all__ = [
    "BehaviouralNovelty",
    "COMPOSITE_NAME",
    "CutoffComparison",
    "FrozenFactor",
    "FrozenFactorSet",
    "GENESIS_HASH",
    "LedgerIntegrityError",
    "LedgerRecord",
    "Novelty",
    "RegistryError",
    "RevealResult",
    "SealError",
    "SealedHoldout",
    "SplitError",
    "Splits",
    "Study",
    "StudyRegistry",
    "StudySpec",
    "TrialLedger",
    "behavioural_novelty",
    "chronological_splits",
    "composite_signal",
    "library_novelty",
    "library_signals",
    "mean_novelty",
    "oriented_signal",
    "split_ic_at_cutoff",
    "splits_from_dict",
    "truncate_panel",
    "verify_ledger_file",
]
