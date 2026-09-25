"""Order proposals: the only trading-related action the agent can take.

A proposal is an inert record.  This package contains no broker client and no
method that transmits, routes, or executes a proposal; confirming one is a
human decision taken outside the agent, and even then execution may occur no
earlier than the session after ``as_of`` (the suite's ``t+1`` rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from .provenance import sha256_json


PENDING_HUMAN_APPROVAL = "pending_human_approval"
Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class OrderProposal:
    """An order idea awaiting explicit human review; never executed by the agent."""

    proposal_id: str
    sequence: int
    symbol: str
    side: Side
    quantity: int
    rationale: str
    as_of: date
    reference_date: date
    reference_close: float
    status: str = PENDING_HUMAN_APPROVAL
    requires_human_confirmation: bool = True
    executed: bool = False

    @property
    def reference_notional(self) -> float:
        """Quantity times the last confirmed close; an estimate, not a fill price."""

        return float(self.quantity) * float(self.reference_close)

    @property
    def earliest_execution(self) -> str:
        """Human-readable execution-lag rule for the proposal."""

        return f"no earlier than the first session after {self.as_of.isoformat()} (t+1 rule)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "sequence": self.sequence,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "as_of": self.as_of.isoformat(),
            "reference_date": self.reference_date.isoformat(),
            "reference_close": self.reference_close,
            "reference_notional": self.reference_notional,
            "earliest_execution": self.earliest_execution,
            "status": self.status,
            "requires_human_confirmation": self.requires_human_confirmation,
            "executed": self.executed,
        }


class ProposalBook:
    """Append-only, in-memory list of proposals created during one episode.

    There is deliberately no ``execute``/``submit`` method: the book can only
    grow, and its contents are exported for human review.
    """

    def __init__(self) -> None:
        self._proposals: list[OrderProposal] = []

    def propose(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: int,
        rationale: str,
        as_of: date,
        reference_date: date,
        reference_close: float,
    ) -> OrderProposal:
        """Record a new pending proposal and return it."""

        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if int(quantity) < 1:
            raise ValueError("quantity must be a positive integer")
        sequence = len(self._proposals) + 1
        proposal_id = "p-" + sha256_json(
            {
                "sequence": sequence,
                "symbol": symbol,
                "side": side,
                "quantity": int(quantity),
                "rationale": rationale,
                "as_of": as_of.isoformat(),
                "reference_date": reference_date.isoformat(),
                "reference_close": float(reference_close),
            }
        )[:12]
        proposal = OrderProposal(
            proposal_id=proposal_id,
            sequence=sequence,
            symbol=symbol,
            side=side,
            quantity=int(quantity),
            rationale=rationale,
            as_of=as_of,
            reference_date=reference_date,
            reference_close=float(reference_close),
        )
        self._proposals.append(proposal)
        return proposal

    @property
    def proposals(self) -> tuple[OrderProposal, ...]:
        return tuple(self._proposals)

    def pending(self) -> tuple[OrderProposal, ...]:
        """Return proposals still awaiting human approval (all of them, by design)."""

        return tuple(p for p in self._proposals if p.status == PENDING_HUMAN_APPROVAL)

    def __len__(self) -> int:
        return len(self._proposals)
