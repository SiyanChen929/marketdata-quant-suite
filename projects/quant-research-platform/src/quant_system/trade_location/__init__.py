"""Trade-location layer for separating alpha rank from entry quality."""

from quant_system.trade_location.entry_quality import apply_trade_location_to_targets, attach_trade_location_scores

__all__ = ["apply_trade_location_to_targets", "attach_trade_location_scores"]
