from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommissionModel:
    """Fixed basis-points commission applied on each roll turnover."""
    bps: float = 0.5

    def cost(self, notional: float) -> float:
        return notional * self.bps / 10_000.0


@dataclass(frozen=True)
class SlippageModel:
    """Half-spread slippage applied on each roll (bid-ask cost)."""
    half_spread_bps: float = 0.1

    def cost(self, notional: float) -> float:
        return notional * self.half_spread_bps / 10_000.0


@dataclass(frozen=True)
class TransactionCostModel:
    """Combined commission + slippage model."""
    commission: CommissionModel = CommissionModel()
    slippage: SlippageModel = SlippageModel()

    def total_cost(self, notional: float) -> float:
        return self.commission.cost(notional) + self.slippage.cost(notional)

    @classmethod
    def zero(cls) -> "TransactionCostModel":
        return cls(CommissionModel(0.0), SlippageModel(0.0))

    @classmethod
    def default(cls) -> "TransactionCostModel":
        return cls(CommissionModel(0.5), SlippageModel(0.1))
