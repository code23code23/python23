from __future__ import annotations

from datetime import date

from qis_engine.calculation.calculator_base import CalculatorBase
from qis_engine.context.base import CalculationContext
from qis_engine.core.index import GSCIIndex
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine
from qis_engine.data.models import ContractSpec


class GSCICalculator(CalculatorBase):
    """GSCI: world-production weights; roll per the GSCI roll calendar."""

    def __init__(self, context: CalculationContext) -> None:
        super().__init__(context, GSCIIndex.default_config())
        self._weighting = WeightingEngine("production")

    def _compute_weights(
        self,
        eligible: list[ContractSpec],
        as_of: date,
        constraints: WeightingConstraints,
    ) -> dict[str, float]:
        return self._weights_from_engine(self._weighting, eligible, as_of, constraints)

    def _compute_roll_return(
        self,
        dt: date,
        weights: dict[str, float],
        today_prices: dict[str, float],
    ) -> float:
        return self._scheduled_roll_return(dt, weights, today_prices)
