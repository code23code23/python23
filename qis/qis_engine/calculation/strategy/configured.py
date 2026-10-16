from __future__ import annotations

from datetime import date

from qis_engine.calculation.calculator_base import CalculatorBase
from qis_engine.context.base import CalculationContext
from qis_engine.core.index import IndexConfig
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine
from qis_engine.data.models import ContractSpec


class ConfiguredCalculator(CalculatorBase):
    """
    Calculator driven entirely by an IndexConfig (weighting from config.weighting_method).

    For ad-hoc or test indices that have no dedicated strategy class; components
    can be injected instead of being built from the context.
    """

    def __init__(
        self,
        context: CalculationContext,
        config: IndexConfig,
        universe: CommodityUniverse | None = None,
        weighting: WeightingEngine | None = None,
        roll_calendars: dict[str, RollCalendar] | None = None,
    ) -> None:
        super().__init__(context, config, universe, roll_calendars)
        self._weighting = weighting or WeightingEngine(config.weighting_method)

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
