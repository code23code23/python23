from __future__ import annotations

from datetime import date

import polars as pl

from qis_engine.calculation.calculator import Calculator
from qis_engine.calculation.rebalancer import RebalancingEngine
from qis_engine.context.base import CalculationContext
from qis_engine.core.index import BCOMIndex
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine


class BCOMCalculator(Calculator):
    def __init__(self, context: CalculationContext) -> None:
        super().__init__(context)
        self._config = BCOMIndex.default_config()
        self._universe = CommodityUniverse.from_context(context)
        self._weighting = WeightingEngine(self._config.weighting_method)
        schedule_df = context.reference_data.get_roll_schedule(self._config.name)
        self._roll_calendars: dict[str, RollCalendar] = (
            {t: RollCalendar(t, schedule_df) for t in self._config.tickers}
            if not schedule_df.is_empty() else {}
        )
        self._rebalancer = RebalancingEngine(
            self._config.rebalance_schedule,
            announce_month=self._config.rebalance_announce_month,
        )

    def calculate(
        self,
        start: date,
        end: date,
        constraints: WeightingConstraints | None = None,
    ) -> pl.DataFrame:
        return self._run_calculation(
            self._config, self._universe, self._weighting,
            self._roll_calendars, self._rebalancer, start, end, constraints,
        )
