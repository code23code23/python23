from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from qis_engine.context.base import CalculationContext
from qis_engine.core.index import IndexConfig
from qis_engine.core.weighting import WeightingConstraints


class Calculator(ABC):
    """
    Interface for a per-index calculator.

    The engine creates a fresh instance for each run, bound to the
    application-scoped CalculationContext. Concrete calculators normally
    extend CalculatorBase, which supplies the shared daily loop.
    """

    def __init__(self, context: CalculationContext, config: IndexConfig) -> None:
        self._context = context
        self._config = config

    @property
    def config(self) -> IndexConfig:
        return self._config

    @abstractmethod
    def calculate(
        self,
        start: date,
        end: date,
        constraints: WeightingConstraints | None = None,
        collateral_rate_series: pl.Series | None = None,
    ) -> pl.DataFrame:
        """
        Return a daily DataFrame with columns:
            date, spot_return, roll_return, collateral_return,
            excess_return, total_return, spot_level, excess_level, total_level
        Plus one w_<ticker> column per constituent for weight history.
        """
