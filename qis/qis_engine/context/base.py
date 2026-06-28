from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import polars as pl

from qis_engine.data.models import ContractSpec


class MarketDataService(ABC):
    """Provides time-series market data for futures and options."""

    @abstractmethod
    def get_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Return daily OHLCV+OI with schema: date, open, high, low, close, volume, open_interest."""

    @abstractmethod
    def get_options_chain(
        self, ticker: str, as_of: date, option_type: str
    ) -> pl.DataFrame:
        """Return option chain rows for a given date."""


class ReferenceDataService(ABC):
    """Provides static reference data: contract specs, roll schedules, production weights."""

    @abstractmethod
    def get_contract_specs(self) -> list[ContractSpec]:
        """Return all available contract specifications."""

    @abstractmethod
    def get_roll_schedule(self, index_name: str) -> pl.DataFrame:
        """Return the roll schedule DataFrame for the given index."""

    @abstractmethod
    def get_production_weights(self) -> pl.DataFrame:
        """Return the production weights reference table."""


@dataclass(frozen=True)
class CalculationContext:
    """
    Service provider passed into every calculate() call.

    Analogous to a Java ServletContext: application-scoped, holds handles to
    underlying services, shared safely across concurrent requests because all
    service implementations are stateless readers.
    """
    market_data: MarketDataService
    reference_data: ReferenceDataService
