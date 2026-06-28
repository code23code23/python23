from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from qis_engine.data.models import ContractSpec


class DataProviderBase(ABC):
    """
    Abstract interface for market data. The only concrete implementation
    is CSVStubProvider. External adapters (Bloomberg, Quandl, etc.) can
    implement this interface without touching the engine.
    """

    @abstractmethod
    def get_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Return daily OHLCV+OI with schema: date, open, high, low, close, volume, open_interest."""

    @abstractmethod
    def get_options_chain(
        self,
        ticker: str,
        as_of: date,
        option_type: str,  # "call" | "put"
    ) -> pl.DataFrame:
        """Return option chain rows for a given date with schema: date, expiry, strike, price,
        implied_vol, delta, volume, open_interest."""

    @abstractmethod
    def get_contract_specs(self) -> list[ContractSpec]:
        """Return all available contract specifications."""

    @abstractmethod
    def load_all(self, store: "DataStore") -> None:  # type: ignore[name-defined]  # noqa: F821
        """Ingest all available data into the provided DataStore."""
