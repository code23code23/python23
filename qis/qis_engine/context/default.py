from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from qis_engine.context.base import (
    CalculationContext,
    MarketDataService,
    ReferenceDataService,
)
from qis_engine.data.models import ContractSpec


class DataStoreMarketDataService(MarketDataService):
    """MarketDataService backed by a DataStore (Parquet or ArcticDB)."""

    def __init__(self, store: "DataStore") -> None:  # noqa: F821
        self._store = store

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        return self._store.read_ohlcv(ticker, start, end)

    def get_options_chain(self, ticker: str, as_of: date, option_type: str) -> pl.DataFrame:
        return self._store.read_options_chain(ticker, as_of, option_type)


class CSVReferenceDataService(ReferenceDataService):
    """ReferenceDataService backed by a CSVStubProvider."""

    def __init__(self, provider: "CSVStubProvider") -> None:  # noqa: F821
        self._provider = provider

    def get_contract_specs(self) -> list[ContractSpec]:
        return self._provider.get_contract_specs()

    def get_roll_schedule(self, index_name: str) -> pl.DataFrame:  # noqa: ARG002
        # CSV stubs have a single roll schedule file shared across indices
        return self._provider.get_roll_schedule()

    def get_production_weights(self) -> pl.DataFrame:
        return self._provider.get_production_weights()


def build_default_context(
    stubs_dir: str | Path,
    store: "DataStore | None" = None,  # noqa: F821
) -> CalculationContext:
    """
    Build a CalculationContext from CSV stubs.

    If no store is provided, creates a temporary Parquet store and loads all
    stub data into it. Suitable for development and the container startup path.
    """
    from qis_engine.data.csv_stub import CSVStubProvider
    from qis_engine.data.store import DataStoreFactory

    stubs = Path(stubs_dir)
    provider = CSVStubProvider(stubs)

    if store is None:
        root = Path(tempfile.mkdtemp())
        store = DataStoreFactory.create_parquet(root)
        provider.load_all(store)
        roll_df = provider.get_roll_schedule()
        for index_name in ("BCOM", "GSCI"):
            store.write_roll_schedule(roll_df, index_name)

    return CalculationContext(
        market_data=DataStoreMarketDataService(store),
        reference_data=CSVReferenceDataService(provider),
    )
