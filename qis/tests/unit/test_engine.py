"""Tests for calculation/calculator.py — return math driven by in-memory stub data."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qis_engine.calculation.strategy.configured import ConfiguredCalculator
from qis_engine.calculation.rebalancer import RebalancingEngine
from qis_engine.core.index import BCOMIndex, IndexConfig, RebalanceSchedule, ReturnType
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingEngine
from qis_engine.data.models import ContractSpec, Exchange, Sector, Settlement
from qis_engine.data.store import DataStoreFactory
from qis_engine.utils.calendars import nth_business_day_of_month

STUBS_DIR = Path(__file__).parent.parent / "stubs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(ticker: str) -> ContractSpec:
    return ContractSpec(
        ticker=ticker, name=ticker, sector=Sector.ENERGY,
        exchange=Exchange.NYMEX, lot_size=1000, tick_size=0.01,
        currency="USD", settlement=Settlement.CASH,
    )


def _flat_store(tmp_path: Path, tickers: list[str], price: float = 100.0) -> "DataStore":
    """Store with flat (constant) price series — spot returns should be 0."""
    from qis_engine.utils.calendars import business_days_series
    store = DataStoreFactory.create_parquet(tmp_path)
    dates = business_days_series(date(2022, 1, 3), date(2022, 3, 31))
    for t in tickers:
        store.write_ohlcv(pl.DataFrame({
            "date": dates,
            "open": [price] * len(dates),
            "high": [price] * len(dates),
            "low": [price] * len(dates),
            "close": [price] * len(dates),
            "volume": [100_000] * len(dates),
            "open_interest": [200_000] * len(dates),
        }), t)
    return store


def _trending_store(tmp_path: Path, ticker: str, daily_return: float = 0.001) -> "DataStore":
    """Store with a deterministic daily return series."""
    from qis_engine.utils.calendars import business_days_series
    store = DataStoreFactory.create_parquet(tmp_path)
    dates = business_days_series(date(2022, 1, 3), date(2022, 3, 31))
    prices = [100.0 * (1 + daily_return) ** i for i in range(len(dates))]
    store.write_ohlcv(pl.DataFrame({
        "date": dates, "open": prices, "high": prices,
        "low": prices, "close": prices,
        "volume": [100_000] * len(dates),
        "open_interest": [200_000] * len(dates),
    }), ticker)
    return store


def _make_context(store):
    from qis_engine.context.base import CalculationContext
    from qis_engine.context.default import DataStoreMarketDataService, CSVReferenceDataService
    from qis_engine.data.csv_stub import CSVStubProvider
    return CalculationContext(
        market_data=DataStoreMarketDataService(store),
        reference_data=CSVReferenceDataService(CSVStubProvider(STUBS_DIR)),
    )


def _make_engine(
    store, tickers: list[str], method: str = "equal"
) -> ConfiguredCalculator:
    config = IndexConfig(
        name="TEST",
        base_date=date(2022, 1, 3),
        tickers=tickers,
        base_level=100.0,
        rebalance_schedule=RebalanceSchedule.ANNUAL,
        weighting_method=method,
    )
    context = _make_context(store)
    universe = CommodityUniverse([_spec(t) for t in tickers])
    weighting = WeightingEngine(method)
    return ConfiguredCalculator(context, config, universe, weighting, roll_calendars={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlatPrices:
    def test_spot_return_zero_on_flat_prices(self, tmp_path):
        store = _flat_store(tmp_path, ["A", "B"])
        engine = _make_engine(store, ["A", "B"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        non_zero = result.filter(pl.col("spot_return").abs() > 1e-10)
        assert len(non_zero) == 0, f"Expected all spot returns ≈ 0 for flat prices"

    def test_level_constant_on_flat_prices(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        levels = result["excess_level"].to_list()
        assert all(abs(l - 100.0) < 1e-6 for l in levels), "Level should stay at 100 for flat prices"

    def test_weights_sum_to_one_every_day(self, tmp_path):
        store = _flat_store(tmp_path, ["A", "B", "C"])
        engine = _make_engine(store, ["A", "B", "C"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        weight_cols = [c for c in result.columns if c.startswith("w_")]
        assert len(weight_cols) == 3
        row_sums = result.select(pl.sum_horizontal(weight_cols).alias("ws"))["ws"]
        assert (row_sums.abs() - 1.0).abs().max() < 1e-9


class TestTrendingPrices:
    def test_positive_return_produces_growing_level(self, tmp_path):
        store = _trending_store(tmp_path, "X", daily_return=0.001)
        engine = _make_engine(store, ["X"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        assert float(result["excess_level"][-1]) > 100.0

    def test_spot_return_matches_daily_return(self, tmp_path):
        dr = 0.005
        store = _trending_store(tmp_path, "X", daily_return=dr)
        engine = _make_engine(store, ["X"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        spot_returns = result.filter(pl.col("spot_return") != 0.0)["spot_return"]
        assert all(abs(r - dr) < 1e-6 for r in spot_returns.to_list()), (
            f"Expected daily spot returns ≈ {dr}"
        )

    def test_cumulative_level_compounds_correctly(self, tmp_path):
        dr = 0.001
        store = _trending_store(tmp_path, "X", daily_return=dr)
        engine = _make_engine(store, ["X"])
        result = engine.calculate(date(2022, 1, 3), date(2022, 3, 31))
        n = len(result) - 1
        expected_level = 100.0 * (1 + dr) ** n
        actual_level = float(result["excess_level"][-1])
        assert abs(actual_level - expected_level) / expected_level < 0.01


class TestReturnDecomposition:
    def test_excess_return_equals_spot_plus_roll(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        diff = (result["excess_return"] - result["spot_return"] - result["roll_return"]).abs()
        assert diff.max() < 1e-12

    def test_total_return_equals_excess_plus_collateral(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        diff = (
            result["total_return"] - result["excess_return"] - result["collateral_return"]
        ).abs()
        assert diff.max() < 1e-12

    def test_collateral_zero_when_rate_zero(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        assert result["collateral_return"].abs().max() < 1e-12

    def test_output_columns_present(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        required = {"date", "spot_return", "roll_return", "collateral_return",
                    "excess_return", "total_return", "spot_level",
                    "excess_level", "total_level"}
        assert required.issubset(set(result.columns))

    def test_levels_positive_throughout(self, tmp_path):
        store = _flat_store(tmp_path, ["A", "B"])
        engine = _make_engine(store, ["A", "B"])
        result = engine.calculate(date(2022, 1, 4), date(2022, 3, 31))
        assert (result["spot_level"] > 0).all()
        assert (result["excess_level"] > 0).all()
        assert (result["total_level"] > 0).all()


class TestRebalancer:
    def test_annual_rebalance_fires_in_february(self):
        rb = RebalancingEngine(RebalanceSchedule.ANNUAL, announce_month=1)
        first_feb_bday = nth_business_day_of_month(2022, 2, 1)
        assert rb.is_rebalance_day(first_feb_bday)

    def test_monthly_rebalance_fires_each_month(self):
        rb = RebalancingEngine(RebalanceSchedule.MONTHLY)
        for month in range(1, 13):
            first_bday = nth_business_day_of_month(2022, month, 1)
            assert rb.is_rebalance_day(first_bday)

    def test_quarterly_rebalance_fires_jan_apr_jul_oct(self):
        rb = RebalancingEngine(RebalanceSchedule.QUARTERLY)
        for month in (1, 4, 7, 10):
            first_bday = nth_business_day_of_month(2022, month, 1)
            assert rb.is_rebalance_day(first_bday)
        for month in (2, 3, 5, 6, 8, 9, 11, 12):
            first_bday = nth_business_day_of_month(2022, month, 1)
            assert not rb.is_rebalance_day(first_bday)

    def test_no_empty_output(self, tmp_path):
        store = _flat_store(tmp_path, ["A"])
        engine = _make_engine(store, ["A"])
        result = engine.calculate(date(2022, 1, 3), date(2022, 1, 31))
        assert len(result) > 0
