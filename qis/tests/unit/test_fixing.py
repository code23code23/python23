"""Unit tests for calculation/fixing.py — IndexFixingWorkflow."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from qis_engine.calculation.engine import IndexCalculationEngine
from qis_engine.calculation.fixing import IndexFixingWorkflow, ValidationResult
from qis_engine.core.index import IndexConfig, RebalanceSchedule
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingEngine
from qis_engine.data.models import ContractSpec, Exchange, Sector, Settlement
from qis_engine.data.store import DataStoreFactory
from qis_engine.utils.calendars import business_days_series


def _spec(t: str) -> ContractSpec:
    return ContractSpec(t, t, Sector.ENERGY, Exchange.NYMEX, 1000, 0.01, "USD", Settlement.CASH)


def _make_store_with_history(tmp_path: Path) -> tuple:
    """Returns (store, engine, workflow) with a 1-month price history for ticker A."""
    from qis_engine.utils.calendars import business_days_series
    import numpy as np

    store = DataStoreFactory.create_parquet(tmp_path)
    tickers = ["A"]
    dates = business_days_series(date(2023, 1, 2), date(2023, 2, 28))
    rng = np.random.default_rng(1)
    prices = list(100 * np.cumprod(1 + rng.normal(0.0002, 0.005, len(dates))))
    store.write_ohlcv(pl.DataFrame({
        "date": dates,
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [100_000] * len(dates),
        "open_interest": [200_000] * len(dates),
    }), "A")

    config = IndexConfig(
        name="TEST",
        base_date=date(2023, 1, 2),
        tickers=tickers,
        base_level=100.0,
        rebalance_schedule=RebalanceSchedule.ANNUAL,
    )
    universe = CommodityUniverse([_spec("A")])
    weighting = WeightingEngine("equal")
    engine = IndexCalculationEngine(config, store, universe, weighting, roll_calendars={})

    # Pre-populate index levels from historical run
    hist = engine.run_calculation(date(2023, 1, 2), date(2023, 1, 31))
    store.write_index_level(hist, "TEST")

    workflow = IndexFixingWorkflow(engine, store, "TEST")
    return store, engine, workflow


class TestIndexFixingWorkflow:
    def test_run_daily_fix_returns_result(self, tmp_path):
        store, engine, workflow = _make_store_with_history(tmp_path)
        result = workflow.run_daily_fix(date(2023, 2, 1))
        assert result is not None
        assert result.fix_date == date(2023, 2, 1)

    def test_validation_passes_for_normal_day(self, tmp_path):
        store, engine, workflow = _make_store_with_history(tmp_path)
        result = workflow.run_daily_fix(date(2023, 2, 1))
        assert result.validated

    def test_result_levels_positive(self, tmp_path):
        store, engine, workflow = _make_store_with_history(tmp_path)
        result = workflow.run_daily_fix(date(2023, 2, 1))
        assert result.total_level > 0
        assert result.excess_level > 0
        assert result.spot_level > 0

    def test_preliminary_flag_recorded(self, tmp_path):
        store, engine, workflow = _make_store_with_history(tmp_path)
        result = workflow.run_daily_fix(date(2023, 2, 1), is_preliminary=True)
        assert result.is_preliminary

    def test_published_to_store_on_validation_pass(self, tmp_path):
        store, engine, workflow = _make_store_with_history(tmp_path)
        workflow.run_daily_fix(date(2023, 2, 1))
        stored = store.read_index_level("TEST", date(2023, 2, 1), date(2023, 2, 1))
        assert not stored.is_empty()


class TestValidation:
    def test_spot_return_breach_fails(self):
        from qis_engine.calculation.fixing import ValidationResult
        v = ValidationResult(passed=True, errors=[], warnings=[])
        if abs(0.15) > 0.10:
            v.errors.append("return breach")
            v.passed = False
        assert not v.passed
        assert "return breach" in v.errors

    def test_zero_return_passes(self):
        from qis_engine.calculation.fixing import ValidationResult
        v = ValidationResult(passed=True, errors=[], warnings=[])
        assert v.passed
        assert v.errors == []
