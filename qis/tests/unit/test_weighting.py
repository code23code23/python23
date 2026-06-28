"""Tests for core/weighting.py — all four weighting methods."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from qis_engine.core.weighting import (
    WeightingConstraints,
    WeightingEngine,
    _apply_constraints,
)
from qis_engine.data.models import ContractSpec, Exchange, Sector, Settlement
from qis_engine.data.store import DataStoreFactory


def _specs(tickers=("CL", "GC", "ZC")) -> list[ContractSpec]:
    return [
        ContractSpec(
            ticker=t, name=t, sector=Sector.ENERGY, exchange=Exchange.NYMEX,
            lot_size=1000, tick_size=0.01, currency="USD", settlement=Settlement.CASH,
        )
        for t in tickers
    ]


def _stub_store(tmp_path: Path, tickers=("CL", "GC", "ZC")) -> "DataStore":
    from qis_engine.utils.calendars import business_days_series
    store = DataStoreFactory.create_parquet(tmp_path)
    rng_state = 42
    import polars as pl
    import numpy as np
    rng = np.random.default_rng(rng_state)
    dates = business_days_series(date(2022, 1, 3), date(2023, 1, 3))
    for ticker in tickers:
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
        df = pl.DataFrame({
            "date": dates,
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.integers(50_000, 500_000, len(dates)).tolist(),
            "open_interest": rng.integers(100_000, 400_000, len(dates)).tolist(),
        })
        store.write_ohlcv(df, ticker)
    return store


class TestEqualWeight:
    def test_equal_weights_sum_to_one(self, tmp_path):
        store = _stub_store(tmp_path)
        engine = WeightingEngine("equal")
        w = engine.compute_weights(_specs(), date(2023, 1, 3), store)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_equal_weights_are_uniform(self, tmp_path):
        store = _stub_store(tmp_path)
        engine = WeightingEngine("equal")
        w = engine.compute_weights(_specs(), date(2023, 1, 3), store)
        vals = w.to_list()
        assert all(abs(v - vals[0]) < 1e-9 for v in vals)

    def test_single_ticker(self, tmp_path):
        store = _stub_store(tmp_path, tickers=("CL",))
        engine = WeightingEngine("equal")
        w = engine.compute_weights(_specs(("CL",)), date(2023, 1, 3), store)
        assert abs(w[0] - 1.0) < 1e-9


class TestRiskParityWeight:
    def test_weights_sum_to_one(self, tmp_path):
        store = _stub_store(tmp_path)
        engine = WeightingEngine("risk_parity")
        w = engine.compute_weights(_specs(), date(2023, 1, 3), store)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_all_weights_positive(self, tmp_path):
        store = _stub_store(tmp_path)
        engine = WeightingEngine("risk_parity")
        w = engine.compute_weights(_specs(), date(2023, 1, 3), store)
        assert (w > 0).all()

    def test_higher_vol_gets_lower_weight(self, tmp_path):
        """Ticker with 2× vol should get roughly half the weight."""
        import numpy as np
        store = DataStoreFactory.create_parquet(tmp_path)
        from qis_engine.utils.calendars import business_days_series
        rng = np.random.default_rng(0)
        dates = business_days_series(date(2022, 1, 3), date(2023, 1, 3))
        # LOW_VOL: σ=1%, HIGH_VOL: σ=2%
        for ticker, vol in [("LV", 0.01), ("HV", 0.02)]:
            prices = 100 * np.cumprod(1 + rng.normal(0, vol, len(dates)))
            store.write_ohlcv(pl.DataFrame({
                "date": dates, "open": prices, "high": prices, "low": prices,
                "close": prices, "volume": [100_000] * len(dates),
                "open_interest": [200_000] * len(dates),
            }), ticker)
        specs = [
            ContractSpec("LV", "LV", Sector.ENERGY, Exchange.NYMEX, 1000, 0.01, "USD", Settlement.CASH),
            ContractSpec("HV", "HV", Sector.ENERGY, Exchange.NYMEX, 1000, 0.01, "USD", Settlement.CASH),
        ]
        engine = WeightingEngine("risk_parity")
        w = engine.compute_weights(specs, date(2023, 1, 3), store)
        lv_w, hv_w = w.to_list()
        assert lv_w > hv_w, "Lower vol ticker should have higher risk-parity weight"


class TestConstraints:
    def test_max_weight_applied(self):
        raw = {"A": 0.8, "B": 0.1, "C": 0.1}
        specs = [
            ContractSpec(t, t, Sector.ENERGY, Exchange.NYMEX, 1, 0.01, "USD", Settlement.CASH)
            for t in ("A", "B", "C")
        ]
        c = WeightingConstraints(max_weight=0.5)
        w = _apply_constraints(raw, specs, c)
        assert (w <= 0.5 + 1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_min_weight_applied(self):
        raw = {"A": 0.99, "B": 0.005, "C": 0.005}
        specs = [
            ContractSpec(t, t, Sector.ENERGY, Exchange.NYMEX, 1, 0.01, "USD", Settlement.CASH)
            for t in ("A", "B", "C")
        ]
        c = WeightingConstraints(min_weight=0.05)
        w = _apply_constraints(raw, specs, c)
        assert (w >= 0.05 - 1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_empty_eligible_returns_empty_series(self, tmp_path):
        store = _stub_store(tmp_path)
        engine = WeightingEngine("equal")
        w = engine.compute_weights([], date(2023, 1, 3), store)
        assert len(w) == 0
