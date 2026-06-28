"""Unit tests for backtest/simulator.py, backtest/costs.py, backtest/risk.py."""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from qis_engine.backtest.costs import CommissionModel, SlippageModel, TransactionCostModel
from qis_engine.backtest.risk import RiskCalculator
from qis_engine.backtest.simulator import VectorizedSimulator
from qis_engine.calculation.engine import IndexCalculationEngine
from qis_engine.core.index import IndexConfig, RebalanceSchedule
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingEngine
from qis_engine.data.models import ContractSpec, Exchange, Sector, Settlement
from qis_engine.data.store import DataStoreFactory
from qis_engine.utils.calendars import business_days_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(t: str) -> ContractSpec:
    return ContractSpec(t, t, Sector.ENERGY, Exchange.NYMEX, 1000, 0.01, "USD", Settlement.CASH)


def _make_engine_and_store(tmp_path: Path, tickers=("A", "B"), daily_return=0.001):
    import numpy as np
    store = DataStoreFactory.create_parquet(tmp_path)
    dates = business_days_series(date(2022, 1, 3), date(2023, 12, 30))
    rng = np.random.default_rng(7)
    for t in tickers:
        prices = list(100 * np.cumprod(1 + rng.normal(daily_return, 0.005, len(dates))))
        store.write_ohlcv(pl.DataFrame({
            "date": dates, "open": prices, "high": prices,
            "low": prices, "close": prices,
            "volume": [100_000] * len(dates),
            "open_interest": [200_000] * len(dates),
        }), t)
    config = IndexConfig(
        name="TEST",
        base_date=date(2022, 1, 3),
        tickers=list(tickers),
        base_level=100.0,
        rebalance_schedule=RebalanceSchedule.ANNUAL,
    )
    universe = CommodityUniverse([_spec(t) for t in tickers])
    weighting = WeightingEngine("equal")
    engine = IndexCalculationEngine(config, store, universe, weighting, roll_calendars={})
    return engine, store


# ---------------------------------------------------------------------------
# Cost model tests
# ---------------------------------------------------------------------------

class TestCostModels:
    def test_commission_zero_bps(self):
        m = CommissionModel(0.0)
        assert m.cost(1_000_000) == 0.0

    def test_commission_half_bps(self):
        m = CommissionModel(0.5)
        assert abs(m.cost(1_000_000) - 50.0) < 1e-9

    def test_slippage_half_bps(self):
        m = SlippageModel(0.1)
        assert abs(m.cost(1_000_000) - 10.0) < 1e-9

    def test_total_cost_zero_model(self):
        m = TransactionCostModel.zero()
        assert m.total_cost(1_000_000) == 0.0

    def test_default_model_positive(self):
        m = TransactionCostModel.default()
        assert m.total_cost(1_000_000) > 0


# ---------------------------------------------------------------------------
# Risk metric tests
# ---------------------------------------------------------------------------

class TestRiskCalculator:
    def _daily_returns(self, n: int = 500, mu: float = 0.0005, sigma: float = 0.01):
        import numpy as np
        rng = np.random.default_rng(42)
        return pl.Series(rng.normal(mu, sigma, n).tolist())

    def test_positive_returns_positive_sharpe(self):
        r = pl.Series([0.001] * 252)
        m = RiskCalculator(risk_free_rate=0.0).compute(r)
        assert m.sharpe > 0

    def test_zero_returns_zero_sharpe(self):
        r = pl.Series([0.0] * 252)
        m = RiskCalculator().compute(r)
        assert m.sharpe == 0.0

    def test_ann_vol_positive(self):
        r = self._daily_returns()
        m = RiskCalculator().compute(r)
        assert m.annualised_vol > 0

    def test_max_drawdown_negative_or_zero(self):
        r = self._daily_returns()
        m = RiskCalculator().compute(r)
        assert m.max_drawdown <= 0

    def test_var_99_le_var_95(self):
        r = self._daily_returns()
        m = RiskCalculator().compute(r)
        assert m.var_99 <= m.var_95

    def test_cvar_le_var(self):
        r = self._daily_returns()
        m = RiskCalculator().compute(r)
        assert m.cvar_95 <= m.var_95
        assert m.cvar_99 <= m.var_99

    def test_total_return_compounds(self):
        r = pl.Series([0.01] * 252)
        m = RiskCalculator().compute(r)
        expected = (1.01 ** 252) - 1.0
        assert abs(m.total_return - expected) < 1e-6

    def test_empty_returns_zero_metrics(self):
        m = RiskCalculator().compute(pl.Series([], dtype=pl.Float64))
        assert m.n_days == 0
        assert m.sharpe == 0.0

    def test_n_days_correct(self):
        r = self._daily_returns(300)
        m = RiskCalculator().compute(r)
        assert m.n_days == 300

    def test_summary_df_is_single_row(self):
        r = self._daily_returns()
        calc = RiskCalculator()
        m = calc.compute(r)
        df = calc.summary_df(m)
        assert len(df) == 1
        assert "sharpe" in df.columns


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestVectorizedSimulator:
    START = date(2022, 1, 4)
    END = date(2023, 12, 29)

    def test_run_returns_result(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.zero())
        result = sim.run(engine, self.START, self.END)
        assert result is not None
        assert not result.timeseries.is_empty()

    def test_net_le_gross_with_nonzero_costs(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.default())
        result = sim.run(engine, self.START, self.END)
        net_total = float(result.timeseries["net_level"][-1])
        gross_total = float(result.timeseries["gross_level"][-1])
        assert net_total <= gross_total

    def test_zero_cost_gross_equals_net(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.zero())
        result = sim.run(engine, self.START, self.END)
        diff = (result.timeseries["gross_return"] - result.timeseries["net_return"]).abs()
        assert float(diff.max()) < 1e-12

    def test_output_columns_present(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.zero())
        result = sim.run(engine, self.START, self.END)
        required = {"gross_return", "net_return", "gross_level", "net_level", "turnover", "cost"}
        assert required.issubset(set(result.timeseries.columns))

    def test_risk_metrics_populated(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store)
        result = sim.run(engine, self.START, self.END)
        assert math.isfinite(result.risk_net.sharpe)
        assert math.isfinite(result.risk_net.max_drawdown)

    def test_to_csv_writes_file(self, tmp_path):
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.zero())
        result = sim.run(engine, self.START, self.END)
        out = str(tmp_path / "backtest.csv")
        result.to_csv(out)
        import os
        assert os.path.getsize(out) > 0

    def test_performance_2year_under_5s(self, tmp_path):
        import time
        engine, store = _make_engine_and_store(tmp_path)
        sim = VectorizedSimulator(store, TransactionCostModel.zero())
        t0 = time.perf_counter()
        result = sim.run(engine, self.START, self.END)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"2-year backtest took {elapsed:.1f}s"
