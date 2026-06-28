"""
Integration test: full BCOM-style index calculation from CSV stub data.
Validates the end-to-end pipeline: CSVStubProvider → DataStore → IndexCalculationEngine.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

STUBS_DIR = Path(__file__).parent.parent / "stubs"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from qis_engine.data.csv_stub import CSVStubProvider
    from qis_engine.data.store import DataStoreFactory

    root = tmp_path_factory.mktemp("store")
    provider = CSVStubProvider(STUBS_DIR)
    s = DataStoreFactory.create_parquet(root)
    provider.load_all(s)

    # Also write roll schedule to store for engine use
    roll_df = provider.get_roll_schedule()
    s.write_roll_schedule(roll_df, "BCOM")
    s.write_roll_schedule(roll_df, "GSCI")
    return s


@pytest.fixture(scope="module")
def provider():
    from qis_engine.data.csv_stub import CSVStubProvider
    return CSVStubProvider(STUBS_DIR)


@pytest.fixture(scope="module")
def bcom_engine(store, provider):
    from qis_engine.calculation.engine import IndexCalculationEngine
    from qis_engine.core.index import BCOMIndex
    config = BCOMIndex.default_config()
    return IndexCalculationEngine.from_config(config, store, provider)


class TestCSVStubLoading:
    def test_stubs_directory_exists(self):
        assert STUBS_DIR.exists(), f"Stubs directory not found: {STUBS_DIR}"

    def test_ohlcv_files_present(self):
        for ticker in ("CL", "GC", "NG", "HG", "ZC", "ZW", "LC"):
            path = STUBS_DIR / "futures_ohlcv" / f"{ticker}.csv"
            assert path.exists(), f"Missing stub: {path}"

    def test_reference_files_present(self):
        for fname in ("contract_specs.csv", "roll_schedule.csv", "production_weights.csv"):
            assert (STUBS_DIR / "reference" / fname).exists()

    def test_ohlcv_rows_non_empty(self, store):
        df = store.read_ohlcv("CL", date(2015, 1, 1), date(2020, 12, 31))
        assert len(df) > 1000, "Expected 5+ years of daily CL data"

    def test_ohlcv_has_required_columns(self, store):
        df = store.read_ohlcv("GC", date(2020, 1, 1), date(2020, 12, 31))
        assert set(df.columns) >= {"date", "open", "high", "low", "close", "volume", "open_interest"}

    def test_close_prices_positive(self, store):
        df = store.read_ohlcv("CL", date(2010, 1, 1), date(2024, 12, 31))
        assert (df["close"] > 0).all()

    def test_contract_specs_loaded(self, provider):
        specs = provider.get_contract_specs()
        tickers = [s.ticker for s in specs]
        for t in ("CL", "GC", "NG", "HG", "ZC", "ZW", "LC"):
            assert t in tickers


class TestBCOMCalculation:
    START = date(2020, 1, 2)
    END = date(2021, 12, 31)

    def test_calculation_runs_without_error(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_required_columns_present(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        required = {
            "date", "spot_return", "roll_return", "collateral_return",
            "excess_return", "total_return", "spot_level", "excess_level", "total_level",
        }
        assert required.issubset(set(result.columns))

    def test_levels_start_at_base(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        assert abs(float(result["excess_level"][0]) - 100.0) < 1e-6

    def test_levels_always_positive(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        assert (result["spot_level"] > 0).all()
        assert (result["excess_level"] > 0).all()
        assert (result["total_level"] > 0).all()

    def test_no_null_values_in_returns(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        for col in ("spot_return", "excess_return", "total_return"):
            assert result[col].null_count() == 0

    def test_excess_return_decomposition(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        diff = (result["excess_return"] - result["spot_return"] - result["roll_return"]).abs()
        assert diff.max() < 1e-12

    def test_weights_sum_to_one(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        w_cols = [c for c in result.columns if c.startswith("w_")]
        assert len(w_cols) == 7  # 7 commodities
        row_sums = result.select(pl.sum_horizontal(w_cols).alias("ws"))["ws"]
        assert (row_sums - 1.0).abs().max() < 1e-9

    def test_date_range_covers_requested_period(self, bcom_engine):
        result = bcom_engine.run_calculation(self.START, self.END)
        assert result["date"].min() >= self.START
        assert result["date"].max() <= self.END

    def test_10_year_calculation_performance(self, bcom_engine):
        import time
        t0 = time.perf_counter()
        result = bcom_engine.run_calculation(date(2014, 1, 1), date(2024, 12, 31))
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"10-year calculation took {elapsed:.1f}s (limit: 30s)"
        assert len(result) > 2000
