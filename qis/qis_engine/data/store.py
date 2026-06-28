from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl


class DataStore:
    """
    Unified time-series store. Backend is either Parquet files (dev) or
    ArcticDB (production). All data surfaces as Polars DataFrames.

    Parquet layout:
        root/
            futures_ohlcv/<ticker>.parquet
            options_chain/<ticker>_<type>.parquet
            index_levels/<index_name>.parquet
            weights_history/<index_name>.parquet
            roll_schedules/<index_name>.parquet
    """

    LIBRARIES = (
        "futures_ohlcv",
        "options_chain",
        "index_levels",
        "weights_history",
        "roll_schedules",
    )

    def __init__(
        self,
        backend: Literal["parquet", "arcticdb"],
        root: Path | None = None,
        arcticdb_uri: str | None = None,
    ) -> None:
        self._backend = backend
        if backend == "parquet":
            assert root is not None, "root required for parquet backend"
            self._root = Path(root)
            for lib in self.LIBRARIES:
                (self._root / lib).mkdir(parents=True, exist_ok=True)
        elif backend == "arcticdb":
            assert arcticdb_uri is not None, "arcticdb_uri required"
            import arcticdb as adb  # optional dep
            self._arctic = adb.Arctic(arcticdb_uri)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------
    # Futures OHLCV
    # ------------------------------------------------------------------

    def write_ohlcv(self, df: pl.DataFrame, ticker: str) -> None:
        self._write("futures_ohlcv", ticker, df)

    def read_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        df = self._read("futures_ohlcv", ticker)
        if df.is_empty():
            return df
        return df.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        )

    def read_ohlcv_latest(self, ticker: str, as_of: date) -> pl.DataFrame:
        """Return the single row for the most recent date ≤ as_of."""
        df = self._read("futures_ohlcv", ticker)
        if df.is_empty():
            return df
        return (
            df.filter(pl.col("date") <= pl.lit(as_of))
            .sort("date")
            .tail(1)
        )

    def read_ohlcv_multi(
        self, tickers: list[str], start: date, end: date
    ) -> pl.DataFrame:
        """Return concatenated OHLCV for multiple tickers with a 'ticker' column."""
        frames = []
        for t in tickers:
            df = self.read_ohlcv(t, start, end)
            if not df.is_empty():
                frames.append(df.with_columns(pl.lit(t).alias("ticker")))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def write_options_chain(self, df: pl.DataFrame, ticker: str, chain_type: str) -> None:
        self._write("options_chain", f"{ticker}_{chain_type}", df)

    def read_options_chain(
        self, ticker: str, as_of: date, option_type: str
    ) -> pl.DataFrame:
        suffix = "calls" if option_type == "call" else "puts"
        df = self._read("options_chain", f"{ticker}_{suffix}")
        if df.is_empty():
            return df
        return df.filter(pl.col("date") == pl.lit(as_of))

    # ------------------------------------------------------------------
    # Index levels
    # ------------------------------------------------------------------

    def write_index_level(self, df: pl.DataFrame, index_name: str) -> None:
        existing = self._read("index_levels", index_name)
        if not existing.is_empty():
            df = pl.concat([existing, df], how="diagonal").unique("date").sort("date")
        self._write("index_levels", index_name, df)

    def read_index_level(
        self, index_name: str, start: date, end: date
    ) -> pl.DataFrame:
        df = self._read("index_levels", index_name)
        if df.is_empty():
            return df
        return df.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        )

    # ------------------------------------------------------------------
    # Weights history
    # ------------------------------------------------------------------

    def write_weights(self, df: pl.DataFrame, index_name: str) -> None:
        existing = self._read("weights_history", index_name)
        if not existing.is_empty():
            df = pl.concat([existing, df]).unique("date").sort("date")
        self._write("weights_history", index_name, df)

    def read_weights(
        self, index_name: str, start: date, end: date
    ) -> pl.DataFrame:
        df = self._read("weights_history", index_name)
        if df.is_empty():
            return df
        return df.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        )

    # ------------------------------------------------------------------
    # Roll schedules
    # ------------------------------------------------------------------

    def write_roll_schedule(self, df: pl.DataFrame, index_name: str) -> None:
        self._write("roll_schedules", index_name, df)

    def read_roll_schedule(self, index_name: str) -> pl.DataFrame:
        return self._read("roll_schedules", index_name)

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _write(self, library: str, key: str, df: pl.DataFrame) -> None:
        if self._backend == "parquet":
            path = self._root / library / f"{key}.parquet"
            df.write_parquet(path)
        else:
            lib = self._arctic.get_library(library, create_if_missing=True)
            lib.write(key, df.to_pandas())

    def _read(self, library: str, key: str) -> pl.DataFrame:
        if self._backend == "parquet":
            path = self._root / library / f"{key}.parquet"
            if not path.exists():
                return pl.DataFrame()
            return pl.read_parquet(path)
        else:
            lib = self._arctic.get_library(library, create_if_missing=True)
            try:
                return pl.from_pandas(lib.read(key).data)
            except Exception:
                return pl.DataFrame()


class DataStoreFactory:
    @staticmethod
    def create_parquet(root: str | Path) -> DataStore:
        return DataStore("parquet", root=Path(root))

    @staticmethod
    def create_arcticdb(uri: str) -> DataStore:
        return DataStore("arcticdb", arcticdb_uri=uri)
