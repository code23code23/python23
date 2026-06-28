from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from qis_engine.data.base import DataProviderBase
from qis_engine.data.models import ContractSpec


class CSVStubProvider(DataProviderBase):
    """
    Reads all price and reference data from CSV test stubs under `stubs_dir/`.
    This is the only concrete DataProvider in the system.

    Directory layout expected:
        stubs_dir/
            futures_ohlcv/<TICKER>.csv
            options_chain/<TICKER>_calls.csv
            options_chain/<TICKER>_puts.csv
            reference/contract_specs.csv
            reference/roll_schedule.csv
            reference/production_weights.csv
    """

    def __init__(self, stubs_dir: str | Path) -> None:
        self._root = Path(stubs_dir)
        self._specs: list[ContractSpec] | None = None

    # ------------------------------------------------------------------
    # DataProviderBase interface
    # ------------------------------------------------------------------

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        path = self._root / "futures_ohlcv" / f"{ticker}.csv"
        if not path.exists():
            return pl.DataFrame(schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64,
                                         "low": pl.Float64, "close": pl.Float64,
                                         "volume": pl.Int64, "open_interest": pl.Int64})
        df = (
            pl.scan_csv(path, try_parse_dates=True)
            .filter(
                (pl.col("date") >= pl.lit(start))
                & (pl.col("date") <= pl.lit(end))
            )
            .collect()
        )
        return df

    def get_options_chain(self, ticker: str, as_of: date, option_type: str) -> pl.DataFrame:
        suffix = "calls" if option_type == "call" else "puts"
        path = self._root / "options_chain" / f"{ticker}_{suffix}.csv"
        if not path.exists():
            return pl.DataFrame()
        df = (
            pl.scan_csv(path, try_parse_dates=True)
            .filter(pl.col("date") == pl.lit(as_of))
            .collect()
        )
        return df

    def get_contract_specs(self) -> list[ContractSpec]:
        if self._specs is None:
            self._specs = self._load_specs()
        return self._specs

    def load_all(self, store: "DataStore") -> None:  # type: ignore[name-defined]  # noqa: F821
        """Ingest all stub CSVs into the provided DataStore."""
        for spec in self.get_contract_specs():
            ohlcv_path = self._root / "futures_ohlcv" / f"{spec.ticker}.csv"
            if ohlcv_path.exists():
                df = pl.read_csv(ohlcv_path, try_parse_dates=True)
                store.write_ohlcv(df, spec.ticker)

        for path in (self._root / "options_chain").glob("*.csv"):
            df = pl.read_csv(path, try_parse_dates=True)
            # Derive ticker and type from filename: CL_calls.csv
            stem = path.stem  # e.g. "CL_calls"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                store.write_options_chain(df, parts[0], parts[1])

    # ------------------------------------------------------------------
    # Reference data helpers (used by core/ layer directly)
    # ------------------------------------------------------------------

    def get_roll_schedule(self) -> pl.DataFrame:
        path = self._root / "reference" / "roll_schedule.csv"
        return pl.read_csv(path)

    def get_production_weights(self) -> pl.DataFrame:
        path = self._root / "reference" / "production_weights.csv"
        return pl.read_csv(path)

    # ------------------------------------------------------------------

    def _load_specs(self) -> list[ContractSpec]:
        path = self._root / "reference" / "contract_specs.csv"
        rows = pl.read_csv(path).to_dicts()
        return [ContractSpec.from_dict(r) for r in rows]
