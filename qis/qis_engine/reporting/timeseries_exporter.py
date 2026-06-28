from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


class TimeseriesExporter:
    """
    Exports index level and weight timeseries to CSV or newline-delimited JSON.
    Reads from a DataStore; no re-calculation is performed.
    """

    def __init__(self, store: "DataStore") -> None:  # noqa: F821
        self._store = store

    def export_levels_csv(
        self,
        index_name: str,
        start: date,
        end: date,
        path: str | Path,
    ) -> Path:
        df = self._store.read_index_level(index_name, start, end)
        out = Path(path)
        df.write_csv(str(out))
        return out

    def export_levels_json(
        self,
        index_name: str,
        start: date,
        end: date,
        path: str | Path,
    ) -> Path:
        df = self._store.read_index_level(index_name, start, end)
        out = Path(path)
        df.write_ndjson(str(out))
        return out

    def export_weights_csv(
        self,
        index_name: str,
        start: date,
        end: date,
        path: str | Path,
    ) -> Path:
        df = self._store.read_weights(index_name, start, end)
        out = Path(path)
        df.write_csv(str(out))
        return out

    def levels_to_dicts(
        self,
        index_name: str,
        start: date,
        end: date,
    ) -> list[dict]:
        """Return index level rows as a list of dicts (for API responses)."""
        df = self._store.read_index_level(index_name, start, end)
        return df.with_columns(pl.col("date").cast(pl.Utf8)).to_dicts()

    def weights_to_dicts(
        self,
        index_name: str,
        start: date,
        end: date,
    ) -> list[dict]:
        df = self._store.read_weights(index_name, start, end)
        return df.with_columns(pl.col("date").cast(pl.Utf8)).to_dicts()
