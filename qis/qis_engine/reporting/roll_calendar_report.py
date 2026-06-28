from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from qis_engine.core.roll import RollCalendar


class RollCalendarReport:
    """
    Produces a tabular report of roll windows for one or more tickers
    over a requested date range.
    """

    def __init__(self, calendars: dict[str, RollCalendar]) -> None:
        self._calendars = calendars

    def build(self, start: date, end: date) -> pl.DataFrame:
        """
        Return a DataFrame with columns:
            ticker, year, month, roll_start, roll_end, from_contract, to_contract
        covering all roll events within [start, end].
        """
        rows = []
        for ticker, cal in self._calendars.items():
            for sched in cal.all_schedules(start, end):
                rows.append({
                    "ticker": sched.ticker,
                    "year": sched.year,
                    "month": sched.month,
                    "roll_start": sched.roll_start,
                    "roll_end": sched.roll_end,
                    "from_contract": sched.from_contract,
                    "to_contract": sched.to_contract,
                })
        if not rows:
            return pl.DataFrame(schema={
                "ticker": pl.Utf8, "year": pl.Int32, "month": pl.Int32,
                "roll_start": pl.Date, "roll_end": pl.Date,
                "from_contract": pl.Utf8, "to_contract": pl.Utf8,
            })
        return pl.DataFrame(rows).sort(["ticker", "year", "month"])

    def to_csv(self, start: date, end: date, path: str | Path) -> Path:
        df = self.build(start, end)
        out = Path(path)
        df.write_csv(str(out))
        return out

    def to_dicts(self, start: date, end: date) -> list[dict]:
        df = self.build(start, end)
        return df.with_columns(
            pl.col("roll_start").cast(pl.Utf8),
            pl.col("roll_end").cast(pl.Utf8),
        ).to_dicts()
