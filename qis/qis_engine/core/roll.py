from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from qis_engine.utils.calendars import nth_business_day_of_month

_CME_MONTH = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}


def _parse_cme_code(code: str, ticker: str) -> tuple[int, int]:
    """Parse e.g. 'CLG20' with ticker='CL' → (month=2, year=2020)."""
    suffix = code[len(ticker):]     # 'G20'
    month = _CME_MONTH[suffix[0]]
    yr = int(suffix[1:])
    full_year = 2000 + yr if yr < 70 else 1900 + yr
    return month, full_year


@dataclass
class RollSchedule:
    """One roll event: move from from_contract to to_contract over roll window."""
    ticker: str
    year: int
    month: int
    roll_start: date    # 5th business day
    roll_end: date      # 9th business day
    from_contract: str  # e.g. "CL_2024_01"
    to_contract: str    # e.g. "CL_2024_02"

    def roll_fraction_on(self, dt: date) -> float:
        """
        Linear roll fraction: 0.0 before roll_start, 1.0 after roll_end.
        Increases by 1/(roll_end - roll_start) business days each roll day.
        """
        if dt < self.roll_start:
            return 0.0
        if dt >= self.roll_end:
            return 1.0
        total_days = max((self.roll_end - self.roll_start).days, 1)
        elapsed = (dt - self.roll_start).days
        return min(elapsed / total_days, 1.0)

    def daily_roll_fraction(self, dt: date) -> float:
        """Incremental fraction rolled on date dt (fraction rolled today minus yesterday)."""
        from datetime import timedelta
        prev = dt - timedelta(days=1)
        # skip weekends for prev
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        return self.roll_fraction_on(dt) - self.roll_fraction_on(prev)


class RollCalendar:
    """
    Builds and serves roll schedules for a single ticker.
    Loaded from the CSV stub reference data.
    """

    def __init__(self, ticker: str, schedule_df: pl.DataFrame) -> None:
        self._ticker = ticker
        self._df = schedule_df.filter(pl.col("ticker") == ticker)
        self._cache: dict[tuple[int, int], RollSchedule | None] = {}

    def get_roll_schedule(self, year: int, month: int) -> RollSchedule | None:
        key = (year, month)
        if key in self._cache:
            return self._cache[key]
        row = self._df.filter(
            (pl.col("year") == year) & (pl.col("month") == month)
        )
        if row.is_empty():
            self._cache[key] = None
            return None
        r = row.row(0, named=True)
        roll_start = nth_business_day_of_month(year, month, int(r["roll_start_bday"]))
        roll_end = nth_business_day_of_month(year, month, int(r["roll_end_bday"]))
        cme_code = str(r["contract_month"])
        try:
            to_month = int(cme_code)
            to_year = year if to_month > month else year + 1
        except ValueError:
            to_month, to_year = _parse_cme_code(cme_code, self._ticker)
        result = RollSchedule(
            ticker=self._ticker,
            year=year,
            month=month,
            roll_start=roll_start,
            roll_end=roll_end,
            from_contract=f"{self._ticker}_{year}_{month:02d}",
            to_contract=f"{self._ticker}_{to_year}_{to_month:02d}",
        )
        self._cache[key] = result
        return result

    def get_active_schedule(self, dt: date) -> RollSchedule | None:
        """Return the roll schedule active on date dt, or None if not a roll period."""
        schedule = self.get_roll_schedule(dt.year, dt.month)
        if schedule and schedule.roll_start <= dt <= schedule.roll_end:
            return schedule
        return None

    def all_schedules(self, start: date, end: date) -> list[RollSchedule]:
        result = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            s = self.get_roll_schedule(y, m)
            if s:
                result.append(s)
            m += 1
            if m > 12:
                m = 1
                y += 1
        return result


class OIRollFinder:
    """
    OI-based roll: switch to next contract when front-month OI drops below
    `oi_threshold_pct` of its peak OI.
    """

    def __init__(self, oi_threshold_pct: float = 0.30) -> None:
        self._threshold = oi_threshold_pct

    def find_roll_date(
        self, ohlcv: pl.DataFrame, start: date, end: date
    ) -> date | None:
        """
        Scan OHLCV series for the first date where OI < threshold × peak_OI.
        Returns None if no such date found.
        """
        window = ohlcv.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        ).sort("date")
        if window.is_empty():
            return None
        peak_oi = float(window["open_interest"].max())
        threshold = peak_oi * self._threshold
        crossing = window.filter(pl.col("open_interest") < threshold)
        if crossing.is_empty():
            return None
        return crossing["date"][0]


class MomentumRollFinder:
    """
    Momentum roll: choose the roll date within the standard roll window that
    minimises roll cost (maximises nearby/deferred price ratio → lowest backwardation).
    """

    def find_roll_date(
        self,
        nearby_ohlcv: pl.DataFrame,
        deferred_ohlcv: pl.DataFrame,
        roll_start: date,
        roll_end: date,
    ) -> date:
        """
        Compare nearby.close / deferred.close for each day in window,
        pick the date with the highest ratio (least backwardation = cheapest roll).
        """
        nearby = nearby_ohlcv.filter(
            (pl.col("date") >= pl.lit(roll_start)) & (pl.col("date") <= pl.lit(roll_end))
        ).select(["date", pl.col("close").alias("nearby_close")])

        deferred = deferred_ohlcv.filter(
            (pl.col("date") >= pl.lit(roll_start)) & (pl.col("date") <= pl.lit(roll_end))
        ).select(["date", pl.col("close").alias("deferred_close")])

        joined = nearby.join(deferred, on="date", how="inner").with_columns(
            (pl.col("nearby_close") / pl.col("deferred_close")).alias("ratio")
        )
        if joined.is_empty():
            return roll_start
        best_row = joined.sort("ratio", descending=True).row(0, named=True)
        return best_row["date"]
