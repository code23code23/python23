"""Tests for core/roll.py — roll fraction interpolation and schedule logic."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from qis_engine.core.roll import MomentumRollFinder, OIRollFinder, RollCalendar, RollSchedule
from qis_engine.utils.calendars import nth_business_day_of_month


def _make_schedule(year=2022, month=3) -> RollSchedule:
    roll_start = nth_business_day_of_month(year, month, 5)
    roll_end = nth_business_day_of_month(year, month, 9)
    return RollSchedule(
        ticker="CL",
        year=year,
        month=month,
        roll_start=roll_start,
        roll_end=roll_end,
        from_contract=f"CL_{year}_{month:02d}",
        to_contract=f"CL_{year}_{month + 1:02d}",
    )


class TestRollSchedule:
    def test_fraction_before_roll(self):
        s = _make_schedule()
        before = date(s.roll_start.year, s.roll_start.month, s.roll_start.day - 1)
        # go back until weekday
        from datetime import timedelta
        while before.weekday() >= 5:
            before -= timedelta(days=1)
        assert s.roll_fraction_on(before) == 0.0

    def test_fraction_after_roll(self):
        s = _make_schedule()
        assert s.roll_fraction_on(s.roll_end) == 1.0

    def test_fraction_at_start(self):
        s = _make_schedule()
        frac = s.roll_fraction_on(s.roll_start)
        assert 0.0 <= frac <= 1.0

    def test_fraction_monotone(self):
        s = _make_schedule()
        from qis_engine.utils.calendars import business_days
        bdays = business_days(s.roll_start, s.roll_end)
        fracs = [s.roll_fraction_on(d) for d in bdays]
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1], "Roll fraction must be non-decreasing"

    def test_daily_roll_fraction_sums_to_one(self):
        s = _make_schedule()
        from qis_engine.utils.calendars import business_days
        bdays = business_days(s.roll_start, s.roll_end)
        total = sum(s.daily_roll_fraction(d) for d in bdays)
        assert abs(total - 1.0) < 0.05, f"Daily fractions sum to {total}, expected ~1.0"


class TestRollCalendar:
    def _make_calendar(self) -> tuple[RollCalendar, pl.DataFrame]:
        rows = []
        for y in range(2020, 2023):
            for m in range(1, 13):
                rows.append({
                    "ticker": "CL",
                    "year": y,
                    "month": m,
                    "roll_start_bday": 5,
                    "roll_end_bday": 9,
                    "contract_month": m + 1 if m < 12 else 1,
                })
        df = pl.DataFrame(rows)
        return RollCalendar("CL", df), df

    def test_get_roll_schedule_returns_schedule(self):
        cal, _ = self._make_calendar()
        s = cal.get_roll_schedule(2021, 6)
        assert s is not None
        assert s.ticker == "CL"
        assert s.year == 2021
        assert s.month == 6

    def test_get_roll_schedule_missing_month(self):
        df = pl.DataFrame([{"ticker": "GC", "year": 2021, "month": 6,
                             "roll_start_bday": 5, "roll_end_bday": 9,
                             "contract_month": 8}])
        cal = RollCalendar("CL", df)
        s = cal.get_roll_schedule(2021, 6)
        assert s is None

    def test_get_active_schedule_during_roll(self):
        cal, _ = self._make_calendar()
        roll_start = nth_business_day_of_month(2021, 6, 5)
        s = cal.get_active_schedule(roll_start)
        assert s is not None
        assert s.roll_start == roll_start

    def test_get_active_schedule_outside_roll(self):
        cal, _ = self._make_calendar()
        non_roll = date(2021, 6, 1)  # before 5th business day
        s = cal.get_active_schedule(non_roll)
        assert s is None

    def test_all_schedules(self):
        cal, _ = self._make_calendar()
        schedules = cal.all_schedules(date(2021, 1, 1), date(2021, 12, 31))
        assert len(schedules) == 12


class TestOIRollFinder:
    def test_finds_roll_when_oi_drops(self):
        dates = [date(2021, 6, i) for i in range(1, 31) if date(2021, 6, i).weekday() < 5]
        # OI starts high and drops below threshold after 2/3 of the month
        oi_values = [400_000] * 10 + [100_000] * (len(dates) - 10)
        df = pl.DataFrame({"date": dates, "open_interest": oi_values})
        finder = OIRollFinder(oi_threshold_pct=0.30)
        roll_date = finder.find_roll_date(df, dates[0], dates[-1])
        assert roll_date is not None
        assert roll_date == dates[10]

    def test_returns_none_when_oi_never_drops(self):
        dates = [date(2021, 6, i) for i in range(1, 21) if date(2021, 6, i).weekday() < 5]
        oi_values = [400_000] * len(dates)
        df = pl.DataFrame({"date": dates, "open_interest": oi_values})
        finder = OIRollFinder(oi_threshold_pct=0.30)
        roll_date = finder.find_roll_date(df, dates[0], dates[-1])
        assert roll_date is None


class TestMomentumRollFinder:
    def test_picks_highest_ratio_date(self):
        dates = [date(2021, 3, 5), date(2021, 3, 8), date(2021, 3, 9)]
        nearby_closes = [80.0, 82.0, 81.0]
        deferred_closes = [79.0, 80.0, 80.5]
        nearby = pl.DataFrame({"date": dates, "close": nearby_closes})
        deferred = pl.DataFrame({"date": dates, "close": deferred_closes})
        finder = MomentumRollFinder()
        best = finder.find_roll_date(nearby, deferred, dates[0], dates[-1])
        # ratios: 80/79≈1.013, 82/80=1.025, 81/80.5≈1.006 → best is date[1]
        assert best == dates[1]
