"""Tests for utils/calendars.py."""
from __future__ import annotations

from datetime import date

import pytest

from qis_engine.utils.calendars import (
    business_days,
    business_days_series,
    next_business_day,
    nth_business_day_of_month,
    prev_business_day,
)


class TestNthBusinessDay:
    def test_first_business_day_jan_2022(self):
        # 2022-01-01 is Saturday → first bday is 2022-01-03 (Monday)
        assert nth_business_day_of_month(2022, 1, 1) == date(2022, 1, 3)

    def test_fifth_business_day(self):
        result = nth_business_day_of_month(2022, 3, 5)
        assert result.weekday() < 5
        assert result.month == 3
        # March 2022: Mon-Fri, 1st=Tue 1st, 2nd=Wed 2nd, 3rd=Thu 3rd, 4th=Fri 4th, 5th=Mon 7th
        assert result == date(2022, 3, 7)

    def test_raises_when_n_exceeds_days(self):
        # Feb 2022 has 20 business days; requesting 21 should raise
        with pytest.raises(ValueError):
            nth_business_day_of_month(2022, 2, 21)


class TestBusinessDays:
    def test_excludes_weekends(self):
        days = business_days(date(2022, 1, 1), date(2022, 1, 9))
        for d in days:
            assert d.weekday() < 5

    def test_count_in_standard_week(self):
        # Mon 2022-01-03 to Fri 2022-01-07 = 5 days
        days = business_days(date(2022, 1, 3), date(2022, 1, 7))
        assert len(days) == 5

    def test_start_equals_end(self):
        # A single weekday
        days = business_days(date(2022, 1, 3), date(2022, 1, 3))
        assert days == [date(2022, 1, 3)]

    def test_weekend_only_range_returns_empty(self):
        days = business_days(date(2022, 1, 1), date(2022, 1, 2))
        assert len(days) == 0


class TestPrevNextBusinessDay:
    def test_prev_business_day_from_monday(self):
        # Previous bday before Mon 2022-01-10 is Fri 2022-01-07
        assert prev_business_day(date(2022, 1, 10)) == date(2022, 1, 7)

    def test_prev_business_day_from_tuesday(self):
        assert prev_business_day(date(2022, 1, 11)) == date(2022, 1, 10)

    def test_next_business_day_from_friday(self):
        # Next bday after Fri 2022-01-07 is Mon 2022-01-10
        assert next_business_day(date(2022, 1, 7)) == date(2022, 1, 10)

    def test_next_business_day_from_weekday(self):
        assert next_business_day(date(2022, 1, 10)) == date(2022, 1, 11)


class TestBusinessDaysSeries:
    def test_returns_polars_series(self):
        import polars as pl
        s = business_days_series(date(2022, 1, 3), date(2022, 1, 7))
        assert isinstance(s, pl.Series)
        assert len(s) == 5
