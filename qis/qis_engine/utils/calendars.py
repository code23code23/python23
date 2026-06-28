from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def business_days(start: date, end: date) -> list[date]:
    """Return all Mon–Fri dates in [start, end] (no holiday adjustment)."""
    result = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0=Mon … 4=Fri
            result.append(current)
        current += timedelta(days=1)
    return result


def nth_business_day_of_month(year: int, month: int, n: int) -> date:
    """Return the nth business day (Mon–Fri) of the given month (1-indexed)."""
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    bdays = business_days(date(year, month, 1), end)
    if len(bdays) < n:
        raise ValueError(f"{year}-{month:02d} has only {len(bdays)} business days")
    return bdays[n - 1]


def prev_business_day(dt: date) -> date:
    """Return the most recent business day strictly before dt."""
    candidate = dt - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def next_business_day(dt: date) -> date:
    """Return the first business day strictly after dt."""
    candidate = dt + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def business_days_series(start: date, end: date) -> pl.Series:
    """Return a Polars Date Series of business days in [start, end]."""
    return pl.Series("date", business_days(start, end), dtype=pl.Date)
