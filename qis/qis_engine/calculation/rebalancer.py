from __future__ import annotations

from datetime import date

import polars as pl

from qis_engine.core.index import RebalanceSchedule


class RebalancingEngine:
    """
    Determines whether a given date is a rebalance day and carries
    weights forward between rebalance events.
    """

    def __init__(self, schedule: RebalanceSchedule, announce_month: int = 1) -> None:
        self._schedule = schedule
        self._announce_month = announce_month  # weights announced in this month take effect next

    def is_rebalance_day(self, dt: date) -> bool:
        """
        Annual: first business day of the month after the announce month.
        Monthly: first business day of each month.
        Quarterly: first business day of Jan/Apr/Jul/Oct.
        """
        from qis_engine.utils.calendars import nth_business_day_of_month

        if self._schedule == RebalanceSchedule.ANNUAL:
            effective_month = (self._announce_month % 12) + 1
            try:
                return dt == nth_business_day_of_month(dt.year, effective_month, 1)
            except ValueError:
                return False

        if self._schedule == RebalanceSchedule.MONTHLY:
            try:
                return dt == nth_business_day_of_month(dt.year, dt.month, 1)
            except ValueError:
                return False

        if self._schedule == RebalanceSchedule.QUARTERLY:
            if dt.month not in (1, 4, 7, 10):
                return False
            try:
                return dt == nth_business_day_of_month(dt.year, dt.month, 1)
            except ValueError:
                return False

        return False

    def next_rebalance_date(self, after: date) -> date:
        """Return the first rebalance date strictly after `after`."""
        from datetime import timedelta
        from qis_engine.utils.calendars import nth_business_day_of_month, next_business_day

        dt = next_business_day(after)
        # scan forward up to 400 days
        for _ in range(400):
            if self.is_rebalance_day(dt):
                return dt
            dt = next_business_day(dt)
        return dt
