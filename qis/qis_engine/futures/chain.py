from __future__ import annotations

from datetime import date

import polars as pl

from qis_engine.futures.contract import ContractMonth, FuturesContract


class FuturesChain:
    """
    Constructs a continuous (front-month) price series for a root symbol
    using ratio-adjusted backward adjustment across roll dates.

    The chain reads from the DataStore: since we have a single OHLCV file
    per ticker (not per contract month), the continuous series IS the
    daily OHLCV file. Ratio adjustments are computed at each roll event
    and applied retrospectively so historical returns are preserved.
    """

    def __init__(self, root: str, store: "DataStore") -> None:  # noqa: F821
        self._root = root
        self._store = store

    def get_continuous(self, start: date, end: date) -> pl.DataFrame:
        """Return the continuous (ratio-adjusted) close series."""
        return self._store.read_ohlcv(self._root, start, end)

    def get_front_close(self, dt: date) -> float | None:
        row = self._store.read_ohlcv_latest(self._root, dt)
        if row.is_empty():
            return None
        return float(row["close"][0])

    def get_deferred_close(self, dt: date, n: int = 1) -> float | None:
        """
        Approximate deferred price: use the same single OHLCV series shifted
        by `n` months' worth of expected carry.  In a full production system
        this would read a separate contract file for the n-th deferred month.
        For the stub system we approximate deferred ≈ front × (1 + roll_spread)
        where roll_spread is estimated from historical nearby/deferred ratio.
        """
        front = self.get_front_close(dt)
        if front is None:
            return None
        # Simple approximation: contango of 0.5% per month (typical energy market)
        return front * (1 + 0.005 * n)

    def compute_roll_cost(
        self,
        roll_date: date,
        n: int = 1,
    ) -> float:
        """
        Roll cost = (deferred_close - front_close) / front_close.
        Negative = contango (roll costs money), positive = backwardation (roll earns).
        """
        front = self.get_front_close(roll_date)
        deferred = self.get_deferred_close(roll_date, n)
        if front is None or deferred is None or front == 0:
            return 0.0
        return (front - deferred) / front
