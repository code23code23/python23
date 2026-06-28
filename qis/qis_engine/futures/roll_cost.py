from __future__ import annotations

from datetime import date

import polars as pl


class RollCostCalculator:
    """
    Computes roll cost timeseries from OHLCV data.

    Roll cost on date d = (front_close_d - deferred_close_d) / front_close_d
    Positive → backwardation (roll earns); negative → contango (roll costs).

    Since stub data has one price series per ticker (no per-month contracts),
    we approximate deferred prices as front × (1 + contango_per_month × n).
    """

    CONTANGO_PER_MONTH: dict[str, float] = {
        # Typical historical contango/backwardation estimates per commodity
        "CL": -0.005,   # ~-0.5%/month contango typical for crude
        "NG": -0.010,   # Natural gas seasonal
        "GC": 0.002,    # Gold slight backwardation (carry)
        "HG": -0.003,   # Copper mild contango
        "ZC": -0.004,   # Corn
        "ZW": -0.004,   # Wheat
        "LC": 0.001,    # Live cattle
    }

    def __init__(self, store: "DataStore", n_deferred: int = 1) -> None:  # noqa: F821
        self._store = store
        self._n = n_deferred

    def calculate(
        self, ticker: str, start: date, end: date
    ) -> pl.DataFrame:
        """
        Return a DataFrame with columns: date, front_close, deferred_close, roll_cost.
        """
        ohlcv = self._store.read_ohlcv(ticker, start, end)
        if ohlcv.is_empty():
            return pl.DataFrame(schema={
                "date": pl.Date,
                "front_close": pl.Float64,
                "deferred_close": pl.Float64,
                "roll_cost": pl.Float64,
            })

        contango = self.CONTANGO_PER_MONTH.get(ticker, -0.004)
        deferred_factor = 1.0 + contango * self._n

        return ohlcv.select([
            "date",
            pl.col("close").alias("front_close"),
            (pl.col("close") / deferred_factor).alias("deferred_close"),
            pl.lit(contango * self._n * -1).alias("roll_cost"),
        ])

    def average_roll_cost(
        self, ticker: str, start: date, end: date
    ) -> float:
        df = self.calculate(ticker, start, end)
        if df.is_empty():
            return 0.0
        return float(df["roll_cost"].mean())
