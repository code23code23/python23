from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from qis_engine.context.base import CalculationContext
from qis_engine.core.index import IndexConfig
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine
from qis_engine.calculation.rebalancer import RebalancingEngine
from qis_engine.utils.calendars import business_days


class Calculator(ABC):
    """
    Application-scoped, per-index calculation engine.

    Constructed once at server startup with an immutable CalculationContext.
    All engine components (universe, weighting, roll calendars) are built
    at construction time and reused across every calculate() call.
    """

    def __init__(self, context: CalculationContext) -> None:
        self._context = context

    @abstractmethod
    def calculate(
        self,
        start: date,
        end: date,
        constraints: WeightingConstraints | None = None,
    ) -> pl.DataFrame:
        """Return a daily DataFrame: date, *_return, *_level, w_<ticker>…"""

    # ------------------------------------------------------------------
    # Shared implementation — concrete subclasses call this from calculate()
    # ------------------------------------------------------------------

    def _run_calculation(
        self,
        config: IndexConfig,
        universe: CommodityUniverse,
        weighting: WeightingEngine,
        roll_calendars: dict[str, RollCalendar],
        rebalancer: RebalancingEngine,
        start: date,
        end: date,
        constraints: WeightingConstraints | None,
        collateral_rate_series: pl.Series | None = None,
    ) -> pl.DataFrame:
        """
        Full daily calculation loop.

        Return decomposition:
            Spot Return    = Σ_i  w_i × (close_i,t / close_i,t-1 − 1)
            Roll Return    = Σ_i  w_i × Δroll_fraction_i × (front_i / deferred_i − 1)
            Excess Return  = Spot Return + Roll Return
            Collateral     = collateral_rate / 360   (Actual/360, daily accrual)
            Total Return   = Excess Return + Collateral Return
        """
        constraints = constraints or WeightingConstraints()
        bdays = business_days(start, end)
        if not bdays:
            return pl.DataFrame()

        eligible = universe.get_eligible(start, tickers=config.tickers)

        # Prefetch all prices as date→close dicts for O(1) daily lookup
        price_lookup: dict[str, dict[date, float]] = {}
        for spec in eligible:
            df = self._context.market_data.get_ohlcv(spec.ticker, start, end)
            if not df.is_empty():
                price_lookup[spec.ticker] = dict(
                    zip(df["date"].to_list(), df["close"].to_list())
                )

        records: list[dict] = []
        spot_level = config.base_level
        excess_level = config.base_level
        total_level = config.base_level

        current_weights: dict[str, float] = {}
        prev_prices: dict[str, float] = {}

        for i, dt in enumerate(bdays):
            if i == 0 or rebalancer.is_rebalance_day(dt):
                w_series = weighting.compute_weights(
                    eligible, dt, self._context, constraints
                )
                current_weights = dict(zip(
                    [s.ticker for s in eligible],
                    w_series.to_list(),
                ))

            today_prices: dict[str, float] = {}
            for t in current_weights:
                p = price_lookup.get(t, {}).get(dt)
                if p is not None:
                    today_prices[t] = p

            if i == 0:
                prev_prices = today_prices
                records.append(self._zero_record(
                    dt, spot_level, excess_level, total_level, current_weights
                ))
                continue

            # --- Spot return ---
            spot_r = 0.0
            for t, w in current_weights.items():
                if t in today_prices and t in prev_prices and prev_prices[t] != 0:
                    spot_r += w * (today_prices[t] / prev_prices[t] - 1.0)

            # --- Roll return ---
            roll_r = self._compute_roll_return(dt, current_weights, today_prices, roll_calendars)

            # --- Collateral return ---
            coll_rate = 0.0
            if collateral_rate_series is not None and i < len(collateral_rate_series):
                coll_rate = float(collateral_rate_series[i])
            coll_r = (config.collateral_rate or coll_rate) / 360.0

            excess_r = spot_r + roll_r
            total_r = excess_r + coll_r

            spot_level *= (1.0 + spot_r)
            excess_level *= (1.0 + excess_r)
            total_level *= (1.0 + total_r)

            records.append({
                "date": dt,
                "spot_return": spot_r,
                "roll_return": roll_r,
                "collateral_return": coll_r,
                "excess_return": excess_r,
                "total_return": total_r,
                "spot_level": spot_level,
                "excess_level": excess_level,
                "total_level": total_level,
                **{f"w_{t}": current_weights.get(t, 0.0) for t in config.tickers},
            })

            prev_prices = today_prices

        return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))

    @staticmethod
    def _compute_roll_return(
        dt: date,
        weights: dict[str, float],
        today_prices: dict[str, float],
        roll_calendars: dict[str, RollCalendar],
    ) -> float:
        from qis_engine.futures.roll_cost import RollCostCalculator
        total = 0.0
        for ticker, w in weights.items():
            cal = roll_calendars.get(ticker)
            if cal is None:
                continue
            schedule = cal.get_active_schedule(dt)
            if schedule is None:
                continue
            delta_frac = schedule.daily_roll_fraction(dt)
            if delta_frac == 0.0:
                continue
            front = today_prices.get(ticker)
            if front is None:
                continue
            contango = RollCostCalculator.CONTANGO_PER_MONTH.get(ticker, -0.004)
            deferred = front / (1.0 + contango)
            roll_cost = (front - deferred) / deferred if deferred != 0 else 0.0
            total += w * delta_frac * roll_cost
        return total

    @staticmethod
    def _zero_record(
        dt: date,
        spot_level: float,
        excess_level: float,
        total_level: float,
        weights: dict[str, float],
    ) -> dict:
        return {
            "date": dt,
            "spot_return": 0.0,
            "roll_return": 0.0,
            "collateral_return": 0.0,
            "excess_return": 0.0,
            "total_return": 0.0,
            "spot_level": spot_level,
            "excess_level": excess_level,
            "total_level": total_level,
            **{f"w_{t}": w for t, w in weights.items()},
        }
