from __future__ import annotations

from abc import abstractmethod
from datetime import date

import polars as pl

from qis_engine.calculation.calculator import Calculator
from qis_engine.calculation.rebalancer import RebalancingEngine
from qis_engine.context.base import CalculationContext
from qis_engine.core.index import IndexConfig
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine
from qis_engine.data.models import ContractSpec
from qis_engine.utils.calendars import business_days


class CalculatorBase(Calculator):
    """
    Partial Calculator: owns the daily loop and level compounding, and
    delegates the strategy-specific steps to subclasses (template method).

    Subclasses must implement:
        _compute_weights(eligible, as_of, constraints)   → target weights on a rebalance day
        _compute_roll_return(dt, weights, today_prices)  → that day's roll return

    Subclasses may override:
        _is_rebalance_day(dt)                            → default: config's RebalanceSchedule
        _collateral_return(day_index, rate_series)       → default: Actual/360 accrual

    Return decomposition:
        Spot Return    = Σ_i  w_i × (close_i,t / close_i,t-1 − 1)
        Roll Return    = subclass-defined
        Excess Return  = Spot Return + Roll Return
        Collateral     = subclass-defined (default collateral_rate / 360)
        Total Return   = Excess Return + Collateral Return
    """

    def __init__(
        self,
        context: CalculationContext,
        config: IndexConfig,
        universe: CommodityUniverse | None = None,
        roll_calendars: dict[str, RollCalendar] | None = None,
    ) -> None:
        super().__init__(context, config)
        self._universe = universe or CommodityUniverse.from_context(context)
        self._roll_calendars = (
            roll_calendars if roll_calendars is not None
            else self._build_roll_calendars(config, context)
        )
        self._rebalancer = RebalancingEngine(
            config.rebalance_schedule,
            announce_month=config.rebalance_announce_month,
        )

    def calculate(
        self,
        start: date,
        end: date,
        constraints: WeightingConstraints | None = None,
        collateral_rate_series: pl.Series | None = None,
    ) -> pl.DataFrame:
        constraints = constraints or WeightingConstraints()
        bdays = business_days(start, end)
        if not bdays:
            return pl.DataFrame()

        eligible = self._universe.get_eligible(start, tickers=self._config.tickers)

        # Prefetch all prices as date→close dicts for O(1) daily lookup
        price_lookup: dict[str, dict[date, float]] = {}
        for spec in eligible:
            df = self._context.market_data.get_ohlcv(spec.ticker, start, end)
            if not df.is_empty():
                price_lookup[spec.ticker] = dict(
                    zip(df["date"].to_list(), df["close"].to_list())
                )

        records: list[dict] = []
        spot_level = self._config.base_level
        excess_level = self._config.base_level
        total_level = self._config.base_level

        current_weights: dict[str, float] = {}
        prev_prices: dict[str, float] = {}

        for i, dt in enumerate(bdays):
            if i == 0 or self._is_rebalance_day(dt):
                current_weights = self._compute_weights(eligible, dt, constraints)

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

            # --- Strategy-specific components ---
            roll_r = self._compute_roll_return(dt, current_weights, today_prices)
            coll_r = self._collateral_return(i, collateral_rate_series)

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
                **{f"w_{t}": current_weights.get(t, 0.0) for t in self._config.tickers},
            })

            prev_prices = today_prices

        return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))

    # ------------------------------------------------------------------
    # Strategy hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_weights(
        self,
        eligible: list[ContractSpec],
        as_of: date,
        constraints: WeightingConstraints,
    ) -> dict[str, float]:
        """Return ticker → weight for a rebalance on `as_of`."""

    @abstractmethod
    def _compute_roll_return(
        self,
        dt: date,
        weights: dict[str, float],
        today_prices: dict[str, float],
    ) -> float:
        """Return the index's roll return on `dt`."""

    def _is_rebalance_day(self, dt: date) -> bool:
        return self._rebalancer.is_rebalance_day(dt)

    def _collateral_return(
        self,
        day_index: int,
        collateral_rate_series: pl.Series | None,
    ) -> float:
        """Actual/360 daily accrual; a fixed config rate takes precedence over the series."""
        coll_rate = 0.0
        if collateral_rate_series is not None and day_index < len(collateral_rate_series):
            coll_rate = float(collateral_rate_series[day_index])
        return (self._config.collateral_rate or coll_rate) / 360.0

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _weights_from_engine(
        self,
        weighting: WeightingEngine,
        eligible: list[ContractSpec],
        as_of: date,
        constraints: WeightingConstraints,
    ) -> dict[str, float]:
        w_series = weighting.compute_weights(eligible, as_of, self._context, constraints)
        return dict(zip([s.ticker for s in eligible], w_series.to_list()))

    def _scheduled_roll_return(
        self,
        dt: date,
        weights: dict[str, float],
        today_prices: dict[str, float],
    ) -> float:
        """Roll return from the roll calendars, using the per-ticker contango estimate."""
        from qis_engine.futures.roll_cost import RollCostCalculator
        total = 0.0
        for ticker, w in weights.items():
            cal = self._roll_calendars.get(ticker)
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
    def _build_roll_calendars(
        config: IndexConfig,
        context: CalculationContext,
    ) -> dict[str, RollCalendar]:
        schedule_df = context.reference_data.get_roll_schedule(config.name)
        if schedule_df.is_empty():
            return {}
        return {t: RollCalendar(t, schedule_df) for t in config.tickers}

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
