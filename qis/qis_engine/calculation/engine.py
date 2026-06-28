from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from qis_engine.context.base import CalculationContext
from qis_engine.core.index import IndexConfig, ReturnType, BCOMIndex, GSCIIndex
from qis_engine.core.roll import RollCalendar
from qis_engine.core.universe import CommodityUniverse
from qis_engine.core.weighting import WeightingConstraints, WeightingEngine
from qis_engine.calculation.rebalancer import RebalancingEngine
from qis_engine.utils.calendars import business_days


@dataclass
class IndexDailyReturn:
    date: date
    spot_return: float
    roll_return: float
    collateral_return: float
    excess_return: float
    total_return: float
    spot_level: float
    excess_level: float
    total_level: float
    weights: dict[str, float]


class IndexCalculationEngine:
    """
    Computes daily index levels across Spot, Excess, and Total return types.

    Return decomposition:
        Spot Return    = Σ_i  w_i × (close_i,t / close_i,t-1 − 1)
        Roll Return    = Σ_i  w_i × Δroll_fraction_i × (front_i / deferred_i − 1)
        Excess Return  = Spot Return + Roll Return
        Collateral     = collateral_rate / 360   (Actual/360, daily accrual)
        Total Return   = Excess Return + Collateral Return
    """

    def __init__(
        self,
        config: IndexConfig,
        context: CalculationContext,
        universe: CommodityUniverse,
        weighting_engine: WeightingEngine,
        roll_calendars: dict[str, RollCalendar],
    ) -> None:
        self._config = config
        self._context = context
        self._universe = universe
        self._weighting = weighting_engine
        self._roll_calendars = roll_calendars
        self._rebalancer = RebalancingEngine(
            config.rebalance_schedule,
            announce_month=config.rebalance_announce_month,
        )

    @classmethod
    def from_config(
        cls,
        config: IndexConfig,
        context: CalculationContext,
    ) -> "IndexCalculationEngine":
        """Convenience factory: builds universe, weighting engine, and roll calendars from context."""
        universe = CommodityUniverse.from_context(context)
        weighting = WeightingEngine(config.weighting_method)

        schedule_df = context.reference_data.get_roll_schedule(config.name)
        roll_calendars: dict[str, RollCalendar] = {}
        if not schedule_df.is_empty():
            for ticker in config.tickers:
                roll_calendars[ticker] = RollCalendar(ticker, schedule_df)

        return cls(config, context, universe, weighting, roll_calendars)

    def run_calculation(
        self,
        start: date,
        end: date,
        constraints: WeightingConstraints | None = None,
        collateral_rate_series: pl.Series | None = None,
    ) -> pl.DataFrame:
        """
        Full daily calculation loop.

        Returns a Polars DataFrame with columns:
            date, spot_return, roll_return, collateral_return,
            excess_return, total_return, spot_level, excess_level, total_level
        Plus one column per ticker for weight history.
        """
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
            if i == 0 or self._rebalancer.is_rebalance_day(dt):
                w_series = self._weighting.compute_weights(
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
            roll_r = self._compute_roll_return(dt, current_weights, today_prices)

            # --- Collateral return ---
            coll_rate = 0.0
            if collateral_rate_series is not None and i < len(collateral_rate_series):
                coll_rate = float(collateral_rate_series[i])
            coll_r = (self._config.collateral_rate or coll_rate) / 360.0

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_roll_return(
        self,
        dt: date,
        weights: dict[str, float],
        today_prices: dict[str, float],
    ) -> float:
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


# ------------------------------------------------------------------
# Registry of known index configs
# ------------------------------------------------------------------

_CONFIG_REGISTRY: dict[str, IndexConfig] = {
    "BCOM": BCOMIndex.default_config(),
    "GSCI": GSCIIndex.default_config(),
}


def calculate(
    index_name: str,
    start: date,
    end: date,
    context: CalculationContext,
    constraints: WeightingConstraints | None = None,
) -> pl.DataFrame:
    """
    Stateless entry point — the servlet service() method.

    Builds a fresh IndexCalculationEngine from the context on every call.
    All calculation state lives only on this call's stack.
    """
    config = _CONFIG_REGISTRY.get(index_name)
    if config is None:
        raise ValueError(f"Unknown index: {index_name!r}. Available: {list(_CONFIG_REGISTRY)}")
    engine = IndexCalculationEngine.from_config(config, context)
    return engine.run_calculation(start, end, constraints)
