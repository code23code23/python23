from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from qis_engine.backtest.costs import TransactionCostModel
from qis_engine.backtest.risk import RiskCalculator, RiskMetrics
from qis_engine.core.index import IndexConfig
from qis_engine.core.roll import RollCalendar


@dataclass
class BacktestResult:
    config_name: str
    start: date
    end: date
    timeseries: pl.DataFrame   # date, gross_return, net_return, cumulative_level, turnover, cost
    risk_gross: RiskMetrics
    risk_net: RiskMetrics

    def to_csv(self, path: str) -> None:
        self.timeseries.write_csv(path)

    def to_json(self, path: str) -> None:
        self.timeseries.write_ndjson(path)


class VectorizedSimulator:
    """
    Fully vectorized backtesting engine built on Polars expressions.

    Takes a pre-computed index calculation DataFrame (from IndexCalculationEngine)
    and applies transaction costs on rebalance / roll dates to produce
    net-of-cost return series and risk metrics.

    No Python date loop — all operations are Polars expression-level.
    """

    def __init__(
        self,
        store: "DataStore",                             # noqa: F821
        cost_model: TransactionCostModel | None = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        self._store = store
        self._costs = cost_model or TransactionCostModel.default()
        self._risk_calc = RiskCalculator(risk_free_rate)

    def run(
        self,
        engine: "IndexCalculationEngine",               # noqa: F821
        start: date,
        end: date,
        constraints: "WeightingConstraints | None" = None,  # noqa: F821
        collateral_rate_series: pl.Series | None = None,
    ) -> BacktestResult:
        """
        Run a full historical simulation.

        1. Call engine.run_calculation() to get gross index returns and weights.
        2. Compute daily turnover from weight changes (vectorized diff).
        3. Apply cost model: net_return = gross_return - cost_per_day.
        4. Compound net level; compute risk metrics.
        """
        gross_df = engine.run_calculation(
            start, end,
            constraints=constraints,
            collateral_rate_series=collateral_rate_series,
        )
        if gross_df.is_empty():
            return self._empty_result(engine._config.name, start, end)

        result_df = self._apply_costs(gross_df, engine._config)
        risk_gross = self._risk_calc.compute(
            result_df["gross_return"], result_df["gross_level"]
        )
        risk_net = self._risk_calc.compute(
            result_df["net_return"], result_df["net_level"]
        )

        return BacktestResult(
            config_name=engine._config.name,
            start=start,
            end=end,
            timeseries=result_df,
            risk_gross=risk_gross,
            risk_net=risk_net,
        )

    # ------------------------------------------------------------------
    # Vectorized cost application
    # ------------------------------------------------------------------

    def _apply_costs(self, df: pl.DataFrame, config: IndexConfig) -> pl.DataFrame:
        w_cols = [c for c in df.columns if c.startswith("w_")]

        if not w_cols:
            # No weight columns — zero turnover
            return df.with_columns([
                pl.col("excess_return").alias("gross_return"),
                pl.col("excess_return").alias("net_return"),
                pl.lit(0.0).alias("turnover"),
                pl.lit(0.0).alias("cost"),
                pl.col("excess_level").alias("gross_level"),
                pl.col("excess_level").alias("net_level"),
            ])

        # Turnover = 0.5 × Σ |w_i,t - w_i,t-1|  (one-way)
        # Compute per-weight absolute change then sum across columns
        abs_changes = [
            pl.col(c).diff().abs().alias(f"_dw_{c}") for c in w_cols
        ]
        turnover_expr = (
            pl.sum_horizontal([f"_dw_{c}" for c in w_cols]) * 0.5
        ).alias("turnover")

        df = df.with_columns(abs_changes).with_columns(turnover_expr)

        # Cost per day = turnover × (commission_bps + slippage_bps) / 10_000
        total_bps = self._costs.commission.bps + self._costs.slippage.half_spread_bps
        df = df.with_columns(
            (pl.col("turnover") * total_bps / 10_000.0).alias("cost")
        )

        # Net return = gross excess return - daily cost
        df = df.with_columns([
            pl.col("excess_return").alias("gross_return"),
            (pl.col("excess_return") - pl.col("cost")).alias("net_return"),
        ])

        # Compound levels (gross and net) — must be done in Python since Polars
        # scan_csv doesn't support stateful cumulative product with subtraction.
        # Use Polars cum_prod on (1 + r) series.
        base = float(config.base_level)
        gross_level = (pl.col("gross_return") + 1.0).cum_prod() * base
        net_level = (pl.col("net_return") + 1.0).cum_prod() * base

        df = df.with_columns([
            gross_level.alias("gross_level"),
            net_level.alias("net_level"),
        ])

        # Drop the temporary _dw_ columns
        drop_cols = [f"_dw_{c}" for c in w_cols]
        return df.drop(drop_cols)

    @staticmethod
    def _empty_result(name: str, start: date, end: date) -> BacktestResult:
        from qis_engine.backtest.risk import RiskCalculator
        rc = RiskCalculator()
        zero = rc._zero_metrics()
        return BacktestResult(
            config_name=name,
            start=start,
            end=end,
            timeseries=pl.DataFrame(),
            risk_gross=zero,
            risk_net=zero,
        )
