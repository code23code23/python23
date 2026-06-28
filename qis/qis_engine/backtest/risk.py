from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class RiskMetrics:
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float         # negative number, e.g. -0.25 = -25%
    var_95: float               # 1-day 95% VaR (negative)
    var_99: float               # 1-day 99% VaR (negative)
    cvar_95: float              # Expected Shortfall at 95%
    cvar_99: float              # Expected Shortfall at 99%
    total_return: float
    n_days: int


class RiskCalculator:
    """
    Computes risk metrics from a daily returns Polars Series.
    All metrics use annualisation factor of 252 business days.
    """

    TRADING_DAYS = 252

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self._rf = risk_free_rate

    def compute(self, returns: pl.Series, level: pl.Series | None = None) -> RiskMetrics:
        """
        Args:
            returns: daily return series (e.g. excess_return column)
            level:   optional cumulative level series for drawdown calc
        """
        r = returns.drop_nulls().cast(pl.Float64)
        n = len(r)
        if n == 0:
            return self._zero_metrics()

        mean_daily = float(r.mean())
        std_daily = float(r.std()) if n > 1 else 0.0

        ann_return = (1.0 + mean_daily) ** self.TRADING_DAYS - 1.0
        ann_vol = std_daily * math.sqrt(self.TRADING_DAYS)

        sharpe = (ann_return - self._rf) / ann_vol if ann_vol > 0 else 0.0

        # Sortino: downside deviation only
        downside = r.filter(r < 0)
        dd_std = float(downside.std()) * math.sqrt(self.TRADING_DAYS) if len(downside) > 1 else 0.0
        sortino = (ann_return - self._rf) / dd_std if dd_std > 0 else 0.0

        # Max drawdown from level series or computed from returns
        mdd = self._max_drawdown(level, r)

        calmar = abs(ann_return / mdd) if mdd != 0 else 0.0

        # VaR / CVaR (historical)
        sorted_r = r.sort()
        idx_95 = max(0, int(math.floor(n * 0.05)) - 1)
        idx_99 = max(0, int(math.floor(n * 0.01)) - 1)
        var_95 = float(sorted_r[idx_95])
        var_99 = float(sorted_r[idx_99])
        cvar_95 = float(sorted_r[:idx_95 + 1].mean()) if idx_95 >= 0 else var_95
        cvar_99 = float(sorted_r[:idx_99 + 1].mean()) if idx_99 >= 0 else var_99

        total_ret = float((r + 1.0).product()) - 1.0

        return RiskMetrics(
            annualised_return=ann_return,
            annualised_vol=ann_vol,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=mdd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            total_return=total_ret,
            n_days=n,
        )

    def _max_drawdown(self, level: pl.Series | None, returns: pl.Series) -> float:
        if level is not None and len(level) > 0:
            lv = level.cast(pl.Float64)
        else:
            lv = (returns + 1.0).cum_prod()

        peaks = lv.cum_max()
        drawdowns = (lv - peaks) / peaks
        return float(drawdowns.min())

    @staticmethod
    def _zero_metrics() -> RiskMetrics:
        return RiskMetrics(
            annualised_return=0.0, annualised_vol=0.0, sharpe=0.0, sortino=0.0,
            calmar=0.0, max_drawdown=0.0, var_95=0.0, var_99=0.0,
            cvar_95=0.0, cvar_99=0.0, total_return=0.0, n_days=0,
        )

    def summary_df(self, metrics: RiskMetrics) -> pl.DataFrame:
        """Return metrics as a single-row Polars DataFrame."""
        return pl.DataFrame([{
            "annualised_return": metrics.annualised_return,
            "annualised_vol": metrics.annualised_vol,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "calmar": metrics.calmar,
            "max_drawdown": metrics.max_drawdown,
            "var_95": metrics.var_95,
            "var_99": metrics.var_99,
            "cvar_95": metrics.cvar_95,
            "cvar_99": metrics.cvar_99,
            "total_return": metrics.total_return,
            "n_days": metrics.n_days,
        }])
