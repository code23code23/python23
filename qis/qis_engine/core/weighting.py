from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from qis_engine.data.models import ContractSpec


@dataclass
class WeightingConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_sector_weight: float = 0.0
    max_sector_weight: float = 1.0


class WeightingEngine:
    """
    Dispatches to the appropriate weighting strategy based on method name.
    All methods return a pl.Series indexed by ticker with weights summing to 1.0.
    """

    def __init__(
        self,
        method: str,
        provider: "DataProviderBase | None" = None,  # noqa: F821
    ) -> None:
        self._method = method
        self._provider = provider
        self._strategies: dict[str, "_WeightingStrategy"] = {
            "equal": _EqualWeight(),
            "production": _ProductionWeight(provider),
            "liquidity": _LiquidityWeight(provider),
            "risk_parity": _RiskParity(),
        }

    def compute_weights(
        self,
        eligible: list[ContractSpec],
        as_of: date,
        store: "DataStore",  # noqa: F821
        constraints: WeightingConstraints | None = None,
    ) -> pl.Series:
        if not eligible:
            return pl.Series("weight", [], dtype=pl.Float64)
        constraints = constraints or WeightingConstraints()
        strategy = self._strategies[self._method]
        weights = strategy.compute(eligible, as_of, store)
        return _apply_constraints(weights, eligible, constraints)


# ------------------------------------------------------------------
# Strategy implementations
# ------------------------------------------------------------------

class _WeightingStrategy:
    def compute(
        self,
        eligible: list[ContractSpec],
        as_of: date,
        store: "DataStore",  # noqa: F821
    ) -> dict[str, float]:
        raise NotImplementedError


class _EqualWeight(_WeightingStrategy):
    def compute(self, eligible, as_of, store):
        n = len(eligible)
        return {s.ticker: 1.0 / n for s in eligible}


class _ProductionWeight(_WeightingStrategy):
    """GSCI-style: weights proportional to world production averages."""

    def __init__(self, provider) -> None:
        self._provider = provider
        self._weights_df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        if self._weights_df is None and self._provider is not None:
            self._weights_df = self._provider.get_production_weights()
        return self._weights_df if self._weights_df is not None else pl.DataFrame()

    def compute(self, eligible, as_of, store):
        df = self._load()
        tickers = {s.ticker for s in eligible}
        if df.is_empty():
            # fallback to equal weight if no production data
            n = len(eligible)
            return {s.ticker: 1.0 / n for s in eligible}

        year = as_of.year
        subset = df.filter(
            (pl.col("year") == year) & (pl.col("ticker").is_in(list(tickers)))
        )
        if subset.is_empty():
            # try previous years
            subset = df.filter(pl.col("ticker").is_in(list(tickers))).sort("year").tail(
                len(tickers) * 5
            )

        total = subset["weight"].sum()
        if total == 0:
            n = len(eligible)
            return {s.ticker: 1.0 / n for s in eligible}
        return {row["ticker"]: row["weight"] / total for row in subset.to_dicts()}


class _LiquidityWeight(_WeightingStrategy):
    """
    BCOM-style: 1/3 production weight + 2/3 liquidity weight.
    Liquidity proxy = average daily volume × close price (dollar volume) over lookback.
    """

    def __init__(self, provider) -> None:
        self._production = _ProductionWeight(provider)
        self._lookback_days = 252  # 1 year of daily volume data

    def compute(self, eligible, as_of, store):
        from datetime import timedelta
        prod_w = self._production.compute(eligible, as_of, store)

        # Liquidity weight from dollar volume
        liq: dict[str, float] = {}
        lookback_start = date(as_of.year - 1, as_of.month, as_of.day)
        for spec in eligible:
            df = store.read_ohlcv(spec.ticker, lookback_start, as_of)
            if df.is_empty():
                liq[spec.ticker] = 0.0
            else:
                dollar_vol = (df["volume"].cast(pl.Float64) * df["close"]).mean()
                liq[spec.ticker] = float(dollar_vol)

        total_liq = sum(liq.values()) or 1.0
        liq_w = {t: v / total_liq for t, v in liq.items()}

        # Blend: 1/3 production + 2/3 liquidity
        tickers = [s.ticker for s in eligible]
        blended: dict[str, float] = {}
        for t in tickers:
            blended[t] = (1 / 3) * prod_w.get(t, 0.0) + (2 / 3) * liq_w.get(t, 0.0)

        total = sum(blended.values()) or 1.0
        return {t: v / total for t, v in blended.items()}


class _RiskParity(_WeightingStrategy):
    """
    Inverse-volatility weights, optionally correlation-adjusted (1/σ_i / Σ 1/σ_j).
    Uses 1-year rolling annualised volatility of daily returns.
    """

    def __init__(self, lookback_days: int = 252) -> None:
        self._lookback = lookback_days

    def compute(self, eligible, as_of, store):
        from datetime import timedelta
        import math
        start = date(as_of.year - 1, as_of.month, as_of.day)
        inv_vol: dict[str, float] = {}
        for spec in eligible:
            df = store.read_ohlcv(spec.ticker, start, as_of)
            if len(df) < 5:
                inv_vol[spec.ticker] = 0.0
                continue
            rets = df["close"].pct_change().drop_nulls()
            vol = float(rets.std()) * math.sqrt(252)
            inv_vol[spec.ticker] = 1.0 / vol if vol > 0 else 0.0

        total = sum(inv_vol.values()) or 1.0
        return {t: v / total for t, v in inv_vol.items()}


# ------------------------------------------------------------------
# Constraint application
# ------------------------------------------------------------------

def _apply_constraints(
    raw: dict[str, float],
    eligible: list[ContractSpec],
    c: WeightingConstraints,
) -> pl.Series:
    tickers = [s.ticker for s in eligible]
    weights = {t: raw.get(t, 0.0) for t in tickers}
    # Iterative clip-and-renormalize until all weights satisfy bounds
    for _ in range(50):
        weights = {t: max(c.min_weight, min(c.max_weight, w)) for t, w in weights.items()}
        total = sum(weights.values()) or 1.0
        weights = {t: w / total for t, w in weights.items()}
        if all(c.min_weight - 1e-10 <= w <= c.max_weight + 1e-10 for w in weights.values()):
            break
    return pl.Series(
        name="weight",
        values=[weights[t] for t in tickers],
        dtype=pl.Float64,
    ).alias("weight")
