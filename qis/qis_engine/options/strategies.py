from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from qis_engine.options.pricing import Black76Pricer, OptionPrice


@dataclass
class StrategyPosition:
    """One leg of an options overlay strategy."""
    option_type: str        # 'c' or 'p'
    direction: int          # +1 long, -1 short
    K: float
    T: float
    sigma: float
    price: float
    delta: float


@dataclass
class StrategyResult:
    date: date
    F: float
    legs: list[StrategyPosition]
    net_premium: float      # cash received (positive = credit)
    net_delta: float
    pnl: float = 0.0        # realised on expiry / close


class PutWriteStrategy:
    """
    Systematic short put overlay (cash-secured put write).

    Sells the nearest OTM put at or below `delta_target` on each roll date,
    collected premium compounds into total return.
    """

    def __init__(
        self,
        delta_target: float = -0.25,
        r: float = 0.05,
        pricer: Black76Pricer | None = None,
    ) -> None:
        self._delta = abs(delta_target)
        self._r = r
        self._pricer = pricer or Black76Pricer()

    def select_strike(
        self,
        chain: pl.DataFrame,
        F: float,
        as_of: date,
    ) -> StrategyResult | None:
        """
        Choose the put strike whose absolute delta is closest to delta_target.

        chain must have columns: expiry, strike, implied_vol.
        Returns None if no suitable option found.
        """
        puts = chain.filter(pl.col("option_type") == "p") if "option_type" in chain.columns else chain
        if puts.is_empty():
            return None

        best: dict | None = None
        best_dist = float("inf")
        best_price: OptionPrice | None = None

        for row in puts.to_dicts():
            expiry = row["expiry"]
            if isinstance(expiry, str):
                from datetime import datetime
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            T = max((expiry - as_of).days / 365.0, 1e-6)
            K = float(row["strike"])
            sigma = float(row["implied_vol"])
            p = self._pricer.price("p", F, K, T, self._r, sigma)
            dist = abs(abs(p.delta) - self._delta)
            if dist < best_dist:
                best_dist = dist
                best = {**row, "_T": T, "_K": K, "_sigma": sigma}
                best_price = p

        if best is None or best_price is None:
            return None

        expiry = best["expiry"]
        if isinstance(expiry, str):
            from datetime import datetime
            expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = best["_T"]

        leg = StrategyPosition(
            option_type="p",
            direction=-1,
            K=best["_K"],
            T=T,
            sigma=best["_sigma"],
            price=best_price.value,
            delta=best_price.delta,
        )
        return StrategyResult(
            date=as_of,
            F=F,
            legs=[leg],
            net_premium=best_price.value,   # credit received
            net_delta=-best_price.delta,    # short put = long delta
        )


class CoveredCallStrategy:
    """
    Systematic covered call overlay (short call against long futures position).
    Sells the nearest OTM call at or above `delta_target`.
    """

    def __init__(
        self,
        delta_target: float = 0.25,
        r: float = 0.05,
        pricer: Black76Pricer | None = None,
    ) -> None:
        self._delta = abs(delta_target)
        self._r = r
        self._pricer = pricer or Black76Pricer()

    def select_strike(
        self,
        chain: pl.DataFrame,
        F: float,
        as_of: date,
    ) -> StrategyResult | None:
        calls = chain.filter(pl.col("option_type") == "c") if "option_type" in chain.columns else chain
        if calls.is_empty():
            return None

        best: dict | None = None
        best_dist = float("inf")
        best_price: OptionPrice | None = None

        for row in calls.to_dicts():
            expiry = row["expiry"]
            if isinstance(expiry, str):
                from datetime import datetime
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            T = max((expiry - as_of).days / 365.0, 1e-6)
            K = float(row["strike"])
            sigma = float(row["implied_vol"])
            p = self._pricer.price("c", F, K, T, self._r, sigma)
            dist = abs(p.delta - self._delta)
            if dist < best_dist:
                best_dist = dist
                best = {**row, "_T": T, "_K": K, "_sigma": sigma}
                best_price = p

        if best is None or best_price is None:
            return None

        leg = StrategyPosition(
            option_type="c",
            direction=-1,
            K=best["_K"],
            T=best["_T"],
            sigma=best["_sigma"],
            price=best_price.value,
            delta=best_price.delta,
        )
        return StrategyResult(
            date=as_of,
            F=F,
            legs=[leg],
            net_premium=best_price.value,
            net_delta=-best_price.delta,
        )


class CollarStrategy:
    """
    Collar: short call at call_delta_target + long put at put_delta_target.
    Net premium is (call credit − put debit).
    """

    def __init__(
        self,
        call_delta_target: float = 0.25,
        put_delta_target: float = -0.25,
        r: float = 0.05,
        pricer: Black76Pricer | None = None,
    ) -> None:
        self._call_strat = CoveredCallStrategy(call_delta_target, r, pricer)
        self._put_strat = PutWriteStrategy(put_delta_target, r, pricer)
        self._pricer = pricer or Black76Pricer()

    def select_strikes(
        self,
        call_chain: pl.DataFrame,
        put_chain: pl.DataFrame,
        F: float,
        as_of: date,
    ) -> StrategyResult | None:
        call_result = self._call_strat.select_strike(call_chain, F, as_of)
        put_result = self._put_strat.select_strike(put_chain, F, as_of)

        if call_result is None or put_result is None:
            return None

        call_leg = call_result.legs[0]
        put_leg = put_result.legs[0]

        # For the collar put leg we are LONG the put (hedge), not short
        put_leg_long = StrategyPosition(
            option_type="p",
            direction=+1,           # long put
            K=put_leg.K,
            T=put_leg.T,
            sigma=put_leg.sigma,
            price=put_leg.price,
            delta=put_leg.delta,
        )

        net_premium = call_leg.price - put_leg.price  # call credit − put debit
        net_delta = -call_leg.delta + put_leg.delta   # short call − long put (put delta is negative)

        return StrategyResult(
            date=as_of,
            F=F,
            legs=[call_leg, put_leg_long],
            net_premium=net_premium,
            net_delta=net_delta,
        )
