from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from qis_engine.options.pricing import Black76Pricer, OptionPrice


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float    # dDelta/dVol  (finite diff)
    volga: float    # dVega/dVol   (finite diff)


class GreeksCalculator:
    """
    Computes analytical greeks (from optlib Black-76 output) and higher-order
    greeks (vanna, volga) via central finite differences.
    """

    def __init__(self, pricer: Black76Pricer | None = None) -> None:
        self._pricer = pricer or Black76Pricer()

    def compute(
        self,
        option_type: str,
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        american: bool = False,
        bump: float = 1e-4,
    ) -> Greeks:
        """
        Return full Greeks for a single option position.

        Analytical: delta, gamma, theta, vega, rho  (from Black-76 directly).
        Finite-difference: vanna = dDelta/dVol, volga = dVega/dVol.
        """
        base = self._pricer.price(option_type, F, K, T, r, sigma, american)

        # vanna: central diff in sigma
        up = self._pricer.price(option_type, F, K, T, r, sigma + bump, american)
        dn = self._pricer.price(option_type, F, K, T, r, sigma - bump, american)
        vanna = (up.delta - dn.delta) / (2 * bump)
        volga = (up.vega - dn.vega) / (2 * bump)

        return Greeks(
            delta=base.delta,
            gamma=base.gamma,
            theta=base.theta,
            vega=base.vega,
            rho=base.rho,
            vanna=vanna,
            volga=volga,
        )

    def compute_chain(
        self,
        chain: pl.DataFrame,
        F: float,
        r: float,
        as_of: date,
        american: bool = False,
    ) -> pl.DataFrame:
        """
        Compute greeks for each row in an options chain.

        Expected input columns: expiry (date), strike, implied_vol, option_type.
        Returns input DataFrame augmented with greek columns.
        """
        records = []
        for row in chain.to_dicts():
            expiry = row["expiry"]
            if isinstance(expiry, str):
                from datetime import datetime
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            T = max((expiry - as_of).days / 365.0, 0.0)
            K = float(row["strike"])
            sigma = float(row["implied_vol"])
            opt_type = row.get("option_type", "c")
            g = self.compute(opt_type, F, K, T, r, sigma, american)
            records.append({
                **row,
                "delta": g.delta,
                "gamma": g.gamma,
                "theta": g.theta,
                "vega": g.vega,
                "rho": g.rho,
                "vanna": g.vanna,
                "volga": g.volga,
            })
        return pl.DataFrame(records)
