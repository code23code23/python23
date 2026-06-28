from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

# Load optlib/gbs.py directly to avoid the broken optlib __init__.py
# (optlib/__init__.py imports 'requests' which is not a project dep)
_GBS_PATH = Path(__file__).parents[3] / "optlib" / "optlib" / "gbs.py"
_gbs_spec = importlib.util.spec_from_file_location("_optlib_gbs", _GBS_PATH)
_gbs = importlib.util.module_from_spec(_gbs_spec)
_gbs_spec.loader.exec_module(_gbs)


@dataclass(frozen=True)
class OptionPrice:
    value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class Black76Pricer:
    """
    Thin wrapper over optlib.gbs.black_76 / american_76 and their IV solvers.

    All pricing uses the Black-76 model (b=0 cost-of-carry), suitable for
    commodity futures options.
    """

    def price(
        self,
        option_type: str,
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        american: bool = False,
    ) -> OptionPrice:
        """
        Price an option.

        Args:
            option_type: 'c' for call, 'p' for put
            F: futures price
            K: strike price
            T: time to expiry in years
            r: risk-free rate (annualised)
            sigma: implied volatility (annualised)
            american: use American pricing (Barone-Adesi Whaley approximation)

        Returns:
            OptionPrice with value and all greeks.
        """
        if T <= 0:
            intrinsic = max(F - K, 0.0) if option_type == "c" else max(K - F, 0.0)
            return OptionPrice(value=intrinsic, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        if american:
            try:
                v, delta, gamma, theta, vega, rho = _gbs.american_76(option_type, F, K, T, r, sigma)
                return OptionPrice(value=v, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
            except Exception:
                # Fallback to European if American solver unavailable (e.g. missing statsmodels)
                pass
        v, delta, gamma, theta, vega, rho = _gbs.black_76(option_type, F, K, T, r, sigma)
        return OptionPrice(value=v, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    def implied_vol(
        self,
        option_type: str,
        F: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        american: bool = False,
    ) -> float:
        """
        Compute implied volatility from a market price via bisection.

        Returns NaN if the solver cannot converge (e.g. deep ITM/OTM with no
        time value, or price below intrinsic).
        """
        if T <= 0:
            return float("nan")
        try:
            if american:
                try:
                    return float(_gbs.amer_implied_vol_76(option_type, F, K, T, r, market_price))
                except Exception:
                    pass
            return float(_gbs.euro_implied_vol_76(option_type, F, K, T, r, market_price))
        except Exception:
            return float("nan")

    def price_chain(
        self,
        chain: pl.DataFrame,
        F: float,
        r: float,
        as_of: date,
        american: bool = False,
    ) -> pl.DataFrame:
        """
        Vectorised pricing of an options chain DataFrame.

        Expected input columns: expiry (date), strike, implied_vol.
        Added output columns: model_price, model_delta, model_gamma, model_vega, model_theta.
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
            p = self.price(opt_type, F, K, T, r, sigma, american)
            records.append({
                **row,
                "model_price": p.value,
                "model_delta": p.delta,
                "model_gamma": p.gamma,
                "model_vega": p.vega,
                "model_theta": p.theta,
            })
        return pl.DataFrame(records)
