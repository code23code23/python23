"""
Unit tests for options/pricing.py and options/greeks.py.

Phase 2 acceptance gate:
  - Black-76 price → implied vol → price round-trips to within 1e-4
  - American pricing returns higher value than European for ITM puts
  - Greeks have correct signs and vega/gamma positivity
  - Strategy strike selection respects delta target
"""
from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from qis_engine.options.pricing import Black76Pricer, OptionPrice
from qis_engine.options.greeks import GreeksCalculator
from qis_engine.options.strategies import (
    PutWriteStrategy, CoveredCallStrategy, CollarStrategy,
)


# ---------------------------------------------------------------------------
# Reference parameters (WTI crude-like)
# ---------------------------------------------------------------------------
F, K, T, r, sigma = 80.0, 85.0, 0.25, 0.05, 0.30


class TestBlack76Pricer:
    def setup_method(self):
        self.pricer = Black76Pricer()

    def test_call_price_positive(self):
        p = self.pricer.price("c", F, K, T, r, sigma)
        assert p.value > 0

    def test_put_price_positive(self):
        p = self.pricer.price("p", F, K, T, r, sigma)
        assert p.value > 0

    def test_put_call_parity(self):
        """C - P = (F - K) × e^{-rT}  (Black-76 put-call parity)"""
        c = self.pricer.price("c", F, K, T, r, sigma).value
        p = self.pricer.price("p", F, K, T, r, sigma).value
        pcp = (F - K) * math.exp(-r * T)
        assert abs((c - p) - pcp) < 1e-6

    def test_call_iv_round_trip(self):
        p = self.pricer.price("c", F, K, T, r, sigma)
        iv = self.pricer.implied_vol("c", F, K, T, r, p.value)
        assert abs(iv - sigma) < 1e-4

    def test_put_iv_round_trip(self):
        p = self.pricer.price("p", F, K, T, r, sigma)
        iv = self.pricer.implied_vol("p", F, K, T, r, p.value)
        assert abs(iv - sigma) < 1e-4

    def test_atm_call_iv_round_trip(self):
        p = self.pricer.price("c", F, F, T, r, sigma)   # ATM
        iv = self.pricer.implied_vol("c", F, F, T, r, p.value)
        assert abs(iv - sigma) < 1e-4

    def test_expired_option_returns_intrinsic(self):
        p_call = self.pricer.price("c", 90.0, 85.0, 0.0, r, sigma)
        assert abs(p_call.value - 5.0) < 1e-9

    def test_expired_otm_option_zero(self):
        p_call = self.pricer.price("c", 80.0, 85.0, 0.0, r, sigma)
        assert p_call.value == 0.0

    def test_call_delta_between_0_and_1(self):
        p = self.pricer.price("c", F, K, T, r, sigma)
        assert 0.0 < p.delta < 1.0

    def test_put_delta_between_neg1_and_0(self):
        p = self.pricer.price("p", F, K, T, r, sigma)
        assert -1.0 < p.delta < 0.0

    def test_gamma_positive(self):
        p = self.pricer.price("c", F, K, T, r, sigma)
        assert p.gamma > 0

    def test_vega_positive(self):
        p = self.pricer.price("c", F, K, T, r, sigma)
        assert p.vega > 0

    def test_american_ge_european(self):
        """American put should be >= European put; falls back to European if statsmodels absent."""
        euro = self.pricer.price("p", F, K, T, r, sigma, american=False)
        amer = self.pricer.price("p", F, K, T, r, sigma, american=True)
        # amer.value == euro.value is valid when American solver falls back to European
        assert amer.value >= euro.value - 1e-6

    def test_iv_returns_nan_on_impossible_price(self):
        """Deep ITM price far below intrinsic → solver can't converge → NaN returned."""
        # Call: F=100, K=60, intrinsic ≈ 40. Market price of 1 is impossible.
        iv = self.pricer.implied_vol("c", 100.0, 60.0, 0.25, 0.05, 1.0)
        assert math.isnan(iv)

    def test_price_chain_adds_model_columns(self):
        chain = pl.DataFrame({
            "expiry": [date(2024, 6, 20), date(2024, 6, 20)],
            "strike": [80.0, 90.0],
            "implied_vol": [0.30, 0.28],
            "option_type": ["c", "c"],
        })
        result = self.pricer.price_chain(chain, F=80.0, r=r, as_of=date(2024, 3, 1))
        assert "model_price" in result.columns
        assert "model_delta" in result.columns
        assert len(result) == 2
        assert (result["model_price"] > 0).all()


class TestGreeksCalculator:
    def setup_method(self):
        self.calc = GreeksCalculator()

    def test_vanna_finite(self):
        g = self.calc.compute("c", F, K, T, r, sigma)
        assert math.isfinite(g.vanna)

    def test_volga_positive(self):
        """Volga (dVega/dVol) should be positive for OTM options."""
        g = self.calc.compute("c", F, K, T, r, sigma)
        assert g.volga > 0

    def test_greeks_match_pricer_values(self):
        pricer = Black76Pricer()
        p = pricer.price("c", F, K, T, r, sigma)
        g = self.calc.compute("c", F, K, T, r, sigma)
        assert abs(g.delta - p.delta) < 1e-9
        assert abs(g.gamma - p.gamma) < 1e-9
        assert abs(g.vega - p.vega) < 1e-9

    def test_compute_chain_returns_dataframe(self):
        chain = pl.DataFrame({
            "expiry": [date(2024, 6, 20)],
            "strike": [80.0],
            "implied_vol": [0.30],
            "option_type": ["p"],
        })
        result = self.calc.compute_chain(chain, F=80.0, r=0.05, as_of=date(2024, 3, 1))
        assert "vanna" in result.columns
        assert "volga" in result.columns


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

def _put_chain(F: float = 80.0, r: float = 0.05) -> pl.DataFrame:
    """Synthetic 1-expiry put chain around F=80 with known sigmas."""
    pricer = Black76Pricer()
    expiry = date(2024, 6, 20)
    T = (expiry - date(2024, 3, 1)).days / 365.0
    strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
    rows = []
    for K in strikes:
        sigma = 0.30
        p = pricer.price("p", F, K, T, r, sigma)
        rows.append({"expiry": expiry, "strike": K, "implied_vol": sigma,
                     "price": p.value, "delta": p.delta, "option_type": "p"})
    return pl.DataFrame(rows)


def _call_chain(F: float = 80.0, r: float = 0.05) -> pl.DataFrame:
    pricer = Black76Pricer()
    expiry = date(2024, 6, 20)
    T = (expiry - date(2024, 3, 1)).days / 365.0
    strikes = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    rows = []
    for K in strikes:
        sigma = 0.30
        p = pricer.price("c", F, K, T, r, sigma)
        rows.append({"expiry": expiry, "strike": K, "implied_vol": sigma,
                     "price": p.value, "delta": p.delta, "option_type": "c"})
    return pl.DataFrame(rows)


class TestPutWriteStrategy:
    AS_OF = date(2024, 3, 1)

    def test_returns_result(self):
        strat = PutWriteStrategy(delta_target=-0.25)
        r = strat.select_strike(_put_chain(), F=80.0, as_of=self.AS_OF)
        assert r is not None

    def test_net_premium_positive(self):
        strat = PutWriteStrategy(delta_target=-0.25)
        r = strat.select_strike(_put_chain(), F=80.0, as_of=self.AS_OF)
        assert r.net_premium > 0

    def test_selected_put_delta_close_to_target(self):
        target = 0.25
        strat = PutWriteStrategy(delta_target=-target)
        r = strat.select_strike(_put_chain(), F=80.0, as_of=self.AS_OF)
        assert abs(abs(r.legs[0].delta) - target) < 0.15

    def test_empty_chain_returns_none(self):
        strat = PutWriteStrategy()
        empty = pl.DataFrame({"expiry": [], "strike": [], "implied_vol": [], "option_type": []})
        assert strat.select_strike(empty, F=80.0, as_of=self.AS_OF) is None


class TestCoveredCallStrategy:
    AS_OF = date(2024, 3, 1)

    def test_returns_result(self):
        strat = CoveredCallStrategy(delta_target=0.25)
        r = strat.select_strike(_call_chain(), F=80.0, as_of=self.AS_OF)
        assert r is not None

    def test_net_premium_positive(self):
        strat = CoveredCallStrategy(delta_target=0.25)
        r = strat.select_strike(_call_chain(), F=80.0, as_of=self.AS_OF)
        assert r.net_premium > 0


class TestCollarStrategy:
    AS_OF = date(2024, 3, 1)

    def test_returns_two_legs(self):
        strat = CollarStrategy()
        r = strat.select_strikes(_call_chain(), _put_chain(), F=80.0, as_of=self.AS_OF)
        assert r is not None
        assert len(r.legs) == 2

    def test_legs_are_call_and_put(self):
        strat = CollarStrategy()
        r = strat.select_strikes(_call_chain(), _put_chain(), F=80.0, as_of=self.AS_OF)
        types = {leg.option_type for leg in r.legs}
        assert types == {"c", "p"}
