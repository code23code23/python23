"""Tests for the Calculator → CalculatorBase → strategy hierarchy."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qis_engine.calculation.calculator import Calculator
from qis_engine.calculation.calculator_base import CalculatorBase
from qis_engine.calculation.strategy import BCOMCalculator, ConfiguredCalculator, GSCICalculator
from qis_engine.context.default import build_default_context
from qis_engine.core.index import BCOMIndex

STUBS_DIR = Path(__file__).parent.parent / "stubs"
START, END = date(2022, 1, 3), date(2022, 3, 31)


@pytest.fixture(scope="module")
def context():
    return build_default_context(STUBS_DIR)


class TestAbstractness:
    def test_calculator_is_abstract(self, context):
        with pytest.raises(TypeError):
            Calculator(context, BCOMIndex.default_config())

    def test_calculator_base_is_abstract(self, context):
        with pytest.raises(TypeError):
            CalculatorBase(context, BCOMIndex.default_config())

    def test_subclass_missing_hook_is_abstract(self, context):
        class WeightsOnly(CalculatorBase):
            def _compute_weights(self, eligible, as_of, constraints):
                return {s.ticker: 1.0 / len(eligible) for s in eligible}

        with pytest.raises(TypeError):
            WeightsOnly(context, BCOMIndex.default_config())


class TestStrategies:
    @pytest.mark.parametrize("cls", [BCOMCalculator, GSCICalculator])
    def test_concrete_strategies_are_calculators(self, context, cls):
        calc = cls(context)
        assert isinstance(calc, CalculatorBase)
        assert isinstance(calc, Calculator)

    def test_bcom_and_gsci_weights_differ(self, context):
        bcom = BCOMCalculator(context).calculate(START, END)
        gsci = GSCICalculator(context).calculate(START, END)
        w_cols = [c for c in bcom.columns if c.startswith("w_")]
        assert bcom.select(w_cols).row(0) != gsci.select(w_cols).row(0)

    def test_bcom_matches_config_driven_equivalent(self, context):
        expected = ConfiguredCalculator(context, BCOMIndex.default_config()).calculate(START, END)
        assert BCOMCalculator(context).calculate(START, END).equals(expected)

    def test_hooks_drive_the_template(self, context):
        class FlatRoll(ConfiguredCalculator):
            def _compute_roll_return(self, dt, weights, today_prices):
                return 0.0

        df = FlatRoll(context, BCOMIndex.default_config()).calculate(START, END)
        assert (df["roll_return"] == 0.0).all()
        assert df["excess_return"].equals(df["spot_return"], check_names=False)
