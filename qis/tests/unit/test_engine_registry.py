"""Tests for calculation/engine.py — calculator registry and per-run instantiation."""
from __future__ import annotations

from pathlib import Path

import pytest

from qis_engine.calculation import engine
from qis_engine.calculation.calculator import Calculator
from qis_engine.calculation.strategy.bcom import BCOMCalculator
from qis_engine.context.default import build_default_context

STUBS_DIR = Path(__file__).parent.parent / "stubs"


@pytest.fixture(scope="module")
def context():
    return build_default_context(STUBS_DIR)


@pytest.fixture
def initialised(context, monkeypatch):
    monkeypatch.setattr(engine, "_context", None)
    engine.init_engine(context)
    yield context


class TestRegistry:
    def test_default_calculators_registered(self):
        assert set(engine.available_calculators()) >= {"BCOM", "GSCI"}

    def test_creates_new_instance_per_run(self, initialised):
        first = engine.create_calculator("BCOM")
        second = engine.create_calculator("BCOM")
        assert isinstance(first, BCOMCalculator)
        assert first is not second

    def test_unknown_index_raises(self, initialised):
        with pytest.raises(ValueError, match="Unknown index"):
            engine.create_calculator("NOPE")

    def test_requires_init(self, monkeypatch):
        monkeypatch.setattr(engine, "_context", None)
        with pytest.raises(RuntimeError, match="init_engine"):
            engine.create_calculator("BCOM")

    def test_register_calculator(self, initialised, monkeypatch):
        monkeypatch.setitem(engine._CALCULATOR_REGISTRY, "CUSTOM", BCOMCalculator)
        assert "CUSTOM" in engine.available_calculators()
        assert isinstance(engine.create_calculator("CUSTOM"), Calculator)
