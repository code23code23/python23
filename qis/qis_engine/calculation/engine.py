from __future__ import annotations

from datetime import date

import polars as pl

from qis_engine.calculation.calculator import Calculator
from qis_engine.calculation.strategy.bcom import BCOMCalculator
from qis_engine.calculation.strategy.gsci import GSCICalculator
from qis_engine.context.base import CalculationContext
from qis_engine.core.weighting import WeightingConstraints


# ------------------------------------------------------------------
# Calculator registry — index name → Calculator class.
# Instances are created per run by create_calculator(), never cached.
# ------------------------------------------------------------------

_CALCULATOR_REGISTRY: dict[str, type[Calculator]] = {
    "BCOM": BCOMCalculator,
    "GSCI": GSCICalculator,
}

_context: CalculationContext | None = None


def register_calculator(index_name: str, calculator_cls: type[Calculator]) -> None:
    """Add (or replace) the Calculator class used for `index_name`."""
    _CALCULATOR_REGISTRY[index_name] = calculator_cls


def available_calculators() -> list[str]:
    return list(_CALCULATOR_REGISTRY)


def init_engine(context: CalculationContext) -> None:
    """
    Store the application-scoped CalculationContext.
    Call once at startup (server lifespan / batch bootstrap) before any run.
    """
    global _context
    _context = context


def create_calculator(index_name: str) -> Calculator:
    """Create a fresh Calculator for `index_name`, bound to the engine's context."""
    calculator_cls = _CALCULATOR_REGISTRY.get(index_name)
    if calculator_cls is None:
        raise ValueError(
            f"Unknown index: {index_name!r}. Available: {available_calculators()}"
        )
    if _context is None:
        raise RuntimeError("Engine not initialised — call init_engine(context) first")
    return calculator_cls(_context)


def calculate(
    index_name: str,
    start: date,
    end: date,
    constraints: WeightingConstraints | None = None,
) -> pl.DataFrame:
    """Entry point — creates a Calculator for this run and calculates [start, end]."""
    return create_calculator(index_name).calculate(start, end, constraints)
