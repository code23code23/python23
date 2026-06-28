from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from qis_engine.api.schemas import (
    BacktestRequest, IndexLevelRow, RiskSummaryResponse, WeightRow,
)

router = APIRouter(prefix="/indices", tags=["indices"])


def get_store():
    """Dependency: returns the DataStore singleton injected at app startup."""
    from qis_engine.api.app import get_app_store
    return get_app_store()


def get_calculator_registry():
    """Dependency: returns the application-scoped Calculator registry."""
    from qis_engine.calculation.engine import _CALCULATOR_REGISTRY
    return _CALCULATOR_REGISTRY


@router.get("/")
def list_indices(registry=Depends(get_calculator_registry)) -> list[str]:
    """List available index names."""
    return list(registry.keys())


@router.get("/{index_name}/level", response_model=list[IndexLevelRow])
def get_index_level(
    index_name: str,
    start_date: Annotated[date, Query()],
    end_date: Annotated[date, Query()],
    store=Depends(get_store),
) -> list[dict]:
    from qis_engine.reporting.timeseries_exporter import TimeseriesExporter
    exp = TimeseriesExporter(store)
    rows = exp.levels_to_dicts(index_name, start_date, end_date)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No levels found for {index_name}")
    return rows


@router.get("/{index_name}/weights")
def get_weights(
    index_name: str,
    start_date: Annotated[date, Query()],
    end_date: Annotated[date, Query()],
    store=Depends(get_store),
) -> list[dict]:
    from qis_engine.reporting.timeseries_exporter import TimeseriesExporter
    exp = TimeseriesExporter(store)
    return exp.weights_to_dicts(index_name, start_date, end_date)


@router.get("/{index_name}/roll_schedule")
def get_roll_schedule(
    index_name: str,
    start_date: Annotated[date, Query()],
    end_date: Annotated[date, Query()],
    store=Depends(get_store),
) -> list[dict]:
    import polars as pl
    df = store.read_roll_schedule(index_name)
    if df.is_empty():
        return []
    filtered = df.filter(
        (pl.col("year") >= start_date.year) & (pl.col("year") <= end_date.year)
    )
    return filtered.to_dicts()


@router.post("/{index_name}/backtest", response_model=RiskSummaryResponse)
def run_backtest(
    index_name: str,
    req: BacktestRequest,
    registry=Depends(get_calculator_registry),
    store=Depends(get_store),
) -> RiskSummaryResponse:
    from qis_engine.backtest.costs import CommissionModel, SlippageModel, TransactionCostModel
    from qis_engine.backtest.simulator import VectorizedSimulator
    from qis_engine.calculation.engine import IndexCalculationEngine

    calculator = registry.get(index_name)
    if calculator is None:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not configured")

    # Build an IndexCalculationEngine from the Calculator's owned config and context
    engine = IndexCalculationEngine.from_config(calculator._config, calculator._context)

    cost_model = TransactionCostModel(
        commission=CommissionModel(req.commission_bps),
        slippage=SlippageModel(req.slippage_bps),
    )
    sim = VectorizedSimulator(store, cost_model)
    result = sim.run(engine, req.start_date, req.end_date)
    m = result.risk_net

    return RiskSummaryResponse(
        index_name=index_name,
        start_date=str(req.start_date),
        end_date=str(req.end_date),
        annualised_return=m.annualised_return,
        annualised_vol=m.annualised_vol,
        sharpe=m.sharpe,
        sortino=m.sortino,
        calmar=m.calmar,
        max_drawdown=m.max_drawdown,
        var_95=m.var_95,
        var_99=m.var_99,
        total_return=m.total_return,
        n_days=m.n_days,
    )
