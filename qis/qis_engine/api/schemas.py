from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class IndexLevelRow(BaseModel):
    date: str
    spot_level: float | None = None
    excess_level: float | None = None
    total_level: float | None = None
    spot_return: float | None = None
    excess_return: float | None = None
    total_return: float | None = None


class WeightRow(BaseModel):
    date: str
    weights: dict[str, float]


class RollEventRow(BaseModel):
    ticker: str
    year: int
    month: int
    roll_start: str
    roll_end: str
    from_contract: str
    to_contract: str


class BacktestRequest(BaseModel):
    index_name: str = Field(..., description="'BCOM' or 'GSCI'")
    start_date: date
    end_date: date
    commission_bps: float = Field(0.5, ge=0.0)
    slippage_bps: float = Field(0.1, ge=0.0)


class RiskSummaryResponse(BaseModel):
    index_name: str
    start_date: str
    end_date: str
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    var_95: float
    var_99: float
    total_return: float
    n_days: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
