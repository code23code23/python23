from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class ReturnType(str, Enum):
    SPOT = "spot"
    EXCESS = "excess"
    TOTAL = "total"


class RebalanceSchedule(str, Enum):
    ANNUAL = "annual"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class IndexConfig:
    name: str
    base_date: date
    tickers: list[str]
    base_level: float = 100.0
    rebalance_schedule: RebalanceSchedule = RebalanceSchedule.ANNUAL
    collateral_rate: float = 0.0  # annualised; 0 → excess return only
    return_types: list[ReturnType] = field(
        default_factory=lambda: [ReturnType.EXCESS, ReturnType.TOTAL]
    )
    # Month in which annual rebalance is announced (weights become effective following month)
    rebalance_announce_month: int = 1   # January
    weighting_method: str = "equal"     # equal | production | liquidity | risk_parity


# ------------------------------------------------------------------
# Pre-built index configs using 7 stub commodities
# ------------------------------------------------------------------

_STUB_TICKERS = ["CL", "GC", "NG", "HG", "ZC", "ZW", "LC"]


def _bcom_config() -> IndexConfig:
    return IndexConfig(
        name="BCOM",
        base_date=date(2010, 1, 4),
        tickers=_STUB_TICKERS,
        base_level=100.0,
        rebalance_schedule=RebalanceSchedule.ANNUAL,
        collateral_rate=0.0,
        return_types=[ReturnType.SPOT, ReturnType.EXCESS, ReturnType.TOTAL],
        rebalance_announce_month=1,
        weighting_method="liquidity",
    )


def _gsci_config() -> IndexConfig:
    return IndexConfig(
        name="GSCI",
        base_date=date(2010, 1, 4),
        tickers=_STUB_TICKERS,
        base_level=100.0,
        rebalance_schedule=RebalanceSchedule.ANNUAL,
        collateral_rate=0.0,
        return_types=[ReturnType.SPOT, ReturnType.EXCESS, ReturnType.TOTAL],
        rebalance_announce_month=1,
        weighting_method="production",
    )


class BCOMIndex:
    @staticmethod
    def default_config() -> IndexConfig:
        return _bcom_config()


class GSCIIndex:
    @staticmethod
    def default_config() -> IndexConfig:
        return _gsci_config()
