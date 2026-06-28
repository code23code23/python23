from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Sector(str, Enum):
    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    LIVESTOCK = "livestock"


class Exchange(str, Enum):
    NYMEX = "NYMEX"
    COMEX = "COMEX"
    CBOT = "CBOT"
    CME = "CME"
    ICE = "ICE"
    LME = "LME"


class Settlement(str, Enum):
    CASH = "cash"
    PHYSICAL = "physical"


# Polars schema constants — used by CSVStubProvider and DataStore
OHLCV_SCHEMA: dict[str, str] = {
    "date": "Date",
    "open": "Float64",
    "high": "Float64",
    "low": "Float64",
    "close": "Float64",
    "volume": "Int64",
    "open_interest": "Int64",
}

OPTIONS_CHAIN_SCHEMA: dict[str, str] = {
    "date": "Date",
    "expiry": "Date",
    "strike": "Float64",
    "price": "Float64",
    "implied_vol": "Float64",
    "delta": "Float64",
    "volume": "Int64",
    "open_interest": "Int64",
}


@dataclass(frozen=True)
class ContractSpec:
    ticker: str
    name: str
    sector: Sector
    exchange: Exchange
    lot_size: float
    tick_size: float
    currency: str
    settlement: Settlement

    @classmethod
    def from_dict(cls, d: dict) -> "ContractSpec":
        return cls(
            ticker=d["ticker"],
            name=d["name"],
            sector=Sector(d["sector"]),
            exchange=Exchange(d["exchange"]),
            lot_size=float(d["lot_size"]),
            tick_size=float(d["tick_size"]),
            currency=d["currency"],
            settlement=Settlement(d["settlement"]),
        )
