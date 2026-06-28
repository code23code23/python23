from __future__ import annotations

from datetime import date

import polars as pl

from qis_engine.data.models import ContractSpec, Sector


class CommodityUniverse:
    """
    Manages the set of eligible commodity futures constituents for an index.
    Eligibility rules are applied as-of a given date.
    """

    def __init__(self, specs: list[ContractSpec]) -> None:
        self._specs: dict[str, ContractSpec] = {s.ticker: s for s in specs}

    @classmethod
    def from_provider(cls, provider: "DataProviderBase") -> "CommodityUniverse":  # noqa: F821
        return cls(provider.get_contract_specs())

    def get_all(self) -> list[ContractSpec]:
        return list(self._specs.values())

    def get_spec(self, ticker: str) -> ContractSpec:
        return self._specs[ticker]

    def get_eligible(
        self,
        as_of: date,
        tickers: list[str] | None = None,
        min_history_days: int = 0,
        store: "DataStore | None" = None,  # noqa: F821
    ) -> list[ContractSpec]:
        """
        Return constituents eligible on `as_of`.
        If `tickers` is provided, restrict to that subset.
        If `store` and `min_history_days` provided, filter out tickers
        with insufficient history.
        """
        candidates = (
            [self._specs[t] for t in tickers if t in self._specs]
            if tickers
            else self.get_all()
        )
        if store is None or min_history_days == 0:
            return candidates

        eligible = []
        for spec in candidates:
            ohlcv = store.read_ohlcv(spec.ticker,
                                      date(as_of.year - 5, as_of.month, as_of.day),
                                      as_of)
            if len(ohlcv) >= min_history_days:
                eligible.append(spec)
        return eligible

    def get_by_sector(self, sector: Sector) -> list[ContractSpec]:
        return [s for s in self._specs.values() if s.sector == sector]

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([
            {
                "ticker": s.ticker,
                "name": s.name,
                "sector": s.sector.value,
                "exchange": s.exchange.value,
                "lot_size": s.lot_size,
                "tick_size": s.tick_size,
                "currency": s.currency,
                "settlement": s.settlement.value,
            }
            for s in self._specs.values()
        ])
