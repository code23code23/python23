from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import IntEnum


class ContractMonth(IntEnum):
    JAN = 1
    FEB = 2
    MAR = 3
    APR = 4
    MAY = 5
    JUN = 6
    JUL = 7
    AUG = 8
    SEP = 9
    OCT = 10
    NOV = 11
    DEC = 12

    @property
    def cme_code(self) -> str:
        codes = "FGHJKMNQUVXZ"
        return codes[self.value - 1]

    @classmethod
    def from_code(cls, code: str) -> "ContractMonth":
        codes = "FGHJKMNQUVXZ"
        return cls(codes.index(code) + 1)


@dataclass(frozen=True)
class FuturesContract:
    """Identifies a single futures contract instance."""
    root: str             # e.g. "CL"
    year: int
    month: ContractMonth

    @property
    def ticker(self) -> str:
        return f"{self.root}{self.month.cme_code}{str(self.year)[-2:]}"

    @property
    def stub_key(self) -> str:
        """Key used in DataStore and CSV stubs: CL_2024_01."""
        return f"{self.root}_{self.year}_{self.month.value:02d}"

    @classmethod
    def from_stub_key(cls, key: str) -> "FuturesContract":
        parts = key.split("_")
        root = parts[0]
        year = int(parts[1])
        month = ContractMonth(int(parts[2]))
        return cls(root=root, year=year, month=month)

    @classmethod
    def front_month(cls, root: str, as_of: date) -> "FuturesContract":
        """Return the front-month contract for a given date (simplistic: same month)."""
        return cls(root=root, year=as_of.year, month=ContractMonth(as_of.month))

    @classmethod
    def nth_deferred(cls, root: str, as_of: date, n: int) -> "FuturesContract":
        """Return the n-th deferred contract (1 = 1 month out, 2 = 2 months out, ...)."""
        month = as_of.month + n
        year = as_of.year
        while month > 12:
            month -= 12
            year += 1
        return cls(root=root, year=year, month=ContractMonth(month))
