# QIS Engine — Coding Style

## Language and runtime

- Python 3.10+ (`requires-python = ">=3.10"`)
- Every module starts with `from __future__ import annotations` to enable PEP 563 postponed
  evaluation. This allows forward references in type hints without quoting them *most* of the
  time (the exception is string-quoted forward refs inside abstract method signatures where the
  referenced class is defined in another module).

---

## Tooling

| Tool | Config location | Key settings |
|------|----------------|-------------|
| **Ruff** | `pyproject.toml [tool.ruff]` | line-length 100, target py310, rules E/F/I/UP |
| **mypy** | `pyproject.toml [tool.mypy]` | non-strict, `ignore_missing_imports = true` |
| **pytest** | `pyproject.toml [tool.pytest.ini_options]` | `testpaths = ["tests"]`, pattern `test_*.py` |

Run linting and type checks before committing:

```bash
ruff check qis_engine tests
mypy qis_engine
pytest
```

---

## Type hints

Use type hints on all function signatures and class attributes.

```python
# correct
def get_eligible(self, as_of: date, tickers: list[str] | None = None) -> list[ContractSpec]:

# wrong — bare Any, missing return type
def get_eligible(self, as_of, tickers=None):
```

Use `X | None` (PEP 604 union) not `Optional[X]`.

Use `X | Y` unions at call sites. Use `from __future__ import annotations` so the pipe syntax
works on Python 3.10+ without `from typing import Union`.

Forward-reference types that would cause circular imports as quoted strings:

```python
def compute_weights(self, store: "DataStore", ...) -> pl.Series:
```

Add `# noqa: F821` on the same line when the forward ref is in an abstract or interface method
where the concrete class lives in a different module.

---

## Data structures

### Dataclasses

Use `@dataclass` for all structured data types. Use `frozen=True` for value objects
(things that should be compared by value and never mutated after creation).

```python
@dataclass(frozen=True)
class ContractSpec:          # value object — immutable
    ticker: str
    ...

@dataclass
class IndexConfig:           # config — may be mutated before passing to the engine
    name: str
    ...
```

### Enums

All enums inherit from `(str, Enum)` so they serialise to their string value directly
and compare equal to plain strings.

```python
class Sector(str, Enum):
    ENERGY = "energy"
    METALS = "metals"
```

Never use `int` enums — the string form is always readable in CSV/JSON output.

---

## Polars

All time-series and tabular data uses [Polars](https://pola.rs/), not pandas.

### Schemas

Shared column schemas are defined as module-level constants in `data/models.py`:

```python
OHLCV_SCHEMA: dict[str, str] = {
    "date": "Date",
    "open": "Float64",
    ...
}
```

Use these constants in providers and tests to avoid schema drift.

### Expressions vs row iteration

Prefer Polars expressions for bulk operations. Use `.to_dicts()` / row iteration only
for operations that are inherently sequential or require calling external functions
(e.g. pricing each row of an options chain):

```python
# preferred — vectorized
turnover = pl.sum_horizontal([f"_dw_{c}" for c in w_cols]) * 0.5

# acceptable — row-by-row when calling Black76Pricer
for row in chain.to_dicts():
    p = pricer.price(...)
```

### Filtering

Use `pl.lit()` when comparing against Python `date` objects in filter expressions:

```python
ohlcv.filter(
    (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
)
```

---

## Class design

### Interfaces (ABCs)

Data-source abstractions use `abc.ABC` with `@abstractmethod`. The single production
implementation is `CSVStubProvider`; external adapters implement the same interface.

```python
class DataProviderBase(ABC):
    @abstractmethod
    def get_ohlcv(self, ticker: str, start: date, end: date) -> pl.DataFrame: ...
```

### Strategy pattern

Weighting methods are strategy objects with a single `compute()` method. Private strategy
classes use a leading underscore and are never exported:

```python
class _EqualWeight(_WeightingStrategy):
    def compute(self, eligible, as_of, store) -> dict[str, float]: ...
```

The dispatcher (`WeightingEngine`) registers strategies by string key in `__init__` and
selects at call time. Adding a new strategy means adding a class and one entry in the dict.

### Factory methods

Named constructors follow the `from_<source>` convention:

```python
@classmethod
def from_config(cls, config: IndexConfig, store: DataStore, ...) -> "IndexCalculationEngine":

@classmethod
def from_provider(cls, provider: DataProviderBase) -> "CommodityUniverse":

@classmethod
def from_dict(cls, d: dict) -> "ContractSpec":
```

### Private helpers

Internal methods and module-level helpers that are not part of the public API use a leading
underscore:

```python
def _compute_roll_return(self, ...) -> float: ...   # instance helper
def _apply_constraints(...) -> pl.Series: ...       # module-level helper
def _parse_cme_code(code: str, ticker: str) -> tuple[int, int]: ...
```

---

## Comments

Write no comments by default. Add a comment only when the *why* is non-obvious:

```python
# skip weekends for prev
while prev.weekday() >= 5:
    prev -= timedelta(days=1)
```

Do not explain what the code does — well-named functions and variables do that.
Do not add docstrings to internal/private methods. Public methods on ABCs and major
engine classes may have a one-paragraph docstring stating contract and schema expectations.

---

## Naming

| Pattern | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `roll_cost.py` |
| Classes | `PascalCase` | `RollCalendar`, `Black76Pricer` |
| Private classes | `_PascalCase` | `_LiquidityWeight` |
| Functions / methods | `snake_case` | `compute_weights()` |
| Private functions / methods | `_snake_case` | `_apply_constraints()` |
| Constants | `UPPER_SNAKE` | `OHLCV_SCHEMA`, `TRADING_DAYS` |
| CME month-code map | uppercase key | `_CME_MONTH = {'F': 1, ...}` |
| Weight columns in DataFrames | `w_<ticker>` | `w_CL`, `w_GC` |
| Temporary diff columns | `_dw_<col>` | `_dw_w_CL` (dropped after use) |

---

## Error handling

Do not add `try/except` for scenarios that can't happen in normal operation.
Use it at system boundaries where external data or optional deps may fail:

```python
# acceptable — optlib American solver may be unavailable
try:
    v, delta, ... = _gbs.american_76(...)
except Exception:
    pass  # fallback to European
```

Return `None` or an empty DataFrame rather than raising when a lookup yields no data:

```python
def get_front_close(self, dt: date) -> float | None:
    row = self._store.read_ohlcv_latest(self._root, dt)
    if row.is_empty():
        return None
    return float(row["close"][0])
```

---

## Testing

### Structure

```
tests/
├── unit/         # fast, single-class tests; use stub CSV data from tests/stubs/
└── integration/  # full pipeline; may read multiple stubs; slower but no mocking
```

Do not mock `DataStore` or CSV data — use the real `CSVStubProvider` with fixture files
under `tests/stubs/`. This avoids mock/real divergence in schema or data format.

### Fixtures

Stub data lives in:

```
tests/stubs/
├── futures_ohlcv/   # one CSV per ticker (CL.csv, GC.csv, …)
├── options_chain/   # calls and puts per ticker
└── reference/       # contract_specs.csv, production_weights.csv, roll_schedule.csv
```

### Naming

Test files: `test_<module>.py`. Test functions: `test_<what>_<condition>`.

```python
def test_roll_fraction_before_start_is_zero():
def test_equal_weight_sums_to_one():
```
