# QIS Engine — Architecture

## Overview

QIS Engine is a commodity futures index calculation and backtesting library. It replicates
the methodology of production-weight and liquidity-weighted indices (GSCI, BCOM family) and
supports options pricing as a secondary concern.

The design is modelled on the Java Servlet architecture:

- **Container** — the FastAPI/uvicorn process started inside a Docker container
- **CalculationContext** — analogous to `ServletContext`; an application-scoped, immutable
  service provider created once at startup and shared safely across all concurrent requests
- **`calculate()`** — the module-level entry point in `engine.py`, analogous to a Servlet's
  `service()` method; builds a fresh `IndexCalculationEngine` on every call so all state
  lives on the call stack

The core calculation loop is:

```
CalculationContext
       │
       ├── MarketDataService  ──► get_ohlcv(), get_options_chain()
       └── ReferenceDataService ► get_contract_specs(), get_roll_schedule(), get_production_weights()
               │
               ▼
  calculate(index_name, start, end, context)   ← stateless entry point
       │
       └─► IndexCalculationEngine.from_config(config, context)
                   │
                   ▼
           run_calculation(start, end)
                   │
                   ▼
           VectorizedSimulator → BacktestResult
```

---

## Package layout

```
qis/
├── Dockerfile              # python:3.11-slim; polars-lts-cpu; uvicorn on :8080
├── .dockerignore
├── qis_engine/
│   ├── context/        # CalculationContext service provider (NEW)
│   │   ├── base.py     #   CalculationContext dataclass + MarketDataService/ReferenceDataService ABCs
│   │   └── default.py  #   DataStoreMarketDataService, CSVReferenceDataService, build_default_context()
│   ├── core/           # Domain model: config, universe, weighting, roll, rebalancing
│   ├── data/           # I/O abstractions: DataStore, schemas, CSV stubs
│   ├── futures/        # Contract and chain representations, roll cost
│   ├── options/        # Black-76 pricing, greeks, option strategies
│   ├── calculation/    # Index calculation engine, rebalancer, fixing
│   ├── backtest/       # Vectorized simulator, transaction costs, risk metrics
│   ├── api/            # FastAPI routers
│   └── utils/          # Business-day calendars, structured logging
├── qis_engine_ui/      # Standalone FastAPI server + single-page HTML frontend
└── tests/
    ├── unit/           # Fast, isolated tests with stub data
    ├── integration/    # Full calculation pipeline (e.g. BCOM replication)
    └── stubs/          # CSV reference data and OHLCV fixtures
```

---

## Layer responsibilities

### `context/` _(new)_

The service-provider layer that decouples calculation logic from data infrastructure.

| Module | Responsibility |
|--------|---------------|
| `base.py` | `MarketDataService` ABC (`get_ohlcv`, `get_options_chain`); `ReferenceDataService` ABC (`get_contract_specs`, `get_roll_schedule`, `get_production_weights`); `CalculationContext` frozen dataclass holding one of each |
| `default.py` | `DataStoreMarketDataService` — wraps `DataStore`; `CSVReferenceDataService` — wraps `CSVStubProvider`; `build_default_context(stubs_dir)` factory |

`CalculationContext` is created once at server startup and held for the lifetime of the
process. It carries no per-request state, making it safe to share across concurrent requests.

### `core/`

Owns the domain model with no I/O.

| Module | Responsibility |
|--------|---------------|
| `index.py` | `IndexConfig` dataclass; `ReturnType` / `RebalanceSchedule` enums; pre-built `BCOMIndex` / `GSCIIndex` factories |
| `universe.py` | `CommodityUniverse` — filters the eligible constituent set by date and history length; constructed via `from_context(context)` |
| `weighting.py` | `WeightingEngine` — dispatches to strategy objects; `WeightingConstraints` with clip-and-renormalize; each strategy receives `context` at `compute_weights()` call time |
| `roll.py` | `RollCalendar`, `RollSchedule`; `OIRollFinder`, `MomentumRollFinder` — alternative roll-date algorithms |

### `data/`

Low-level I/O building blocks used by the `context/` layer.

| Module | Responsibility |
|--------|---------------|
| `store.py` | `DataStore` — Parquet-backed time-series store; keyed by ticker |
| `models.py` | `ContractSpec` (frozen dataclass); `Sector`, `Exchange`, `Settlement` enums; `OHLCV_SCHEMA` / `OPTIONS_CHAIN_SCHEMA` constants |
| `csv_stub.py` | `CSVStubProvider` — reads local CSV fixtures; used in tests and the default context |

### `futures/`

| Module | Responsibility |
|--------|---------------|
| `contract.py` | `FuturesContract`, `ContractMonth` value types |
| `chain.py` | `FuturesChain` — ratio-adjusted continuous series; approximated deferred pricing |
| `roll_cost.py` | `RollCostCalculator` — per-ticker contango constants |

### `options/`

| Module | Responsibility |
|--------|---------------|
| `pricing.py` | `Black76Pricer` — wraps `optlib.gbs` for European / American commodity options; `OptionPrice` frozen dataclass |
| `greeks.py` | `GreeksCalculator` — analytical greeks + vanna/volga via central finite differences |
| `strategies.py` | Option strategy builders (straddles, collars, etc.) |

### `calculation/`

| Module | Responsibility |
|--------|---------------|
| `engine.py` | `IndexCalculationEngine` — main daily calculation loop; `from_config(config, context)` factory; module-level `calculate()` stateless entry point; `_CONFIG_REGISTRY` mapping index names to configs |
| `rebalancer.py` | `RebalancingEngine` — determines rebalance dates for annual, monthly, and quarterly schedules |
| `fixing.py` | `IndexFixingWorkflow` — handles official index fixing and publication |

### `backtest/`

| Module | Responsibility |
|--------|---------------|
| `simulator.py` | `VectorizedSimulator` — applies costs via Polars expressions (no Python date loop); returns `BacktestResult` |
| `costs.py` | `TransactionCostModel`, `CommissionModel`, `SlippageModel` — frozen dataclasses |
| `risk.py` | `RiskCalculator` — annualised return/vol, Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR |

---

## Data flow

```
CSVStubProvider (or any ReferenceDataService impl)
        │
        │  get_contract_specs()
        │  get_roll_schedule(index_name)
        │  get_production_weights()
        ▼
DataStore (via DataStoreMarketDataService)
        │
        │  get_ohlcv(ticker, start, end)
        ▼
CalculationContext  ──────────────────────────────────────────┐
        │                                                      │ (application-scoped,
        │  calculate(index_name, start, end, context)          │  shared across requests)
        ▼                                                      │
IndexCalculationEngine.from_config(config, context)           │
        │  • CommodityUniverse.from_context(context)           │
        │  • WeightingEngine(method)                           │
        │  • RollCalendar per ticker                           │
        ▼
run_calculation(start, end):
        │  prefetch all closes as dict[ticker → dict[date → float]]
        │  iterate business_days(start, end):
        │    • spot_return   = Σ w_i × (close_i,t / close_i,t-1 − 1)
        │    • roll_return   = Σ w_i × Δfrac_i × (front_i/deferred_i − 1)
        │    • coll_return   = collateral_rate / 360
        │    • excess_return = spot + roll
        │    • total_return  = excess + collateral
        │    on rebalance days: WeightingEngine.compute_weights(eligible, dt, context)
        ▼
pl.DataFrame (date, *_return, *_level, w_<ticker>…)
        │
        ▼
VectorizedSimulator.run()
        │  turnover = 0.5 × Σ|Δw_i|  (Polars .diff())
        │  cost = turnover × (commission_bps + slippage_bps) / 10_000
        │  net_return = gross_return − cost
        │  levels via cum_prod()
        ▼
BacktestResult(timeseries, risk_gross, risk_net)
```

---

## Servlet analogy

| Servlet concept | QIS equivalent |
|----------------|---------------|
| Servlet Container | FastAPI/uvicorn process inside Docker |
| `ServletContext` | `CalculationContext` (application-scoped, immutable after startup) |
| `service(request, response)` | `calculate(index_name, start, end, context) → pl.DataFrame` |
| `HttpServlet` subclass | `IndexCalculationEngine` (instantiated fresh per request) |
| Servlet init | `lifespan()` in `qis_engine_ui/main.py` — builds context once |

---

## Return decomposition

| Return component | Formula | Notes |
|-----------------|---------|-------|
| **Spot** | Σᵢ wᵢ × (Pᵢ,t / Pᵢ,t−1 − 1) | Price-only, no roll |
| **Roll** | Σᵢ wᵢ × Δfracᵢ × (front / deferred − 1) | Negative in contango |
| **Excess** | Spot + Roll | Standard benchmark |
| **Collateral** | collateral_rate / 360 | Actual/360 daily accrual |
| **Total** | Excess + Collateral | Cash-collateralised index |

---

## Weighting strategies

| Method | Class | Description |
|--------|-------|-------------|
| `equal` | `_EqualWeight` | 1/N |
| `production` | `_ProductionWeight` | GSCI-style; proportional to world production averages from `context.reference_data` |
| `liquidity` | `_LiquidityWeight` | BCOM-style; ⅓ production + ⅔ dollar-volume over 252-day lookback from `context.market_data` |
| `risk_parity` | `_RiskParity` | Inverse-volatility; 1-year rolling annualised vol from `context.market_data` |

All strategies return `dict[str, float]` which `_apply_constraints()` clips to
`[min_weight, max_weight]` and renormalises iteratively (≤50 iterations).

---

## Roll algorithms

| Class | Trigger |
|-------|---------|
| `RollCalendar` | Fixed schedule: 5th–9th business day of the delivery month (from CSV) |
| `OIRollFinder` | Roll when front-month OI drops below 30% of its peak |
| `MomentumRollFinder` | Roll on the day within the standard window with the best nearby/deferred ratio |

`RollSchedule.roll_fraction_on(dt)` returns a linear fraction [0, 1] across the window.
`daily_roll_fraction(dt)` returns the incremental fraction for that day only.

---

## Options layer

`Black76Pricer` (b=0 cost-of-carry) wraps `optlib.gbs` and is loaded via `importlib` to
avoid optlib's optional `requests` dependency. `GreeksCalculator` adds vanna and volga via
central finite differences (bump = 1e-4 in σ).

---

## Docker

The server runs in a `python:3.11-slim` container. Key detail: standard `polars` requires
AVX2/FMA CPU instructions not present on all hosts. The Dockerfile explicitly swaps it for
`polars-lts-cpu` after the main `pip install`:

```dockerfile
RUN pip install --no-cache-dir -e . && \
    pip uninstall -y polars && \
    pip install --no-cache-dir polars-lts-cpu
```

Build and run:
```bash
cd qis
docker build -t qis-engine .
docker run -p 8080:8080 qis-engine
```

---

## Extension points

| Concern | How to extend |
|---------|--------------|
| New data source | Implement `MarketDataService` and/or `ReferenceDataService` and pass them into a new `CalculationContext` |
| New weighting method | Add a subclass of `_WeightingStrategy` and register it in `WeightingEngine.__init__` |
| New roll algorithm | Implement a class with `find_roll_date(...)` matching the `OIRollFinder` / `MomentumRollFinder` signature |
| New index | Call `IndexConfig(...)` directly or add a factory in `core/index.py`; register in `_CONFIG_REGISTRY` in `engine.py` |
