# QIS Engine — Architecture

## Overview

QIS Engine is a commodity futures index calculation and backtesting library. It replicates
the methodology of production-weight and liquidity-weighted indices (GSCI, BCOM family) and
supports options pricing as a secondary concern.

The core loop is:

```
IndexConfig → CommodityUniverse → WeightingEngine → IndexCalculationEngine → VectorizedSimulator → BacktestResult
```

---

## Package layout

```
qis/
├── qis_engine/
│   ├── core/           # Domain model: config, universe, weighting, roll, rebalancing
│   ├── data/           # I/O abstractions: provider interface, DataStore, schemas, CSV stubs
│   ├── futures/        # Contract and chain representations, roll cost
│   ├── options/        # Black-76 pricing, greeks, option strategies
│   ├── calculation/    # Index calculation engine, rebalancer, fixing
│   ├── backtest/       # Vectorized simulator, transaction costs, risk metrics
│   ├── reporting/      # (placeholder)
│   ├── api/            # FastAPI routers (placeholder)
│   └── utils/          # Business-day calendars, structured logging
└── tests/
    ├── unit/           # Fast, isolated tests with stub data
    ├── integration/    # Full calculation pipeline (e.g. BCOM replication)
    └── stubs/          # CSV reference data and OHLCV fixtures
```

---

## Layer responsibilities

### `core/`

Owns the domain model with no I/O.

| Module | Responsibility |
|--------|---------------|
| `index.py` | `IndexConfig` dataclass; `ReturnType` / `RebalanceSchedule` enums; pre-built `BCOMIndex` / `GSCIIndex` factories |
| `universe.py` | `CommodityUniverse` — filters the eligible constituent set by date and history length |
| `weighting.py` | `WeightingEngine` — dispatches to strategy objects; `WeightingConstraints` with clip-and-renormalize |
| `roll.py` | `RollCalendar`, `RollSchedule`; `OIRollFinder`, `MomentumRollFinder` — alternative roll-date algorithms |

### `data/`

Decouples market data from the engine. All concrete data access goes through one of these two objects.

| Module | Responsibility |
|--------|---------------|
| `base.py` | `DataProviderBase` ABC — the extension point for external market data (Bloomberg, Quandl, etc.) |
| `store.py` | `DataStore` — in-memory keyed store populated by a provider's `load_all()` |
| `models.py` | `ContractSpec` (frozen dataclass); `Sector`, `Exchange`, `Settlement` enums; `OHLCV_SCHEMA` / `OPTIONS_CHAIN_SCHEMA` constants |
| `csv_stub.py` | `CSVStubProvider` — reference implementation using local CSV fixtures |

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
| `engine.py` | `IndexCalculationEngine` — the main daily calculation loop; prefetches all prices, iterates business days |
| `rebalancer.py` | `RebalancingEngine` — determines rebalance dates for annual, monthly, and quarterly schedules |
| `fixing.py` | `FixingEngine` — handles official index fixing and publication |

### `backtest/`

| Module | Responsibility |
|--------|---------------|
| `simulator.py` | `VectorizedSimulator` — applies costs via Polars expressions (no Python date loop); returns `BacktestResult` |
| `costs.py` | `TransactionCostModel`, `CommissionModel`, `SlippageModel` — frozen dataclasses |
| `risk.py` | `RiskCalculator` — annualised return/vol, Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR |

---

## Data flow

```
CSVStubProvider (or external provider)
        │
        │ load_all()
        ▼
     DataStore                          ← in-memory; keyed by (ticker, start, end)
        │
        ├─► CommodityUniverse.get_eligible()   ← filters by ticker list + history length
        │
        ├─► WeightingEngine.compute_weights()  ← equal | production | liquidity | risk_parity
        │         └─ _apply_constraints()      ← iterative clip-and-renormalize (≤50 iterations)
        │
        ├─► RollCalendar.get_active_schedule() ← 5th–9th business day roll window per month
        │
        └─► IndexCalculationEngine.run_calculation()
                    │
                    │  prefetch all closes as dict[ticker → dict[date → float]]
                    │  iterate business_days(start, end):
                    │    • spot_return   = Σ w_i × (close_i,t / close_i,t-1 − 1)
                    │    • roll_return   = Σ w_i × Δfrac_i × (front_i/deferred_i − 1)
                    │    • coll_return   = collateral_rate / 360
                    │    • excess_return = spot + roll
                    │    • total_return  = excess + collateral
                    │
                    ▼
            pl.DataFrame (date, *_return, *_level, w_<ticker>…)
                    │
                    ▼
        VectorizedSimulator.run()
                    │  turnover = 0.5 × Σ|Δw_i| (Polars .diff())
                    │  cost = turnover × (commission_bps + slippage_bps) / 10_000
                    │  net_return = gross_return − cost
                    │  levels via cum_prod()
                    ▼
            BacktestResult(timeseries, risk_gross, risk_net)
```

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
| `production` | `_ProductionWeight` | GSCI-style; proportional to world production averages |
| `liquidity` | `_LiquidityWeight` | BCOM-style; ⅓ production + ⅔ dollar-volume over 252-day lookback |
| `risk_parity` | `_RiskParity` | Inverse-volatility; 1-year rolling annualised vol |

All strategies return `dict[str, float]` which `_apply_constraints()` clips to
`[min_weight, max_weight]` and renormalises iteratively.

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

## Extension points

| Concern | How to extend |
|---------|--------------|
| New data source | Subclass `DataProviderBase` and implement four abstract methods |
| New weighting method | Add a subclass of `_WeightingStrategy` and register it in `WeightingEngine.__init__` |
| New roll algorithm | Implement a class with `find_roll_date(...)` matching the `OIRollFinder` / `MomentumRollFinder` signature |
| New index | Call `IndexConfig(...)` directly or add a factory function in `core/index.py` |
