# QIS Engine UI

A minimal web frontend for the QIS Engine. No JavaScript framework, no build step — a single HTML file served by a lightweight FastAPI server.

---

## Running the UI

Install the engine package first (from the `qis/` directory):

```bash
cd qis
pip install -e ".[dev]"
```

Then start the UI server:

```bash
cd qis/qis_engine_ui
python -m uvicorn main:app --reload --port 8080
```

Open `http://localhost:8080` in a browser.

> The `--reload` flag restarts the server automatically on code changes. Omit it in production.
>
> If the `uvicorn` command is not on your PATH (common on Windows user installs), always use
> `python -m uvicorn` — it works regardless of PATH configuration.

---

## How it works

On startup the server:
1. Creates a temporary Parquet store in a system temp directory
2. Loads all CSV stub data from `qis/tests/stubs/` (OHLCV, options chains, roll schedules)
3. Hands the `CalculationContext` to the engine (`init_engine`)

Each calculation request creates a new `Calculator` for the chosen index and runs `calculate(start, end)` — nothing is cached between requests.

---

## API

### `POST /api/calculate`

**Request body:**
```json
{
  "index": "BCOM",
  "start": "2020-01-02",
  "end":   "2021-12-31"
}
```

`index` must be `"BCOM"` or `"GSCI"`.

**Response:**
```json
{
  "columns": ["date", "spot_return", "roll_return", "collateral_return",
              "excess_return", "total_return",
              "spot_level", "excess_level", "total_level",
              "w_CL", "w_GC", "w_NG", "w_HG", "w_ZC", "w_ZW", "w_LC"],
  "rows": [
    { "date": "2020-01-02", "spot_return": 0.0, "excess_level": 100.0, ... },
    ...
  ],
  "summary": {
    "n_days": 502,
    "spot_final": 108.23,
    "excess_final": 105.11,
    "total_final": 105.11
  }
}
```

Returns HTTP 400 if the date range is invalid or no stub data exists for the requested period.

---

## UI design

### Layout

```
┌─────────────────────────────────────────────────┐
│  QIS Index Engine                               │
│                                                 │
│  Index [BCOM ▾]  Start [____]  End [____]  [Calculate] ⟳  │
│                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────┐ │
│  │ Spot level   │ │ Excess level │ │ Total level  │ │ Days │ │
│  │   108.23     │ │   105.11     │ │   105.11     │ │  502 │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────┘ │
│                                                 │
│  ┌──────────────────────────────── scrollable ──┐  │
│  │ DATE  SPOT RET  ROLL RET  …  EXCESS LVL  w_CL … │  │
│  │ …     +0.02%   -0.01%    …  100.22       14.3% │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Styling

- Dark background (`#0f1117`) with muted card surfaces (`#1e2433`)
- System font stack (no external font load)
- Accent color: indigo `#6366f1` for the button and focus rings

### Table formatting

| Column pattern | Format | Notes |
|---------------|--------|-------|
| `date` | `YYYY-MM-DD` | Left-aligned |
| `*_return` | `+0.0000%` / `-0.0000%` | Green for positive, red for negative |
| `*_level` | `0.00` | 2 decimal places |
| `w_<ticker>` | `0.0%` | 1 decimal place |

The table body is scrollable (`max-height: 65vh`) with a sticky header row so column names stay visible while scrolling through multi-year date ranges.

### Error handling

- Invalid date range or missing stub data → red error banner below the form
- Network failure → error banner with the exception message
- Button is disabled and a CSS spinner is shown while the request is in flight

---

## File structure

```
qis_engine_ui/
├── main.py          # FastAPI app (startup, /api/calculate, static serving)
├── static/
│   └── index.html   # Single-page UI — HTML + CSS + JS, no external deps
└── README.md
```

---

## Available indices and date ranges

Indices are backed by CSV stub data in `qis/tests/stubs/futures_ohlcv/`.

| Index | Weighting | Tickers |
|-------|-----------|---------|
| BCOM  | Liquidity (⅓ production + ⅔ dollar volume) | CL, GC, NG, HG, ZC, ZW, LC |
| GSCI  | Production (world output averages) | CL, GC, NG, HG, ZC, ZW, LC |

Stub data covers approximately 2010–2024. Requests outside this range will return a 400 error.
