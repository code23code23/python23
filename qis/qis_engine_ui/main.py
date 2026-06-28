from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qis_engine.calculation.engine import IndexCalculationEngine
from qis_engine.core.index import BCOMIndex, GSCIIndex
from qis_engine.data.csv_stub import CSVStubProvider
from qis_engine.data.store import DataStoreFactory

STUBS_DIR = Path(__file__).parents[1] / "tests" / "stubs"
STATIC_DIR = Path(__file__).parent / "static"

_store: Any = None
_engines: dict[str, IndexCalculationEngine] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _engines
    root = Path(tempfile.mkdtemp())
    provider = CSVStubProvider(STUBS_DIR)
    _store = DataStoreFactory.create_parquet(root)
    provider.load_all(_store)

    roll_df = provider.get_roll_schedule()
    _store.write_roll_schedule(roll_df, "BCOM")
    _store.write_roll_schedule(roll_df, "GSCI")

    for factory in (BCOMIndex.default_config, GSCIIndex.default_config):
        cfg = factory()
        _engines[cfg.name] = IndexCalculationEngine.from_config(cfg, _store, provider)

    yield


app = FastAPI(title="QIS Engine UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CalcRequest(BaseModel):
    index: Literal["BCOM", "GSCI"]
    start: date
    end: date


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/calculate")
def calculate(req: CalcRequest):
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end date must be on or after start date")

    engine = _engines.get(req.index)
    if engine is None:
        raise HTTPException(status_code=400, detail=f"Unknown index: {req.index}")

    df = engine.run_calculation(req.start, req.end)
    if df.is_empty():
        raise HTTPException(
            status_code=400,
            detail=f"No data for {req.index} between {req.start} and {req.end}",
        )

    last = df.row(-1, named=True)
    summary = {
        "n_days": len(df),
        "spot_final": round(float(last["spot_level"]), 4),
        "excess_final": round(float(last["excess_level"]), 4),
        "total_final": round(float(last["total_level"]), 4),
    }

    rows = df.with_columns(pl.col("date").cast(pl.Utf8)).to_dicts()

    return {
        "columns": df.columns,
        "rows": rows,
        "summary": summary,
    }
