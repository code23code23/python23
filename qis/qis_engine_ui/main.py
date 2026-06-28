from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qis_engine.calculation.engine import calculate
from qis_engine.context.default import build_default_context

STUBS_DIR = Path(__file__).parents[1] / "tests" / "stubs"
STATIC_DIR = Path(__file__).parent / "static"

# Application-scoped context — stateless service handles, safe to share across requests
_context: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _context
    _context = build_default_context(STUBS_DIR)
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
def calculate_endpoint(req: CalcRequest):
    """
    Servlet service() equivalent — stateless per-request handler.

    Each call builds a fresh IndexCalculationEngine from the shared context.
    All calculation state lives only on this call's stack.
    """
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end date must be on or after start date")

    try:
        df = calculate(req.index, req.start, req.end, _context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
