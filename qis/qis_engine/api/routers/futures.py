from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

router = APIRouter(prefix="/futures", tags=["futures"])


def get_store():
    from qis_engine.api.app import get_app_store
    return get_app_store()


@router.get("/{ticker}/ohlcv")
def get_ohlcv(
    ticker: str,
    start_date: Annotated[date, Query()],
    end_date: Annotated[date, Query()],
    store=Depends(get_store),
) -> list[dict]:
    import polars as pl
    df = store.read_ohlcv(ticker, start_date, end_date)
    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"No OHLCV data for {ticker}")
    return df.with_columns(pl.col("date").cast(pl.Utf8)).to_dicts()


@router.get("/{ticker}/roll_schedule")
def get_ticker_roll_schedule(
    ticker: str,
    index_name: Annotated[str, Query()] = "BCOM",
    store=Depends(get_store),
) -> list[dict]:
    import polars as pl
    df = store.read_roll_schedule(index_name)
    if df.is_empty():
        return []
    filtered = df.filter(pl.col("ticker") == ticker)
    return filtered.to_dicts()
