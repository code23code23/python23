from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/options", tags=["options"])


class PriceRequest(BaseModel):
    option_type: Literal["c", "p"]
    F: float
    K: float
    T: float
    r: float
    sigma: float
    american: bool = False


class PriceResponse(BaseModel):
    value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_vol: float | None = None


class IVRequest(BaseModel):
    option_type: Literal["c", "p"]
    F: float
    K: float
    T: float
    r: float
    market_price: float
    american: bool = False


def get_store():
    from qis_engine.api.app import get_app_store
    return get_app_store()


@router.post("/price", response_model=PriceResponse)
def price_option(req: PriceRequest) -> PriceResponse:
    from qis_engine.options.pricing import Black76Pricer
    pricer = Black76Pricer()
    p = pricer.price(req.option_type, req.F, req.K, req.T, req.r, req.sigma, req.american)
    return PriceResponse(
        value=p.value, delta=p.delta, gamma=p.gamma,
        theta=p.theta, vega=p.vega, rho=p.rho,
    )


@router.post("/implied_vol")
def implied_vol(req: IVRequest) -> dict:
    from qis_engine.options.pricing import Black76Pricer
    import math
    pricer = Black76Pricer()
    iv = pricer.implied_vol(req.option_type, req.F, req.K, req.T, req.r, req.market_price, req.american)
    if math.isnan(iv):
        raise HTTPException(status_code=422, detail="Implied vol solver did not converge")
    return {"implied_vol": iv}


@router.get("/{ticker}/chain")
def get_options_chain(
    ticker: str,
    as_of: Annotated[date, Query()],
    option_type: Annotated[Literal["c", "p"], Query()] = "c",
    store=Depends(get_store),
) -> list[dict]:
    import polars as pl
    df = store.read_options_chain(ticker, as_of, option_type)
    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"No options chain for {ticker} on {as_of}")
    return df.with_columns(
        pl.col("date").cast(pl.Utf8),
        pl.col("expiry").cast(pl.Utf8),
    ).to_dicts()
