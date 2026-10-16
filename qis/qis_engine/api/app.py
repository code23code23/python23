from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from qis_engine.api.schemas import HealthResponse
from qis_engine.api.routers import indices, futures, options

# Module-level singletons injected at startup
_store: Any = None
_context: Any = None


def get_app_store():
    if _store is None:
        raise RuntimeError("DataStore not initialised — call init_app() before serving")
    return _store


def get_app_context():
    if _context is None:
        raise RuntimeError("CalculationContext not initialised — call init_app() before serving")
    return _context


def create_app(
    store: Any | None = None,
    context: Any | None = None,
) -> FastAPI:
    global _store, _context
    if store is not None:
        _store = store
    if context is not None:
        _context = context

    app = FastAPI(
        title="QIS Index Engine",
        description="REST API for commodity futures QIS index calculation, backtesting, and options pricing.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(indices.router)
    app.include_router(futures.router)
    app.include_router(options.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse()

    return app


def init_app(
    store_root: str | Path | None = None,
    stubs_dir: str | Path | None = None,
) -> FastAPI:
    """
    Convenience initialiser for development: loads CSV stubs into a
    temporary Parquet store and wires up BCOM + GSCI Calculator singletons.
    """
    import tempfile
    from qis_engine.context.default import build_default_context
    from qis_engine.data.csv_stub import CSVStubProvider
    from qis_engine.data.store import DataStoreFactory
    from qis_engine.calculation.engine import init_engine

    root = Path(store_root) if store_root else Path(tempfile.mkdtemp())
    stubs = Path(stubs_dir) if stubs_dir else Path(__file__).parents[2] / "tests" / "stubs"

    provider = CSVStubProvider(stubs)
    store = DataStoreFactory.create_parquet(root)
    provider.load_all(store)

    roll_df = provider.get_roll_schedule()
    store.write_roll_schedule(roll_df, "BCOM")
    store.write_roll_schedule(roll_df, "GSCI")

    context = build_default_context(stubs, store)
    init_engine(context)

    return create_app(store=store, context=context)


# Convenience: `uvicorn qis_engine.api.app:app` in dev with stubs
app = init_app()
