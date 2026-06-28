from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from qis_engine.api.schemas import HealthResponse
from qis_engine.api.routers import indices, futures, options

# Module-level singletons injected at startup
_store: Any = None
_engines: dict[str, Any] = {}


def get_app_store():
    if _store is None:
        raise RuntimeError("DataStore not initialised — call init_app() before serving")
    return _store


def get_app_engines():
    return _engines


def create_app(
    store: Any | None = None,
    engines: dict[str, Any] | None = None,
) -> FastAPI:
    """
    Factory for the FastAPI application.

    Args:
        store:   DataStore instance to use for all requests. If None the app
                 will raise at runtime when store access is attempted.
        engines: dict mapping index name → IndexCalculationEngine, used by
                 the backtest endpoint.
    """
    global _store, _engines
    if store is not None:
        _store = store
    if engines is not None:
        _engines = engines

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
    temporary Parquet store and wires up BCOM + GSCI engines.
    """
    import tempfile
    from qis_engine.data.csv_stub import CSVStubProvider
    from qis_engine.data.store import DataStoreFactory
    from qis_engine.core.index import BCOMIndex, GSCIIndex
    from qis_engine.calculation.engine import IndexCalculationEngine

    root = Path(store_root) if store_root else Path(tempfile.mkdtemp())
    stubs = Path(stubs_dir) if stubs_dir else Path(__file__).parents[2] / "tests" / "stubs"

    provider = CSVStubProvider(stubs)
    store = DataStoreFactory.create_parquet(root)
    provider.load_all(store)

    roll_df = provider.get_roll_schedule()
    store.write_roll_schedule(roll_df, "BCOM")
    store.write_roll_schedule(roll_df, "GSCI")

    engines = {}
    for config_factory in (BCOMIndex.default_config, GSCIIndex.default_config):
        cfg = config_factory()
        eng = IndexCalculationEngine.from_config(cfg, store, provider)
        engines[cfg.name] = eng

    return create_app(store=store, engines=engines)


# Convenience: `uvicorn qis_engine.api.app:app` in dev with stubs
app = init_app()
