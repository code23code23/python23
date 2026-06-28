"""Unit tests for reporting/ and API smoke tests."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from qis_engine.data.store import DataStoreFactory
from qis_engine.reporting.timeseries_exporter import TimeseriesExporter
from qis_engine.utils.calendars import business_days_series


def _populated_store(tmp_path: Path) -> "DataStore":  # noqa: F821
    store = DataStoreFactory.create_parquet(tmp_path)
    dates = business_days_series(date(2022, 1, 3), date(2022, 3, 31))
    levels = [float(100 + i * 0.1) for i in range(len(dates))]
    df = pl.DataFrame({
        "date": dates,
        "spot_level": levels,
        "excess_level": levels,
        "total_level": levels,
        "spot_return": [0.001] * len(dates),
        "excess_return": [0.001] * len(dates),
        "total_return": [0.001] * len(dates),
    })
    store.write_index_level(df, "TEST")
    return store


class TestTimeseriesExporter:
    def test_export_levels_csv_creates_file(self, tmp_path):
        store = _populated_store(tmp_path)
        exp = TimeseriesExporter(store)
        out = exp.export_levels_csv("TEST", date(2022, 1, 3), date(2022, 3, 31), tmp_path / "out.csv")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_levels_json_creates_file(self, tmp_path):
        store = _populated_store(tmp_path)
        exp = TimeseriesExporter(store)
        out = exp.export_levels_json("TEST", date(2022, 1, 3), date(2022, 3, 31), tmp_path / "out.ndjson")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_levels_to_dicts_returns_list(self, tmp_path):
        store = _populated_store(tmp_path)
        exp = TimeseriesExporter(store)
        rows = exp.levels_to_dicts("TEST", date(2022, 1, 3), date(2022, 3, 31))
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert "date" in rows[0]

    def test_date_is_string_in_dicts(self, tmp_path):
        store = _populated_store(tmp_path)
        exp = TimeseriesExporter(store)
        rows = exp.levels_to_dicts("TEST", date(2022, 1, 3), date(2022, 3, 31))
        assert isinstance(rows[0]["date"], str)

    def test_export_covers_requested_range(self, tmp_path):
        store = _populated_store(tmp_path)
        exp = TimeseriesExporter(store)
        rows = exp.levels_to_dicts("TEST", date(2022, 1, 3), date(2022, 1, 31))
        dates = [r["date"] for r in rows]
        assert all(d >= "2022-01-03" for d in dates)
        assert all(d <= "2022-01-31" for d in dates)


class TestRollCalendarReport:
    def test_build_returns_dataframe(self):
        from qis_engine.core.roll import RollCalendar, RollSchedule
        from qis_engine.reporting.roll_calendar_report import RollCalendarReport

        df = pl.DataFrame([{
            "ticker": "CL", "year": y, "month": m,
            "roll_start_bday": 5, "roll_end_bday": 9,
            "contract_month": m + 1 if m < 12 else 1,
        } for y in (2022,) for m in range(1, 13)])
        cal = RollCalendar("CL", df)
        report = RollCalendarReport({"CL": cal})
        result = report.build(date(2022, 1, 1), date(2022, 12, 31))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 12
        assert "roll_start" in result.columns

    def test_to_dicts_dates_are_strings(self):
        from qis_engine.core.roll import RollCalendar
        from qis_engine.reporting.roll_calendar_report import RollCalendarReport

        df = pl.DataFrame([{
            "ticker": "GC", "year": 2022, "month": 3,
            "roll_start_bday": 5, "roll_end_bday": 9,
            "contract_month": 4,
        }])
        cal = RollCalendar("GC", df)
        report = RollCalendarReport({"GC": cal})
        rows = report.to_dicts(date(2022, 1, 1), date(2022, 12, 31))
        assert len(rows) == 1
        assert isinstance(rows[0]["roll_start"], str)


class TestAPISmoke:
    """Smoke tests — import the app and verify routing without starting a server."""

    def test_app_creates_without_error(self):
        from fastapi.testclient import TestClient
        from qis_engine.api.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_list_indices(self):
        from fastapi.testclient import TestClient
        from qis_engine.api.app import app
        client = TestClient(app)
        response = client.get("/indices/")
        assert response.status_code == 200
        data = response.json()
        assert "BCOM" in data or "GSCI" in data or isinstance(data, list)

    def test_price_option_endpoint(self):
        from fastapi.testclient import TestClient
        from qis_engine.api.app import app
        client = TestClient(app)
        response = client.post("/options/price", json={
            "option_type": "c",
            "F": 80.0, "K": 85.0, "T": 0.25, "r": 0.05, "sigma": 0.30,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["value"] > 0
        assert 0 < data["delta"] < 1

    def test_implied_vol_endpoint(self):
        from fastapi.testclient import TestClient
        from qis_engine.api.app import app
        client = TestClient(app)
        # First get a market price
        p_resp = client.post("/options/price", json={
            "option_type": "c", "F": 80.0, "K": 85.0, "T": 0.25, "r": 0.05, "sigma": 0.30,
        })
        market_price = p_resp.json()["value"]
        # Then recover the vol
        iv_resp = client.post("/options/implied_vol", json={
            "option_type": "c", "F": 80.0, "K": 85.0, "T": 0.25, "r": 0.05,
            "market_price": market_price,
        })
        assert iv_resp.status_code == 200
        assert abs(iv_resp.json()["implied_vol"] - 0.30) < 1e-4
