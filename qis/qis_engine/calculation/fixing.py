from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import polars as pl


@dataclass
class IndexFixResult:
    fix_date: date
    is_preliminary: bool
    spot_level: float
    excess_level: float
    total_level: float
    spot_return: float
    excess_return: float
    total_return: float
    weights: dict[str, float]
    fixed_at: datetime
    validated: bool = False
    validation_notes: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]

    SPOT_RETURN_LIMIT = 0.10
    WEIGHT_SUM_TOLERANCE = 0.001
    LEVEL_MOVE_WARN = 0.15


class IndexFixingWorkflow:
    """
    T+0 daily index fixing: preliminary at market close, final after confirmation.
    """

    def __init__(
        self,
        calculator: "Calculator",  # noqa: F821
        store: "DataStore",  # noqa: F821
        index_name: str,
    ) -> None:
        self._calculator = calculator
        self._store = store
        self._index_name = index_name

    def run_daily_fix(
        self,
        fix_date: date,
        is_preliminary: bool = False,
        collateral_rate: float = 0.0,
    ) -> IndexFixResult:
        from qis_engine.utils.calendars import prev_business_day

        prior_date = prev_business_day(fix_date)
        prior_df = self._store.read_index_level(
            self._index_name,
            prior_date,
            prior_date,
        )
        if prior_df.is_empty():
            raise ValueError(
                f"No prior index level for {self._index_name} on {prior_date}. "
                "Run a full historical calculation first."
            )
        prior = prior_df.row(0, named=True)

        # Calculate today's returns
        result_df = self._calculator.calculate(
            fix_date, fix_date,
            collateral_rate_series=pl.Series([collateral_rate]),
        )
        if result_df.is_empty():
            raise ValueError(f"No market data available for {fix_date}")

        row = result_df.row(0, named=True)
        weights = {
            k[2:]: v for k, v in row.items()
            if k.startswith("w_") and isinstance(v, float)
        }

        result = IndexFixResult(
            fix_date=fix_date,
            is_preliminary=is_preliminary,
            spot_level=float(prior["spot_level"]) * (1 + row["spot_return"]),
            excess_level=float(prior["excess_level"]) * (1 + row["excess_return"]),
            total_level=float(prior["total_level"]) * (1 + row["total_return"]),
            spot_return=row["spot_return"],
            excess_return=row["excess_return"],
            total_return=row["total_return"],
            weights=weights,
            fixed_at=datetime.now(timezone.utc),
        )

        validation = self._validate(result, prior)
        result.validated = validation.passed
        result.validation_notes = validation.errors + validation.warnings

        if validation.passed or is_preliminary:
            self._publish(result)

        return result

    def _validate(self, result: IndexFixResult, prior: dict) -> ValidationResult:
        errors, warnings = [], []

        if abs(result.spot_return) > ValidationResult.SPOT_RETURN_LIMIT:
            errors.append(
                f"Spot return {result.spot_return:.2%} exceeds "
                f"±{ValidationResult.SPOT_RETURN_LIMIT:.0%} threshold"
            )
        w_sum = sum(result.weights.values())
        if abs(w_sum - 1.0) > ValidationResult.WEIGHT_SUM_TOLERANCE:
            errors.append(f"Weights sum to {w_sum:.6f}, expected 1.0")
        if result.total_level <= 0:
            errors.append("Total index level is non-positive")
        prior_total = float(prior.get("total_level", 1.0))
        if prior_total > 0 and abs(result.total_level / prior_total - 1.0) > ValidationResult.LEVEL_MOVE_WARN:
            warnings.append("Total level moved >15% vs prior — verify data")

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _publish(self, result: IndexFixResult) -> None:
        row = pl.DataFrame([{
            "date": result.fix_date,
            "spot_level": result.spot_level,
            "excess_level": result.excess_level,
            "total_level": result.total_level,
            "spot_return": result.spot_return,
            "excess_return": result.excess_return,
            "total_return": result.total_return,
            "is_preliminary": result.is_preliminary,
            "validated": result.validated,
        }]).with_columns(pl.col("date").cast(pl.Date))
        self._store.write_index_level(row, self._index_name)
