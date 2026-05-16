from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from civic_signal.storage.io import write_parquet

RECALIBRATION_SCHEMA: dict[str, pl.DataType] = {
    "method": pl.String,
    "status": pl.String,
    "input_probability": pl.String,
    "intercept": pl.Float64,
    "slope": pl.Float64,
    "sample_size": pl.Int64,
    "ridge": pl.Float64,
    "min_slope": pl.Float64,
    "max_slope": pl.Float64,
    "max_abs_intercept": pl.Float64,
    "uncalibrated_log_loss": pl.Float64,
    "calibrated_log_loss": pl.Float64,
    "fit_cycles": pl.String,
    "as_of_cuts": pl.String,
    "fit_at": pl.String,
    "source_manifest_hash": pl.String,
    "model_config_hash": pl.String,
}


@dataclass(frozen=True)
class RecalibrationMap:
    method: str
    status: str
    input_probability: str
    intercept: float
    slope: float
    sample_size: int
    ridge: float
    min_slope: float
    max_slope: float
    max_abs_intercept: float
    uncalibrated_log_loss: float | None
    calibrated_log_loss: float | None
    fit_cycles: tuple[int, ...]
    as_of_cuts: tuple[int, ...]
    fit_at: str
    source_manifest_hash: str = ""
    model_config_hash: str = ""

    def apply(self, probability: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(probability, dtype=np.float64)
        if self.status != "fitted":
            return values
        from civic_signal.scoring.learning import apply_platt_calibration

        return apply_platt_calibration(values, {"intercept": self.intercept, "slope": self.slope})

    def to_frame(self) -> pl.DataFrame:
        row = {
            "method": self.method,
            "status": self.status,
            "input_probability": self.input_probability,
            "intercept": self.intercept,
            "slope": self.slope,
            "sample_size": self.sample_size,
            "ridge": self.ridge,
            "min_slope": self.min_slope,
            "max_slope": self.max_slope,
            "max_abs_intercept": self.max_abs_intercept,
            "uncalibrated_log_loss": self.uncalibrated_log_loss,
            "calibrated_log_loss": self.calibrated_log_loss,
            "fit_cycles": ",".join(str(value) for value in self.fit_cycles),
            "as_of_cuts": ",".join(str(value) for value in self.as_of_cuts),
            "fit_at": self.fit_at,
            "source_manifest_hash": self.source_manifest_hash,
            "model_config_hash": self.model_config_hash,
        }
        return pl.DataFrame([row], schema=RECALIBRATION_SCHEMA)

    def to_parquet(self, path: Path) -> Path:
        return write_parquet(self.to_frame(), path)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> RecalibrationMap:
        if frame.is_empty():
            raise ValueError("recalibration map frame is empty")
        row = frame.row(0, named=True)
        return cls(
            method=str(row["method"]),
            status=str(row["status"]),
            input_probability=str(row["input_probability"]),
            intercept=float(row["intercept"]),
            slope=float(row["slope"]),
            sample_size=int(row["sample_size"]),
            ridge=float(row["ridge"]),
            min_slope=float(row["min_slope"]),
            max_slope=float(row["max_slope"]),
            max_abs_intercept=float(row["max_abs_intercept"]),
            uncalibrated_log_loss=_optional_float(row.get("uncalibrated_log_loss")),
            calibrated_log_loss=_optional_float(row.get("calibrated_log_loss")),
            fit_cycles=_parse_int_tuple(row.get("fit_cycles")),
            as_of_cuts=_parse_int_tuple(row.get("as_of_cuts")),
            fit_at=str(row["fit_at"]),
            source_manifest_hash=str(row.get("source_manifest_hash") or ""),
            model_config_hash=str(row.get("model_config_hash") or ""),
        )

    @classmethod
    def from_parquet(cls, path: Path) -> RecalibrationMap:
        return cls.from_frame(pl.read_parquet(path))


def fit_recalibration(
    frame: pl.DataFrame,
    *,
    probability_col: str = "learned_ensemble_probability",
    actual_col: str = "actual_winner",
    config: dict[str, Any] | None = None,
    cycles: list[int] | None = None,
    as_of_cuts: list[int] | None = None,
) -> RecalibrationMap:
    settings = dict((config or {}).get("ensemble_learning", {}))
    if frame.is_empty() or probability_col not in frame.columns or actual_col not in frame.columns:
        calibration = _identity_calibration("no_rows", settings, row_count=0)
    else:
        from civic_signal.scoring.learning import fit_platt_calibration

        calibration = fit_platt_calibration(
            frame[probability_col].cast(pl.Float64).to_numpy(),
            frame[actual_col].cast(pl.Float64).to_numpy(),
            min_rows=int(dict(config or {}).get("minimum_rows_for_trust", 30)),
            ridge=float(settings.get("calibration_ridge", 1e-3)),
            min_slope=float(settings.get("calibration_min_slope", 0.25)),
            max_slope=float(settings.get("calibration_max_slope", 1.0)),
            max_abs_intercept=float(settings.get("calibration_max_abs_intercept", 2.0)),
        )
        calibration["input_probability"] = probability_col
    return recalibration_map_from_calibration(
        calibration,
        cycles=cycles or _unique_ints(frame, "cycle"),
        as_of_cuts=as_of_cuts or _unique_ints(frame, "as_of_offset_days"),
    )


def recalibration_map_from_calibration(
    calibration: dict[str, Any],
    *,
    cycles: list[int] | tuple[int, ...] | None = None,
    as_of_cuts: list[int] | tuple[int, ...] | None = None,
    source_manifest_hash: str = "",
    model_config_hash: str = "",
) -> RecalibrationMap:
    return RecalibrationMap(
        method=str(calibration.get("method", "platt_logistic_ridge")),
        status=str(calibration.get("status", "identity")),
        input_probability=str(calibration.get("input_probability", "learned_ensemble_probability")),
        intercept=float(calibration.get("intercept", 0.0)),
        slope=float(calibration.get("slope", 1.0)),
        sample_size=int(calibration.get("row_count", 0)),
        ridge=float(calibration.get("ridge", 1e-3)),
        min_slope=float(calibration.get("min_slope", 0.25)),
        max_slope=float(calibration.get("max_slope", 1.0)),
        max_abs_intercept=float(calibration.get("max_abs_intercept", 2.0)),
        uncalibrated_log_loss=_optional_float(calibration.get("uncalibrated_log_loss")),
        calibrated_log_loss=_optional_float(calibration.get("calibrated_log_loss")),
        fit_cycles=tuple(sorted(int(value) for value in (cycles or ()))),
        as_of_cuts=tuple(sorted(int(value) for value in (as_of_cuts or ()))),
        fit_at=str(calibration.get("fit_at") or datetime.now(UTC).isoformat()),
        source_manifest_hash=source_manifest_hash,
        model_config_hash=model_config_hash,
    )


def _identity_calibration(
    status: str, settings: dict[str, Any], *, row_count: int
) -> dict[str, Any]:
    return {
        "status": status,
        "method": "platt_logistic_ridge",
        "intercept": 0.0,
        "slope": 1.0,
        "row_count": row_count,
        "ridge": float(settings.get("calibration_ridge", 1e-3)),
        "min_slope": float(settings.get("calibration_min_slope", 0.25)),
        "max_slope": float(settings.get("calibration_max_slope", 1.0)),
        "max_abs_intercept": float(settings.get("calibration_max_abs_intercept", 2.0)),
    }


def _unique_ints(frame: pl.DataFrame, column: str) -> list[int]:
    if frame.is_empty() or column not in frame.columns:
        return []
    return sorted(int(value) for value in frame[column].drop_nulls().unique().to_list())


def _parse_int_tuple(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(int(part) for part in text.split(",") if part.strip())


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    output = float(value)
    return output if np.isfinite(output) else None
