from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import polars as pl


def score_predictions(
    frame: pl.DataFrame, probability_col: str = "ensemble_probability"
) -> dict[str, float]:
    if frame.is_empty():
        return {
            "brier": math.nan,
            "log_score": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
            "expected_calibration_error": math.nan,
            "expected_calibration_error_bins": math.nan,
            "interval_90_coverage": math.nan,
        }
    actual = frame["actual_winner"].cast(pl.Float64).to_numpy()
    probability = np.clip(frame[probability_col].cast(pl.Float64).to_numpy(), 1e-6, 1 - 1e-6)
    brier = float(np.mean((probability - actual) ** 2))
    log_score = float(
        np.mean(-(actual * np.log(probability) + (1 - actual) * np.log(1 - probability)))
    )
    slope, intercept = _calibration_line(probability, actual)
    ece_bins = _ece_bin_count(len(probability))
    ece = _expected_calibration_error(probability, actual, bins=ece_bins)
    coverage = np.mean(
        (frame["actual_vote_share"].to_numpy() >= frame["lower_90"].to_numpy())
        & (frame["actual_vote_share"].to_numpy() <= frame["upper_90"].to_numpy())
    )
    return {
        "brier": brier,
        "log_score": log_score,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "expected_calibration_error": float(ece),
        "expected_calibration_error_bins": float(ece_bins),
        "interval_90_coverage": float(coverage),
    }


def _calibration_line(probability: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    if len(np.unique(probability)) < 2:
        return 0.0, float(np.mean(actual))
    logit_probability = np.log(probability / (1.0 - probability))
    design = np.column_stack([np.ones_like(logit_probability), logit_probability])
    beta = np.zeros(2)
    ridge = 1e-3
    for _ in range(25):
        eta = design @ beta
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        hessian = design.T @ (weights[:, None] * design) + ridge * np.eye(2)
        gradient = design.T @ (actual - fitted) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    intercept, slope = beta
    return float(slope), float(intercept)


def _expected_calibration_error(
    probability: np.ndarray, actual: np.ndarray, bins: int = 5
) -> float:
    quantiles = np.linspace(0, 1, min(bins, len(probability)) + 1)
    unique_edges = np.unique(np.quantile(probability, quantiles))
    edges = unique_edges if unique_edges.size > 1 else np.linspace(0, 1, bins + 1)
    total = len(probability)
    error = 0.0
    for lower, upper in pairwise(edges):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if not np.any(mask):
            continue
        error += np.mean(mask) * abs(float(np.mean(probability[mask]) - np.mean(actual[mask])))
    return float(error if total else math.nan)


def _ece_bin_count(row_count: int) -> int:
    if row_count <= 0:
        return 1
    return max(1, min(15, row_count // 30))
