from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

EPSILON = 1e-6


def fit_simplex_weights(
    frame: pl.DataFrame,
    component_columns: dict[str, str],
    default_weights: dict[str, float],
    eligible_components: dict[str, bool],
    *,
    max_iterations: int = 800,
    learning_rate: float = 0.35,
    l2_prior_strength: float = 0.02,
    min_rows: int = 30,
) -> dict[str, Any]:
    """Fit non-negative simplex ensemble weights on rolling-origin predictions."""

    configured_weights = _float_weight_dict(default_weights, component_columns)
    components = [
        component
        for component, column in component_columns.items()
        if eligible_components.get(component, False) and column in frame.columns
    ]
    if frame.is_empty() or "actual_winner" not in frame.columns:
        return _weight_payload(
            status="no_rows",
            method="configured_fallback",
            components=[],
            component_weights=configured_weights,
            configured_weights=configured_weights,
        )
    if not components:
        return _weight_payload(
            status="no_eligible_components",
            method="configured_fallback",
            components=[],
            component_weights=configured_weights,
            configured_weights=configured_weights,
            row_count=frame.height,
        )

    columns = [component_columns[component] for component in components]
    working = frame.select(["actual_winner", *columns]).drop_nulls()
    if working.height < min_rows:
        return _weight_payload(
            status="insufficient_rows",
            method="configured_fallback",
            components=components,
            component_weights=configured_weights,
            configured_weights=configured_weights,
            row_count=working.height,
        )

    y = working["actual_winner"].cast(pl.Float64).to_numpy()
    x = np.clip(working.select(columns).to_numpy().astype(np.float64), EPSILON, 1.0 - EPSILON)
    default_vector = _normalized_vector(configured_weights, components)
    if len(components) == 1:
        learned_vector = np.ones(1, dtype=np.float64)
        learned_log_loss = _log_loss(y, x[:, 0])
        configured_log_loss = learned_log_loss
        iterations = 0
    else:
        learned_vector, iterations = _projected_gradient(
            x=x,
            y=y,
            initial=default_vector,
            default=default_vector,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            l2_prior_strength=l2_prior_strength,
        )
        configured_log_loss = _log_loss(y, x @ default_vector)
        learned_log_loss = _log_loss(y, x @ learned_vector)

    learned_weights = {component: 0.0 for component in configured_weights}
    for component, weight in zip(components, learned_vector, strict=True):
        learned_weights[component] = float(weight)
    return _weight_payload(
        status="fitted",
        method="projected_gradient_simplex_log_loss",
        components=components,
        component_weights=learned_weights,
        configured_weights=configured_weights,
        row_count=working.height,
        iterations=iterations,
        configured_log_loss=configured_log_loss,
        learned_log_loss=learned_log_loss,
        learning_rate=learning_rate,
        l2_prior_strength=l2_prior_strength,
    )


def stacked_probability(
    frame: pl.DataFrame,
    component_columns: dict[str, str],
    component_weights: dict[str, float],
    *,
    fallback_column: str = "ensemble_probability",
) -> np.ndarray:
    """Apply learned simplex weights to a prediction frame."""

    if frame.is_empty():
        return np.array([], dtype=np.float64)
    components = [
        component
        for component, column in component_columns.items()
        if column in frame.columns and float(component_weights.get(component, 0.0)) > 0.0
    ]
    if not components:
        if fallback_column in frame.columns:
            return np.clip(
                frame[fallback_column].cast(pl.Float64).to_numpy(), EPSILON, 1.0 - EPSILON
            )
        return np.full(frame.height, 0.5, dtype=np.float64)

    columns = [component_columns[component] for component in components]
    weights = _normalized_vector(component_weights, components)
    x = frame.select(columns).to_numpy().astype(np.float64)
    fallback = (
        frame[fallback_column].cast(pl.Float64).to_numpy()
        if fallback_column in frame.columns
        else np.full(frame.height, 0.5, dtype=np.float64)
    )
    x = np.where(np.isfinite(x), x, fallback[:, None])
    return np.clip(x @ weights, EPSILON, 1.0 - EPSILON)


def fit_platt_calibration(
    probability: np.ndarray,
    actual: np.ndarray,
    *,
    min_rows: int = 30,
    ridge: float = 1e-3,
    max_iterations: int = 50,
    min_slope: float = 0.25,
    max_slope: float = 1.0,
    max_abs_intercept: float = 2.0,
) -> dict[str, Any]:
    """Fit a small-ridge Platt/logit calibration transform."""

    probability = np.asarray(probability, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(probability) & np.isfinite(actual)
    probability = np.clip(probability[mask], EPSILON, 1.0 - EPSILON)
    actual = actual[mask]
    identity = {
        "status": "identity",
        "method": "platt_logistic_ridge",
        "intercept": 0.0,
        "slope": 1.0,
        "row_count": len(actual),
        "ridge": ridge,
        "min_slope": min_slope,
        "max_slope": max_slope,
        "max_abs_intercept": max_abs_intercept,
    }
    if len(actual) < min_rows:
        return {**identity, "status": "insufficient_rows"}
    if len(np.unique(actual)) < 2 or len(np.unique(probability)) < 2:
        return {**identity, "status": "insufficient_variation"}

    logit_probability = np.log(probability / (1.0 - probability))
    design = np.column_stack([np.ones_like(logit_probability), logit_probability])
    beta = np.array([0.0, 1.0], dtype=np.float64)
    prior = beta.copy()
    iterations = 0
    for _iteration in range(max_iterations):
        iterations = _iteration + 1
        eta = design @ beta
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), EPSILON, None)
        hessian = design.T @ (weights[:, None] * design) + ridge * np.eye(2)
        gradient = design.T @ (actual - fitted) - ridge * (beta - prior)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        beta[0] = np.clip(beta[0], -max_abs_intercept, max_abs_intercept)
        beta[1] = np.clip(beta[1], min_slope, max_slope)
        if float(np.max(np.abs(step))) < 1e-8:
            break
    calibrated = apply_platt_calibration(probability, {"intercept": beta[0], "slope": beta[1]})
    return {
        "status": "fitted",
        "method": "platt_logistic_ridge",
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "row_count": len(actual),
        "ridge": ridge,
        "min_slope": min_slope,
        "max_slope": max_slope,
        "max_abs_intercept": max_abs_intercept,
        "iterations": iterations,
        "uncalibrated_log_loss": _log_loss(actual, probability),
        "calibrated_log_loss": _log_loss(actual, calibrated),
    }


def apply_platt_calibration(probability: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    output = probability.copy()
    mask = np.isfinite(probability)
    if not np.any(mask):
        return output
    clipped = np.clip(probability[mask], EPSILON, 1.0 - EPSILON)
    intercept = float(model.get("intercept", 0.0))
    slope = float(model.get("slope", 1.0))
    eta = intercept + slope * np.log(clipped / (1.0 - clipped))
    output[mask] = 1.0 / (1.0 + np.exp(-eta))
    return np.clip(output, EPSILON, 1.0 - EPSILON)


def _projected_gradient(
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    default: np.ndarray,
    max_iterations: int,
    learning_rate: float,
    l2_prior_strength: float,
) -> tuple[np.ndarray, int]:
    weights = initial.copy()
    best = weights.copy()
    best_loss = math.inf
    for iteration in range(max_iterations):
        probability = np.clip(x @ weights, EPSILON, 1.0 - EPSILON)
        gradient = x.T @ ((probability - y) / (probability * (1.0 - probability)))
        gradient /= len(y)
        gradient += l2_prior_strength * (weights - default)
        step = learning_rate / math.sqrt(iteration + 1.0)
        candidate = _project_simplex(weights - step * gradient)
        loss = _log_loss(y, x @ candidate) + 0.5 * l2_prior_strength * float(
            np.sum((candidate - default) ** 2)
        )
        if loss < best_loss:
            best_loss = loss
            best = candidate.copy()
        if float(np.max(np.abs(candidate - weights))) < 1e-9:
            return best, iteration + 1
        weights = candidate
    return best, max_iterations


def _project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    rho_candidates = ordered - (cumulative - 1.0) / np.arange(1, len(values) + 1) > 0
    if not np.any(rho_candidates):
        return np.full_like(values, 1.0 / len(values))
    rho = int(np.nonzero(rho_candidates)[0][-1])
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    total = projected.sum()
    return projected / total if total > 0 else np.full_like(values, 1.0 / len(values))


def _normalized_vector(weights: dict[str, float], components: list[str]) -> np.ndarray:
    values = np.array([max(float(weights.get(component, 0.0)), 0.0) for component in components])
    total = values.sum()
    if total <= 0:
        return np.full(len(components), 1.0 / len(components), dtype=np.float64)
    return values / total


def _float_weight_dict(
    default_weights: dict[str, float], component_columns: dict[str, str]
) -> dict[str, float]:
    return {
        component: float(default_weights.get(component, 0.0)) for component in component_columns
    }


def _weight_payload(
    *,
    status: str,
    method: str,
    components: list[str],
    component_weights: dict[str, float],
    configured_weights: dict[str, float],
    row_count: int = 0,
    iterations: int = 0,
    configured_log_loss: float | None = None,
    learned_log_loss: float | None = None,
    learning_rate: float | None = None,
    l2_prior_strength: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "method": method,
        "components": components,
        "component_weights": component_weights,
        "configured_weights": configured_weights,
        "row_count": row_count,
        "iterations": iterations,
    }
    if configured_log_loss is not None:
        payload["configured_log_loss"] = configured_log_loss
    if learned_log_loss is not None:
        payload["learned_log_loss"] = learned_log_loss
        payload["log_loss_delta_vs_configured"] = (
            learned_log_loss - configured_log_loss if configured_log_loss is not None else None
        )
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if l2_prior_strength is not None:
        payload["l2_prior_strength"] = l2_prior_strength
    return payload


def _log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    actual = np.asarray(actual, dtype=np.float64)
    return float(
        np.mean(-(actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability)))
    )
