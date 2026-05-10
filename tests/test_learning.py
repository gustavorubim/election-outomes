from __future__ import annotations

import numpy as np
import polars as pl

from election_outcomes.scoring.learning import (
    apply_platt_calibration,
    fit_platt_calibration,
    fit_simplex_weights,
    stacked_probability,
)

COMPONENT_COLUMNS = {
    "polling": "polls_probability",
    "fundamentals": "fundamentals_probability",
    "markets": "markets_probability",
}


def _learning_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "actual_winner": [True, True, True, False, False, False] * 8,
            "polls_probability": [0.90, 0.84, 0.78, 0.25, 0.18, 0.12] * 8,
            "fundamentals_probability": [0.55, 0.52, 0.50, 0.48, 0.46, 0.44] * 8,
            "markets_probability": [0.60, 0.59, 0.58, 0.42, 0.41, 0.40] * 8,
            "ensemble_probability": [0.62, 0.60, 0.58, 0.40, 0.38, 0.36] * 8,
        }
    )


def test_simplex_learning_prefers_predictive_component() -> None:
    payload = fit_simplex_weights(
        _learning_frame(),
        COMPONENT_COLUMNS,
        {"polling": 0.34, "fundamentals": 0.33, "markets": 0.33},
        {"polling": True, "fundamentals": True, "markets": True},
        min_rows=10,
        max_iterations=120,
    )

    assert payload["status"] == "fitted"
    assert abs(sum(payload["component_weights"].values()) - 1.0) < 1e-9
    assert payload["component_weights"]["polling"] > payload["component_weights"]["fundamentals"]
    assert payload["learned_log_loss"] <= payload["configured_log_loss"]


def test_simplex_learning_fallback_paths() -> None:
    defaults = {"polling": 0.5, "fundamentals": 0.3, "markets": 0.2}
    empty = fit_simplex_weights(
        pl.DataFrame(),
        COMPONENT_COLUMNS,
        defaults,
        {"polling": True},
    )
    no_eligible = fit_simplex_weights(
        _learning_frame(),
        COMPONENT_COLUMNS,
        defaults,
        {"polling": False, "fundamentals": False, "markets": False},
    )
    one_component = fit_simplex_weights(
        _learning_frame(),
        COMPONENT_COLUMNS,
        defaults,
        {"polling": True, "fundamentals": False, "markets": False},
        min_rows=10,
    )

    assert empty["status"] == "no_rows"
    assert no_eligible["status"] == "no_eligible_components"
    assert one_component["component_weights"]["polling"] == 1.0
    assert one_component["component_weights"]["fundamentals"] == 0.0


def test_stacked_probability_uses_fallback_and_missing_values() -> None:
    frame = pl.DataFrame(
        {
            "polls_probability": [0.9, None],
            "fundamentals_probability": [0.2, 0.4],
            "ensemble_probability": [0.7, 0.6],
        }
    )
    stacked = stacked_probability(
        frame,
        {"polling": "polls_probability", "fundamentals": "fundamentals_probability"},
        {"polling": 0.75, "fundamentals": 0.25},
    )
    fallback = stacked_probability(
        frame,
        {"markets": "markets_probability"},
        {"markets": 1.0},
    )

    assert np.allclose(stacked, [0.725, 0.55])
    assert np.allclose(fallback, [0.7, 0.6])


def test_platt_calibration_bounds_and_identity_paths() -> None:
    probability = np.array([0.1, 0.2, 0.8, 0.9] * 10)
    actual = np.array([0.0, 0.0, 1.0, 1.0] * 10)
    default_fitted = fit_platt_calibration(probability, actual, min_rows=10)
    fitted = fit_platt_calibration(
        probability,
        actual,
        min_rows=10,
        max_slope=2.0,
        max_abs_intercept=0.5,
    )
    insufficient = fit_platt_calibration(probability[:4], actual[:4], min_rows=10)
    no_variation = fit_platt_calibration(probability, np.ones_like(actual), min_rows=10)
    calibrated = apply_platt_calibration(
        np.array([0.2, np.nan, 0.8]),
        {"intercept": fitted["intercept"], "slope": fitted["slope"]},
    )

    assert default_fitted["max_slope"] == 2.0
    assert default_fitted["slope"] <= 2.0
    assert fitted["status"] == "fitted"
    assert 0.25 <= fitted["slope"] <= 2.0
    assert abs(fitted["intercept"]) <= 0.5
    assert insufficient["status"] == "insufficient_rows"
    assert no_variation["status"] == "insufficient_variation"
    assert np.isnan(calibrated[1])
    assert calibrated[0] < calibrated[2]
