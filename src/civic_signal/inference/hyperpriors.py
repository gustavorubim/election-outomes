from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def search_hyperpriors(
    rolling_predictions: pl.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    required = {"ensemble_probability", "actual_winner"}
    if rolling_predictions.is_empty() or not required.issubset(rolling_predictions.columns):
        return {
            "status": "no_rows",
            "method": "rolling_origin_probability_shrinkage_grid",
            "row_count": 0,
            "selected": {},
            "candidates": [],
        }
    frame = rolling_predictions.select(["ensemble_probability", "actual_winner"]).drop_nulls()
    if frame.height < int(config.get("reward_thresholds", {}).get("minimum_backtest_rows", 30)):
        return {
            "status": "insufficient_rows",
            "method": "rolling_origin_probability_shrinkage_grid",
            "row_count": frame.height,
            "selected": {},
            "candidates": [],
        }
    probability = np.clip(
        frame["ensemble_probability"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6
    )
    actual = frame["actual_winner"].cast(pl.Float64).to_numpy()
    candidates: list[dict[str, Any]] = []
    for national_sigma in [0.01, 0.015, 0.02]:
        for election_day_extra_sd in [0.015, 0.025, 0.04]:
            for prior_strength in [0.35, 0.5, 0.75]:
                shrink = 1.0 / (1.0 + national_sigma * 8.0 + election_day_extra_sd * 12.0)
                adjusted = np.clip(0.5 + (probability - 0.5) * shrink, 1e-6, 1 - 1e-6)
                log_loss = float(
                    -np.mean(actual * np.log(adjusted) + (1.0 - actual) * np.log(1.0 - adjusted))
                )
                candidates.append(
                    {
                        "national_sigma": national_sigma,
                        "election_day_extra_sd": election_day_extra_sd,
                        "fundamentals_prior_strength": prior_strength,
                        "log_loss": log_loss,
                    }
                )
    selected = min(candidates, key=lambda row: float(row["log_loss"]))
    return {
        "status": "fitted",
        "method": "rolling_origin_probability_shrinkage_grid",
        "row_count": frame.height,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: float(row["log_loss"]))[:10],
    }
