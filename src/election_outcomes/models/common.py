from __future__ import annotations

import math
from collections.abc import Iterable

import polars as pl


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return min(high, max(low, value))


def empty_estimates(component: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "race_id": pl.String,
            "option_id": pl.String,
            "component": pl.String,
            "marginal_win_probability": pl.Float64,
            "vote_share": pl.Float64,
            "uncertainty": pl.Float64,
            "admitted": pl.Boolean,
            "explanation": pl.String,
        }
    ).with_columns(pl.lit(component).alias("component"))


def normalize_rows(rows: Iterable[dict[str, object]]) -> pl.DataFrame:
    frame = pl.DataFrame(list(rows))
    if frame.is_empty():
        return empty_estimates("empty")
    return frame
