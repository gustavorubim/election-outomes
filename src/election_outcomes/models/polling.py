from __future__ import annotations

import math
from datetime import date
from typing import ClassVar

import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.common import clamp, normal_cdf, normalize_rows


class PollingModel:
    component = "polling"

    # Hand-picked placeholder modifiers. Replace with coefficients fit on historical
    # pollster-vs-actual residuals once `pollster_house_effects` carries learned values.
    POPULATION_WEIGHTS: ClassVar[dict[str, float]] = {"lv": 1.1, "rv": 1.0, "a": 0.85}
    METHODOLOGY_WEIGHTS: ClassVar[dict[str, float]] = {
        "live_phone": 1.1,
        "mixed": 1.05,
        "online": 0.95,
    }

    def __init__(self, config: dict[str, object] | None = None, as_of: str | None = None) -> None:
        polling_config = dict((config or {}).get("polling", {}))
        self.half_life_days = float(polling_config.get("half_life_days", 21))
        self.min_nonsampling_error = float(polling_config.get("min_nonsampling_error", 0.035))
        self.pollster_house_effects = {
            str(key): float(value)
            for key, value in dict(polling_config.get("pollster_house_effects", {})).items()
        }
        self.as_of = date.fromisoformat(as_of) if as_of else None

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.polls.is_empty():
            return normalize_rows([])
        as_of = self.as_of or bundle.polls.select(pl.col("end_date").max()).item()
        rows: list[dict[str, object]] = []
        for key, group in bundle.polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            weighted = 0.0
            total_weight = 0.0
            adjusted_shares: list[float] = []
            weights: list[float] = []
            for row in group.iter_rows(named=True):
                sample = max(float(row.get("sample_size") or 600), 1.0)
                pop_weight = self.POPULATION_WEIGHTS.get(str(row.get("population")), 1.0)
                method_weight = self.METHODOLOGY_WEIGHTS.get(str(row.get("methodology")), 1.0)
                sponsor_weight = 0.85 if str(row.get("sponsor_class")) != "nonpartisan" else 1.0
                time_weight = self._time_decay(row["end_date"], as_of)
                weight = sample * pop_weight * method_weight * sponsor_weight * time_weight
                share = clamp(
                    float(row["pct"]) / 100.0
                    - self._house_effect(str(row.get("pollster")), str(option_id)),
                    0.001,
                    0.999,
                )
                if weight <= 0:
                    continue
                weighted += weight * share
                total_weight += weight
                adjusted_shares.append(share)
                weights.append(weight)
            if total_weight <= 0:
                continue
            share = clamp(weighted / total_weight)
            uncertainty = self._posterior_sigma(share, total_weight, adjusted_shares, weights)
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": normal_cdf((share - 0.5) / uncertainty),
                    "vote_share": share,
                    "uncertainty": uncertainty,
                    "admitted": True,
                    "explanation": (
                        "Poll aggregation with sample-size inverse-variance weighting, "
                        "time decay, and posterior uncertainty."
                    ),
                }
            )
        return normalize_rows(rows)

    def _time_decay(self, end_date: object, as_of: date) -> float:
        poll_date = end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date))
        age_days = (as_of - poll_date).days
        if age_days < 0:
            return 0.0
        return 0.5 ** (age_days / max(self.half_life_days, 1.0))

    def _house_effect(self, pollster: str, option_id: str) -> float:
        return self.pollster_house_effects.get(
            f"{pollster}:{option_id}", self.pollster_house_effects.get(pollster, 0.0)
        )

    def _posterior_sigma(
        self, share: float, total_weight: float, shares: list[float], weights: list[float]
    ) -> float:
        sampling_sigma = math.sqrt(max(share * (1.0 - share), 1e-6) / max(total_weight, 1.0))
        dispersion = 0.0
        weight_sum = sum(weights)
        if len(shares) > 1 and weight_sum > 0:
            mean_share = (
                sum(value * weight for value, weight in zip(shares, weights, strict=True))
                / weight_sum
            )
            dispersion = math.sqrt(
                sum(
                    weight * (value - mean_share) ** 2
                    for value, weight in zip(shares, weights, strict=True)
                )
                / weight_sum
            )
        return max(math.sqrt(sampling_sigma**2 + dispersion**2), self.min_nonsampling_error)
