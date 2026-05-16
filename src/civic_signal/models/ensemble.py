from __future__ import annotations

import json

import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.common import clamp, normalize_rows


class EnsembleModel:
    component = "ensemble"

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.weights = {
            str(key): float(value)
            for key, value in dict(config.get("component_weights", {})).items()
        }
        self.trusted = {
            str(key): bool(value)
            for key, value in dict(config.get("trusted_components", {})).items()
        }

    def run(self, bundle: FeatureBundle, component_estimates: list[pl.DataFrame]) -> pl.DataFrame:
        estimates = pl.concat(
            [df for df in component_estimates if not df.is_empty()], how="diagonal"
        )
        rows: list[dict[str, object]] = []
        if estimates.is_empty():
            return normalize_rows(rows)
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        for key, group in estimates.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            race = catalog[str(race_id)]
            if race["tier"] == "C":
                continue
            weighted_probability = weighted_share = weight_total = uncertainty_total = 0.0
            component_shares: list[tuple[float, float]] = []
            drivers: list[str] = []
            contributions: dict[str, dict[str, float]] = {}
            for row in group.iter_rows(named=True):
                component = str(row["component"])
                admitted = bool(row["admitted"]) and self.trusted.get(component, False)
                if not admitted:
                    continue
                weight = self.weights.get(component, 0.0)
                marginal = float(row["marginal_win_probability"])
                weighted_probability += weight * marginal
                vote_share = float(row["vote_share"])
                weighted_share += weight * vote_share
                uncertainty_total += weight * float(row["uncertainty"])
                weight_total += weight
                component_shares.append((weight, vote_share))
                drivers.append(component)
                contributions[component] = {
                    "weight": weight,
                    "marginal_win_probability": marginal,
                    "vote_share": vote_share,
                    "weighted_marginal_win_probability": weight * marginal,
                    "weighted_vote_share": weight * vote_share,
                }
            if weight_total <= 0:
                continue
            mean_share = weighted_share / weight_total
            disagreement = (
                sum(weight * (share - mean_share) ** 2 for weight, share in component_shares)
                / weight_total
            ) ** 0.5
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": clamp(weighted_probability / weight_total),
                    "vote_share": clamp(weighted_share / weight_total),
                    "uncertainty": uncertainty_total / weight_total,
                    "component_disagreement": disagreement,
                    "admitted": True,
                    "explanation": " + ".join(drivers),
                    "component_contributions": json.dumps(contributions, sort_keys=True),
                }
            )
        return self._normalize_vote_share_by_race(normalize_rows(rows))

    @staticmethod
    def _normalize_vote_share_by_race(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        totals = frame.group_by("race_id").agg(pl.col("vote_share").sum().alias("share_total"))
        return (
            frame.join(totals, on="race_id", how="left")
            .with_columns((pl.col("vote_share") / pl.col("share_total")).alias("vote_share"))
            .drop(["share_total"])
        )
