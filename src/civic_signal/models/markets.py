from __future__ import annotations

from statistics import NormalDist

import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.common import clamp, normalize_rows


class MarketModel:
    component = "markets"

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.markets.is_empty():
            return normalize_rows([])
        settings = self.config.get("market_adjustments", {})
        min_open_interest = float(settings.get("min_open_interest", 1000))
        max_spread = float(settings.get("max_spread", 0.18))
        share_sigma = float(settings.get("probability_to_share_sigma", 0.08))
        favorite_longshot_bias = float(settings.get("favorite_longshot_bias", 0.0))
        rows: list[dict[str, object]] = []
        for key, group in bundle.markets.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            filtered = group.filter(
                (pl.col("open_interest") >= min_open_interest) & (pl.col("spread") <= max_spread)
            )
            if filtered.is_empty():
                continue
            latest = filtered.sort("observed_at").tail(1).row(0, named=True)
            probability = clamp(float(latest["probability"]), 0.001, 0.999)
            normal = NormalDist()
            inv_cdf = normal.inv_cdf(probability)
            implied_margin = inv_cdf * share_sigma
            density = max(normal.pdf(inv_cdf), 1e-3)
            spread_in_share = float(latest["spread"]) * share_sigma / density
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": probability,
                    "vote_share": clamp(0.5 + implied_margin - favorite_longshot_bias),
                    "uncertainty": max(spread_in_share, share_sigma),
                    "admitted": True,
                    "explanation": (
                        "Public market probability inverted through a calibrated-normal "
                        "share scale; spread mapped to share-units via local CDF derivative."
                    ),
                }
            )
        return normalize_rows(rows)
