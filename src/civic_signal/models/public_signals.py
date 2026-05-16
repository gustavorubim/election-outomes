from __future__ import annotations

import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.common import clamp, logistic, normalize_rows


class PublicSignalModel:
    component = "public_signals"

    def __init__(self, trusted: bool = False) -> None:
        self.trusted = trusted

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.public_signals.is_empty():
            return normalize_rows([])
        rows: list[dict[str, object]] = []
        for key, group in bundle.public_signals.group_by(
            ["race_id", "option_id"], maintain_order=True
        ):
            race_id, option_id = key
            z_score = float(group.select(pl.col("z_score").mean()).item() or 0.0)
            leakage_checked = bool(group.select(pl.col("leakage_checked").all()).item())
            probability = logistic(z_score / 3.0)
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": probability,
                    "vote_share": clamp(0.5 + z_score * 0.025),
                    "uncertainty": 0.12,
                    "admitted": self.trusted and leakage_checked,
                    "explanation": (
                        "Public attention/news signal; experimental unless admitted by backtest."
                    ),
                }
            )
        return normalize_rows(rows)
