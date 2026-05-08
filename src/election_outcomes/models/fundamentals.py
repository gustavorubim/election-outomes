from __future__ import annotations

from typing import ClassVar

import numpy as np
import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.common import clamp, normal_cdf, normalize_rows


class FundamentalsModel:
    component = "fundamentals"

    DEFAULT_COEFFICIENTS: ClassVar[dict[str, float]] = {
        "partisan_lean": 1.0 / 100.0,
        "economic_index": 1.0 / 50.0,
        "demographic_turnout_index": 1.0 / 80.0,
        "incumbent": 0.01,
        "fundraising_usd": 1.0 / 1_000_000_000,
    }

    def __init__(self, config: dict[str, object] | None = None) -> None:
        cfg = dict((config or {}).get("fundamentals", {}))
        self.ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        self.min_training_rows = int(cfg.get("min_training_rows", 12))
        self.uncertainty = float(cfg.get("uncertainty", 0.08))
        self.coefficients: dict[str, float] = dict(self.DEFAULT_COEFFICIENTS)
        self.fit_status: str = "handpicked_default"
        self.training_rows: int = 0

    def fit(self, bundle: FeatureBundle) -> FundamentalsModel:
        self._fit(bundle)
        return self

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        if self.fit_status == "handpicked_default" and self.training_rows == 0:
            self._fit(bundle)
        rows: list[dict[str, object]] = []
        fundamentals = {row["race_id"]: row for row in bundle.fundamentals.iter_rows(named=True)}
        for race_id, group in bundle.options.group_by("race_id", maintain_order=True):
            race_key = race_id[0] if isinstance(race_id, tuple) else race_id
            fundamental = fundamentals.get(str(race_key))
            if fundamental is None:
                continue
            shares = self._raw_shares(group, fundamental)
            total = sum(shares.values()) or 1.0
            for option_id, share in shares.items():
                normalized_share = clamp(share / total)
                probability = normal_cdf((normalized_share - 0.5) / max(self.uncertainty, 1e-3))
                rows.append(
                    {
                        "race_id": str(race_key),
                        "option_id": option_id,
                        "component": self.component,
                        "win_probability": probability,
                        "vote_share": normalized_share,
                        "uncertainty": self.uncertainty,
                        "admitted": True,
                        "explanation": (
                            f"Fundamentals model ({self.fit_status}, "
                            f"training_rows={self.training_rows})."
                        ),
                    }
                )
        return normalize_rows(rows)

    def _raw_shares(
        self, options: pl.DataFrame, fundamental: dict[str, object]
    ) -> dict[str, float]:
        lean = float(fundamental.get("partisan_lean") or 0.0)
        economy = float(fundamental.get("economic_index") or 0.0)
        demographic = float(fundamental.get("demographic_turnout_index") or 0.0)
        coef = self.coefficients
        shares: dict[str, float] = {}
        for row in options.iter_rows(named=True):
            base = float(row.get("previous_vote_share") or 0.5)
            party = str(row.get("party") or "")
            sign = 1.0 if party in {"DEM", "YES"} else -1.0
            incumbent = 1.0 if bool(row.get("incumbent")) else 0.0
            finance = float(row.get("fundraising_usd") or 0.0)
            prediction = (
                base
                + coef["partisan_lean"] * lean * sign
                + coef["economic_index"] * economy * sign
                + coef["demographic_turnout_index"] * demographic * sign
                + coef["incumbent"] * incumbent
                + coef["fundraising_usd"] * finance
            )
            shares[str(row["option_id"])] = clamp(prediction, 0.05, 0.95)
        return shares

    def _fit(self, bundle: FeatureBundle) -> None:
        training = self._training_frame(bundle)
        self.training_rows = training.height
        if training.height < self.min_training_rows:
            self.fit_status = f"handpicked_default (n={training.height} < {self.min_training_rows})"
            self.coefficients = dict(self.DEFAULT_COEFFICIENTS)
            return
        feature_names = list(self.DEFAULT_COEFFICIENTS.keys())
        x = training.select(feature_names).to_numpy().astype(np.float64)
        y = training["target"].to_numpy().astype(np.float64)
        gram = x.T @ x + self.ridge_alpha * np.eye(x.shape[1])
        coefs = np.linalg.solve(gram, x.T @ y)
        self.coefficients = dict(zip(feature_names, coefs.tolist(), strict=True))
        self.fit_status = f"ridge_fit (n={training.height}, alpha={self.ridge_alpha})"

    @staticmethod
    def _training_frame(bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.results.is_empty() or bundle.options.is_empty() or bundle.fundamentals.is_empty():
            return pl.DataFrame()
        results = bundle.results.select(
            ["race_id", "option_id", pl.col("vote_share").alias("actual_vote_share")]
        )
        options = bundle.options.select(
            ["race_id", "option_id", "party", "incumbent", "previous_vote_share", "fundraising_usd"]
        )
        fundamentals = bundle.fundamentals.select(
            ["race_id", "partisan_lean", "economic_index", "demographic_turnout_index"]
        )
        joined = results.join(options, on=["race_id", "option_id"], how="inner").join(
            fundamentals, on="race_id", how="inner"
        )
        if joined.is_empty():
            return joined
        joined = joined.with_columns(
            pl.when(pl.col("party").is_in(["DEM", "YES"]))
            .then(1.0)
            .otherwise(-1.0)
            .alias("party_sign"),
            pl.col("incumbent").cast(pl.Float64).fill_null(0.0).alias("incumbent_value"),
            pl.col("fundraising_usd").fill_null(0.0).alias("fundraising_value"),
            pl.col("previous_vote_share").fill_null(0.5).alias("previous_share"),
        ).with_columns(
            (pl.col("partisan_lean").fill_null(0.0) * pl.col("party_sign")).alias("partisan_lean"),
            (pl.col("economic_index").fill_null(0.0) * pl.col("party_sign")).alias(
                "economic_index"
            ),
            (pl.col("demographic_turnout_index").fill_null(0.0) * pl.col("party_sign")).alias(
                "demographic_turnout_index"
            ),
            pl.col("incumbent_value").alias("incumbent"),
            pl.col("fundraising_value").alias("fundraising_usd"),
            (pl.col("actual_vote_share") - pl.col("previous_share")).alias("target"),
        )
        return joined.select(
            [
                "partisan_lean",
                "economic_index",
                "demographic_turnout_index",
                "incumbent",
                "fundraising_usd",
                "target",
            ]
        ).drop_nulls()
