from __future__ import annotations

from typing import ClassVar

import numpy as np
import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.common import clamp, logit, normal_cdf, normalize_rows


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
        bayesian_cfg = dict((config or {}).get("bayesian", {}))
        prior_cfg = dict(bayesian_cfg.get("fundamentals_prior", {}))
        self.ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        self.min_training_rows = int(cfg.get("min_training_rows", 12))
        self.uncertainty = float(cfg.get("uncertainty", 0.08))
        self.structural_sd_logit = float(prior_cfg.get("structural_sd", 0.05))
        self.fallback_sd_logit = float(prior_cfg.get("fallback_sd_logit", 0.30))
        self.coefficients: dict[str, float] = dict(self.DEFAULT_COEFFICIENTS)
        self.intercept = 0.0
        self.feature_means: dict[str, float] = {}
        self.feature_stds: dict[str, float] = {}
        self.cv_predictive_variance: float | None = None
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
                        "marginal_win_probability": probability,
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

    def fit_summary(self) -> dict[str, object]:
        return {
            "component": self.component,
            "fit_status": self.fit_status,
            "training_rows": self.training_rows,
            "ridge_alpha": self.ridge_alpha,
            "cv_predictive_variance": self.cv_predictive_variance,
            "structural_sd_logit": self.structural_sd_logit,
            "fallback_sd_logit": self.fallback_sd_logit,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
        }

    def predictive_distribution(self, bundle: FeatureBundle) -> pl.DataFrame:
        if self.fit_status == "handpicked_default" and self.training_rows == 0:
            self._fit(bundle)
        rows: list[dict[str, object]] = []
        fundamentals = {row["race_id"]: row for row in bundle.fundamentals.iter_rows(named=True)}
        fitted = (
            self.cv_predictive_variance is not None and "standardized_ridge_fit" in self.fit_status
        )
        prior_method = "cv_ridge" if fitted else "handpicked_fallback"
        share_variance = (
            max(float(self.cv_predictive_variance), 1e-8)
            if self.cv_predictive_variance is not None
            else self.uncertainty**2
        )
        for race_id, group in bundle.options.group_by("race_id", maintain_order=True):
            race_key = race_id[0] if isinstance(race_id, tuple) else race_id
            fundamental = fundamentals.get(str(race_key))
            if fundamental is None:
                continue
            shares = self._raw_shares(group, fundamental)
            total = sum(shares.values()) or 1.0
            for option_id, share in shares.items():
                normalized_share = clamp(share / total)
                if fitted:
                    delta = max(normalized_share * (1.0 - normalized_share), 1e-6)
                    sd_logit = float(
                        np.sqrt(share_variance / (delta**2) + self.structural_sd_logit**2)
                    )
                else:
                    sd_logit = self.fallback_sd_logit
                rows.append(
                    {
                        "race_id": str(race_key),
                        "option_id": option_id,
                        "mean_share": normalized_share,
                        "mean_logit": logit(normalized_share),
                        "sd_logit": sd_logit,
                        "prior_method": prior_method,
                        "structural_sd_logit": self.structural_sd_logit,
                        "cv_predictive_variance": self.cv_predictive_variance,
                    }
                )
        schema = {
            "race_id": pl.String,
            "option_id": pl.String,
            "mean_share": pl.Float64,
            "mean_logit": pl.Float64,
            "sd_logit": pl.Float64,
            "prior_method": pl.String,
            "structural_sd_logit": pl.Float64,
            "cv_predictive_variance": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

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
                + self.intercept
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
            self.intercept = 0.0
            self.feature_means = {}
            self.feature_stds = {}
            self.cv_predictive_variance = None
            return
        feature_names = list(self.DEFAULT_COEFFICIENTS.keys())
        x = training.select(feature_names).to_numpy().astype(np.float64)
        y = training["target"].to_numpy().astype(np.float64)
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        x_scaled = (x - means) / stds
        design = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        penalty = self.ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefs_scaled = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        raw_coefs = coefs_scaled[1:] / stds
        self.intercept = float(coefs_scaled[0] - np.sum((coefs_scaled[1:] * means) / stds))
        self.coefficients = dict(zip(feature_names, raw_coefs.tolist(), strict=True))
        self.feature_means = dict(zip(feature_names, means.tolist(), strict=True))
        self.feature_stds = dict(zip(feature_names, stds.tolist(), strict=True))
        self.cv_predictive_variance = self._cv_predictive_variance(training, feature_names)
        self.fit_status = f"standardized_ridge_fit (n={training.height}, alpha={self.ridge_alpha})"

    def _cv_predictive_variance(
        self, training: pl.DataFrame, feature_names: list[str]
    ) -> float | None:
        if "cycle" not in training.columns:
            return None
        cycles = sorted(int(value) for value in training["cycle"].unique().to_list())
        if len(cycles) < 2:
            return None
        residuals: list[float] = []
        for cycle in cycles:
            train = training.filter(pl.col("cycle") != cycle)
            holdout = training.filter(pl.col("cycle") == cycle)
            if train.height < len(feature_names) + 1 or holdout.is_empty():
                continue
            beta = self._ridge_coefficients(train, feature_names)
            design = self._design_matrix(holdout, feature_names)
            predicted = design @ beta
            actual = holdout["target"].to_numpy().astype(np.float64)
            residuals.extend((predicted - actual).tolist())
        if not residuals:
            return None
        return float(max(np.mean(np.square(np.array(residuals, dtype=np.float64))), 1e-8))

    def _ridge_coefficients(self, frame: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
        x = frame.select(feature_names).to_numpy().astype(np.float64)
        y = frame["target"].to_numpy().astype(np.float64)
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        x_scaled = (x - means) / stds
        design = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        penalty = self.ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefs_scaled = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        raw_coefs = coefs_scaled[1:] / stds
        intercept = float(coefs_scaled[0] - np.sum((coefs_scaled[1:] * means) / stds))
        return np.concatenate([[intercept], raw_coefs])

    @staticmethod
    def _design_matrix(frame: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
        x = frame.select(feature_names).to_numpy().astype(np.float64)
        return np.column_stack([np.ones(x.shape[0]), x])

    @staticmethod
    def _training_frame(bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.results.is_empty() or bundle.options.is_empty() or bundle.fundamentals.is_empty():
            return pl.DataFrame()
        result_columns = ["race_id", "option_id", pl.col("vote_share").alias("actual_vote_share")]
        if "cycle" in bundle.results.columns:
            result_columns.append("cycle")
        results = bundle.results.select(result_columns)
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
        if "cycle" not in joined.columns:
            joined = joined.with_columns(pl.lit(None, dtype=pl.Int64).alias("cycle"))
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
        output = joined.select(
            [
                "cycle",
                "partisan_lean",
                "economic_index",
                "demographic_turnout_index",
                "incumbent",
                "fundraising_usd",
                "target",
            ]
        )
        return output.drop_nulls(
            [
                "partisan_lean",
                "economic_index",
                "demographic_turnout_index",
                "incumbent",
                "fundraising_usd",
                "target",
            ]
        )
