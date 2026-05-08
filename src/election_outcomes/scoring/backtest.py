from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from election_outcomes.config import ProjectContext, Scenario, ScenarioRegistry
from election_outcomes.features import (
    FeatureBuilder,
    FeatureBundle,
    filter_bundle_by_date,
    subset_bundle,
)
from election_outcomes.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
)
from election_outcomes.models.common import clamp, normal_cdf
from election_outcomes.scoring.metrics import score_predictions
from election_outcomes.storage.io import write_json, write_parquet


@dataclass(frozen=True)
class BacktestArtifacts:
    payload: dict[str, Any]
    rolling_predictions: pl.DataFrame
    component_admission: dict[str, Any]
    residual_covariance: pl.DataFrame


class BacktestRunner:
    COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "baseline": "baseline_probability",
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
        "ensemble": "ensemble_probability",
    }

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def evaluate(
        self,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
    ) -> dict[str, object]:
        return self._evaluate(scenario, start_cycle, holdout_cycle).payload

    def _evaluate(
        self,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
    ) -> BacktestArtifacts:
        bundle = FeatureBuilder(self.context).run()
        model_config = self.context.read_yaml("model.yaml")
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        rolling_predictions = self._rolling_origin_predictions(
            bundle=bundle,
            model_config=model_config,
            scenario=scenario_obj,
            start_cycle=start_cycle,
            holdout_cycle=holdout_cycle,
        )
        config = self.context.read_yaml("backtests.yaml")
        minimum_rows = int(config.get("minimum_rows_for_trust", 30))
        metrics = {
            component: score_predictions(rolling_predictions, column)
            for component, column in self.COMPONENT_COLUMNS.items()
            if column in rolling_predictions.columns
        }
        baseline_brier = metrics.get("baseline", {}).get("brier")
        ablations = {}
        for component, values in metrics.items():
            if component == "baseline" or baseline_brier is None:
                continue
            ablations[component] = {
                "brier_delta_vs_baseline": values["brier"] - baseline_brier,
                "beats_or_matches_baseline": values["brier"] <= baseline_brier,
            }
        rolling = self._rolling_origin_summary(rolling_predictions)
        sample_size_too_small = rolling_predictions.height < minimum_rows
        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "rolling_origin_component_refit",
            "scenario": scenario,
            "start_cycle": start_cycle,
            "holdout_cycle": holdout_cycle,
            "rolling_origin_executed": rolling["executed"],
            "rolling_origin": rolling,
            "minimum_rows_for_trust": minimum_rows,
            "sample_size_too_small": sample_size_too_small,
            "row_count": rolling_predictions.height,
            "metrics": metrics,
            "ablations": ablations,
        }
        component_admission = self._component_admission(
            payload=payload,
            ablations=ablations,
            model_config=model_config,
            scenario=scenario_obj,
        )
        covariance = self._residual_covariance(rolling_predictions)
        return BacktestArtifacts(payload, rolling_predictions, component_admission, covariance)

    def _rolling_origin_predictions(
        self,
        bundle: FeatureBundle,
        model_config: dict[str, Any],
        scenario: Scenario | None,
        start_cycle: int | None,
        holdout_cycle: int | None,
    ) -> pl.DataFrame:
        base_catalog = (
            scenario.filter_catalog(bundle.race_catalog, include_cycle=False)
            if scenario
            else bundle.race_catalog
        )
        target_catalog = base_catalog
        if start_cycle is not None:
            target_catalog = target_catalog.filter(pl.col("cycle") >= start_cycle)
        if holdout_cycle is not None:
            target_cycles = [holdout_cycle]
        else:
            target_cycles = sorted(
                int(value) for value in target_catalog["cycle"].unique().to_list()
            )
        frames: list[pl.DataFrame] = []
        for target_cycle in target_cycles:
            train_catalog = base_catalog.filter(pl.col("cycle") < target_cycle)
            test_catalog = base_catalog.filter(pl.col("cycle") == target_cycle)
            if train_catalog.is_empty() or test_catalog.is_empty():
                continue
            as_of = self._cycle_as_of(test_catalog)
            train_bundle = filter_bundle_by_date(subset_bundle(bundle, train_catalog), as_of)
            test_bundle = filter_bundle_by_date(subset_bundle(bundle, test_catalog), as_of)
            predictions = self._predict_cycle(
                train_bundle=train_bundle,
                test_bundle=test_bundle,
                target_cycle=target_cycle,
                as_of=as_of,
                model_config=model_config,
            )
            if not predictions.is_empty():
                frames.append(predictions)
        return pl.concat(frames, how="diagonal_relaxed") if frames else self._empty_predictions()

    def _predict_cycle(
        self,
        train_bundle: FeatureBundle,
        test_bundle: FeatureBundle,
        target_cycle: int,
        as_of: str,
        model_config: dict[str, Any],
    ) -> pl.DataFrame:
        component_estimates = [
            PollingModel(model_config, as_of=as_of).run(test_bundle),
            FundamentalsModel(model_config).fit(train_bundle).run(test_bundle),
            MarketModel(model_config).run(test_bundle),
            PublicSignalModel(
                trusted=bool(
                    model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(test_bundle),
        ]
        ensemble = EnsembleModel(model_config).run(test_bundle, component_estimates)
        rows: list[dict[str, Any]] = []
        actuals = {
            (row["race_id"], row["option_id"]): row
            for row in test_bundle.results.iter_rows(named=True)
        }
        component_maps = {
            "polls_probability": self._component_probability(component_estimates[0]),
            "fundamentals_probability": self._component_probability(component_estimates[1]),
            "markets_probability": self._component_probability(component_estimates[2]),
            "public_signals_probability": self._component_probability(component_estimates[3]),
            "ensemble_probability": self._component_probability(ensemble),
        }
        ensemble_share = self._component_share(ensemble)
        ensemble_uncertainty = self._component_uncertainty(ensemble)
        catalog = {row["race_id"]: row for row in test_bundle.race_catalog.iter_rows(named=True)}
        for option in test_bundle.options.iter_rows(named=True):
            key = (option["race_id"], option["option_id"])
            actual = actuals.get(key)
            if actual is None:
                continue
            previous_share = float(option.get("previous_vote_share") or 0.5)
            uncertainty = ensemble_uncertainty.get(key, 0.08)
            predicted_share = ensemble_share.get(key, previous_share)
            race = catalog[str(option["race_id"])]
            row = {
                "race_id": option["race_id"],
                "cycle": target_cycle,
                "as_of": as_of,
                "geography": race.get("geography"),
                "office_type": race.get("office_type"),
                "option_id": option["option_id"],
                "party": option.get("party"),
                "actual_winner": bool(actual["winner"]),
                "actual_vote_share": float(actual["vote_share"]),
                "baseline_probability": normal_cdf((previous_share - 0.5) / 0.08),
                "predicted_vote_share": predicted_share,
                "lower_90": clamp(predicted_share - 1.645 * uncertainty, 0.0, 1.0),
                "upper_90": clamp(predicted_share + 1.645 * uncertainty, 0.0, 1.0),
            }
            for column, values in component_maps.items():
                row[column] = values.get(key, row["baseline_probability"])
            rows.append(row)
        return pl.DataFrame(rows) if rows else self._empty_predictions()

    @staticmethod
    def _component_probability(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "marginal_win_probability" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["marginal_win_probability"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _component_share(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "vote_share" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["vote_share"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _component_uncertainty(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "uncertainty" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["uncertainty"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _cycle_as_of(test_catalog: pl.DataFrame) -> str:
        election_date = test_catalog.select(pl.col("election_date").min()).item()
        if not hasattr(election_date, "isoformat"):
            election_date = datetime.fromisoformat(str(election_date)).date()
        return (election_date - timedelta(days=1)).isoformat()

    @staticmethod
    def _rolling_origin_summary(frame: pl.DataFrame) -> dict[str, Any]:
        if frame.is_empty() or "cycle" not in frame.columns:
            return {
                "executed": False,
                "method": "rolling_origin_component_refit",
                "reason": "no scored holdout cycles",
                "cycles": [],
                "per_cycle_metrics": {},
            }
        cycles = sorted(int(value) for value in frame["cycle"].unique().to_list())
        return {
            "executed": True,
            "method": "rolling_origin_component_refit",
            "cycles": cycles,
            "per_cycle_metrics": {
                str(cycle): score_predictions(
                    frame.filter(pl.col("cycle") == cycle), "ensemble_probability"
                )
                for cycle in cycles
            },
        }

    @staticmethod
    def _component_admission(
        payload: dict[str, Any],
        ablations: dict[str, dict[str, Any]],
        model_config: dict[str, Any],
        scenario: Scenario | None,
    ) -> dict[str, Any]:
        trustworthy = bool(payload["rolling_origin_executed"]) and not bool(
            payload["sample_size_too_small"]
        )
        configured = {
            str(key): bool(value)
            for key, value in dict(model_config.get("trusted_components", {})).items()
        }
        trusted_components = {}
        for component, configured_trust in configured.items():
            if component == "public_signals":
                trusted_components[component] = False
                continue
            if trustworthy:
                trusted_components[component] = bool(
                    ablations.get(component, {}).get("beats_or_matches_baseline", False)
                )
            else:
                trusted_components[component] = configured_trust
        return {
            "generated_at": payload["generated_at"],
            "scenario": scenario.name if scenario else None,
            "scenario_family": scenario.family if scenario else None,
            "admission_status": "trusted" if trustworthy else "experimental_insufficient_rows",
            "trusted_components": trusted_components,
            "component_weights": dict(model_config.get("component_weights", {})),
            "ablations": ablations,
            "minimum_rows_for_trust": payload["minimum_rows_for_trust"],
            "row_count": payload["row_count"],
        }

    @staticmethod
    def _residual_covariance(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame(
                schema={
                    "row_group": pl.Utf8,
                    "column_group": pl.Utf8,
                    "covariance": pl.Float64,
                    "correlation": pl.Float64,
                    "sample_size": pl.Int64,
                    "shrinkage": pl.Float64,
                }
            )
        residuals = (
            frame.with_columns(
                (pl.col("predicted_vote_share") - pl.col("actual_vote_share")).alias("residual")
            )
            .group_by(["cycle", "geography"])
            .agg(pl.col("residual").mean().alias("residual"))
        )
        pivot = residuals.pivot(
            index="cycle", on="geography", values="residual", aggregate_function="mean"
        )
        groups = [column for column in pivot.columns if column != "cycle"]
        if not groups:
            return BacktestRunner._residual_covariance(pl.DataFrame())
        matrix_rows = []
        data = pivot.select(groups).fill_null(0.0).to_numpy().astype(np.float64)
        if data.shape[0] > 1:
            covariance = np.cov(data, rowvar=False)
            covariance = np.atleast_2d(covariance)
        else:
            covariance = np.diag(np.maximum(data[0] ** 2, 0.0004))
        shrinkage = 0.25
        covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
        diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        for row_index, row_group in enumerate(groups):
            for column_index, column_group in enumerate(groups):
                denom = diagonal[row_index] * diagonal[column_index]
                correlation = covariance[row_index, column_index] / denom if denom else 0.0
                matrix_rows.append(
                    {
                        "row_group": row_group,
                        "column_group": column_group,
                        "covariance": float(covariance[row_index, column_index]),
                        "correlation": float(correlation),
                        "sample_size": int(data.shape[0]),
                        "shrinkage": shrinkage,
                    }
                )
        return pl.DataFrame(matrix_rows)

    @staticmethod
    def _empty_predictions() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "race_id": pl.Utf8,
                "cycle": pl.Int64,
                "as_of": pl.Utf8,
                "geography": pl.Utf8,
                "office_type": pl.Utf8,
                "option_id": pl.Utf8,
                "party": pl.Utf8,
                "actual_winner": pl.Boolean,
                "actual_vote_share": pl.Float64,
                "baseline_probability": pl.Float64,
                "polls_probability": pl.Float64,
                "fundamentals_probability": pl.Float64,
                "markets_probability": pl.Float64,
                "public_signals_probability": pl.Float64,
                "ensemble_probability": pl.Float64,
                "predicted_vote_share": pl.Float64,
                "lower_90": pl.Float64,
                "upper_90": pl.Float64,
            }
        )

    def run(
        self,
        run_id: str,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
    ) -> dict[str, object]:
        artifacts = self._evaluate(scenario, start_cycle, holdout_cycle)
        out_dir = self.context.artifacts_dir / "backtests" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows = [
            {"component": component, **values}
            for component, values in artifacts.payload["metrics"].items()
        ]
        write_parquet(pl.DataFrame(metrics_rows), out_dir / "scorecard.parquet")
        write_json(artifacts.payload, out_dir / "scorecard.json")
        write_parquet(artifacts.rolling_predictions, out_dir / "rolling_predictions.parquet")
        write_json(artifacts.component_admission, out_dir / "component_admission.json")
        write_parquet(artifacts.residual_covariance, out_dir / "residual_covariance.parquet")
        self._write_latest_artifacts(
            scenario=scenario,
            component_admission=artifacts.component_admission,
            residual_covariance=artifacts.residual_covariance,
        )
        return artifacts.payload

    def _write_latest_artifacts(
        self,
        scenario: str | None,
        component_admission: dict[str, Any],
        residual_covariance: pl.DataFrame,
    ) -> None:
        key = component_admission.get("scenario_family") or scenario or "default"
        latest_dir = self.context.artifacts_dir / "backtests" / "latest"
        write_json(component_admission, latest_dir / f"component_admission_{key}.json")
        write_parquet(residual_covariance, latest_dir / f"residual_covariance_{key}.parquet")
        index_path = latest_dir / "index.json"
        index = {}
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    index = loaded
        index[str(key)] = {
            "component_admission": f"component_admission_{key}.json",
            "residual_covariance": f"residual_covariance_{key}.parquet",
        }
        write_json(index, index_path)
