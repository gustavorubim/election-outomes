from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from election_outcomes.config import ProjectContext, Scenario, ScenarioRegistry
from election_outcomes.features import (
    FeatureBuilder,
    FeatureBundle,
    filter_bundle_by_date,
    filter_results_before_cycle,
    subset_bundle,
)
from election_outcomes.ingest import SyncRunner
from election_outcomes.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from election_outcomes.normalize import CuratedDataBuilder
from election_outcomes.performance.benchmark import PerformanceBenchmark
from election_outcomes.reports import (
    DiagnosticsReport,
    MethodologySnapshot,
    ModelCard,
    PlotGenerator,
    SilverStyleBenchmark,
    benchmark_to_json,
)
from election_outcomes.scoring import (
    BacktestRunner,
    CycleEvaluationReport,
    ResultComparator,
    RewardEvaluator,
)
from election_outcomes.storage.io import read_json, write_json, write_parquet, write_text


class ForecastPipeline:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def sync(self) -> pl.DataFrame:
        return SyncRunner(self.context).run().manifest

    def build_features(self) -> FeatureBundle:
        CuratedDataBuilder(self.context).run()
        return FeatureBuilder(self.context).run()

    def run_forecast(
        self, as_of: str | None, run_id: str | None = None, scenario: str | None = None
    ) -> Path:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        as_of = as_of or (scenario_obj.default_as_of if scenario_obj else None)
        if as_of is None:
            raise ValueError("as_of is required unless the selected scenario defines default_as_of")
        self.sync()
        full_bundle = self.build_features()
        scenario_bundle = self._scenario_bundle(full_bundle, scenario_obj, include_cycle=True)
        bundle = self._active_bundle(scenario_bundle, as_of)
        target_cycle = self._target_cycle(bundle, scenario_obj)
        training_source = self._scenario_bundle(full_bundle, scenario_obj, include_cycle=False)
        training_bundle = filter_bundle_by_date(
            filter_results_before_cycle(training_source, target_cycle), as_of
        )
        model_config = self.context.read_yaml("model.yaml")
        model_config = self._apply_component_admission(model_config, scenario_obj)
        residual_covariance = self._load_residual_covariance(scenario_obj)
        source_manifest = pl.read_parquet(self.context.curated_dir / "source_manifest.parquet")
        fundamentals_model = FundamentalsModel(model_config).fit(training_bundle)
        polling_model = PollingModel(model_config, as_of=as_of)
        polling_estimates = polling_model.run(bundle)
        poll_trajectory = polling_model.trajectory(bundle)
        component_estimates = [
            polling_estimates,
            fundamentals_model.run(bundle),
            MarketModel(model_config).run(bundle),
            PublicSignalModel(
                trusted=bool(
                    model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(bundle),
        ]
        ensemble = EnsembleModel(model_config).run(bundle, component_estimates)
        outputs = SimulationEngine(
            model_config,
            residual_covariance=residual_covariance,
            holdovers=scenario_obj.holdovers if scenario_obj else None,
        ).run(bundle, ensemble)
        race_forecasts = self._attach_lineage(outputs.race_forecasts, model_config, source_manifest)
        race_catalog = self._attach_model_hash(bundle.race_catalog, model_config)
        out_dir = self.context.artifacts_dir / "runs" / run_id
        previous_fingerprint = self._read_reproducibility_fingerprint(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_parquet(race_catalog, out_dir / "race_catalog.parquet")
        write_parquet(race_forecasts, out_dir / "race_forecasts.parquet")
        write_parquet(outputs.draws, out_dir / "forecast_draws.parquet")
        write_parquet(outputs.control_forecasts, out_dir / "control_forecasts.parquet")
        write_parquet(outputs.ecosystem_forecasts, out_dir / "ecosystem_forecasts.parquet")
        write_parquet(poll_trajectory, out_dir / "poll_trajectory.parquet")
        write_json(outputs.performance, out_dir / "performance.json")
        write_parquet(
            source_manifest.with_columns(pl.lit("forecast_artifacts").alias("downstream_usage")),
            out_dir / "source_manifest.parquet",
        )
        backtest_artifacts = BacktestRunner(self.context)._evaluate(
            scenario_obj.family if scenario_obj else None
        )
        backtest_payload = backtest_artifacts.payload
        benchmark_covariance = (
            residual_covariance
            if residual_covariance is not None
            else backtest_artifacts.residual_covariance
        )
        silver_benchmark = SilverStyleBenchmark().evaluate(
            model_config=model_config,
            race_catalog=race_catalog,
            race_forecasts=race_forecasts,
            backtest_payload=backtest_payload,
            residual_covariance=benchmark_covariance,
            source_manifest=source_manifest,
            poll_trajectory=poll_trajectory,
        )
        plot_generator = PlotGenerator()
        plot_manifest = plot_generator.render_all(
            out_dir,
            race_catalog,
            race_forecasts,
            outputs.draws,
            outputs.control_forecasts,
            outputs.ecosystem_forecasts,
            backtest_artifacts.rolling_predictions
            if not backtest_artifacts.rolling_predictions.is_empty()
            else full_bundle.backtest_predictions,
            backtest_payload,
            silver_benchmark,
            poll_trajectory=poll_trajectory,
        )
        plot_generator.write_manifest(plot_manifest, out_dir)
        stability_metrics = self._stability_metrics(
            backtest_artifacts.rolling_predictions,
            scenario_obj.family if scenario_obj else None,
        )
        write_json(stability_metrics, out_dir / "stability_metrics.json")
        methodology = MethodologySnapshot().render(
            run_id, as_of, model_config, source_manifest.height
        )
        write_text(methodology, out_dir / "methodology_snapshot.md")
        write_text(benchmark_to_json(silver_benchmark), out_dir / "silver_benchmark.json")
        write_text(SilverStyleBenchmark.html(silver_benchmark), out_dir / "silver_benchmark.html")
        model_card = ModelCard().render(
            run_id=run_id,
            scenario=scenario_obj.metadata() if scenario_obj else None,
            model_config=model_config,
            backtest_payload=backtest_payload,
            component_admission=backtest_artifacts.component_admission,
            residual_covariance=residual_covariance,
            source_manifest=source_manifest,
            runtime_metadata={"fundamentals": fundamentals_model.fit_summary()},
            pollster_house_effects=polling_model.cached_house_effects,
        )
        write_text(model_card, out_dir / "model_card.md")
        self._write_reproducibility_fingerprint(out_dir, previous_fingerprint)
        reward_card = RewardEvaluator(model_config).evaluate(
            run_id,
            out_dir,
            race_forecasts,
            race_catalog,
            source_manifest,
            backtest_payload,
            plot_manifest,
            outputs.performance,
        )
        write_json(reward_card, out_dir / "reward_card.json")
        diagnostics = DiagnosticsReport().render(
            run_id,
            race_catalog,
            race_forecasts,
            source_manifest,
            backtest_payload,
            reward_card,
            plot_manifest,
            silver_benchmark,
            control_forecasts=outputs.control_forecasts,
            ecosystem_forecasts=outputs.ecosystem_forecasts,
        )
        write_text(diagnostics, out_dir / "diagnostics.html")
        return out_dir

    def run_backtest(
        self,
        run_id: str | None = None,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        self.build_features()
        return BacktestRunner(self.context).run(
            run_id, scenario=scenario, start_cycle=start_cycle, holdout_cycle=holdout_cycle
        )

    def run_benchmark(
        self,
        as_of: str,
        run_id: str | None = None,
        draws: int | None = None,
        repeats: int | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        full_bundle = self.build_features()
        bundle = self._active_bundle(full_bundle, as_of)
        model_config = self.context.read_yaml("model.yaml")
        result = PerformanceBenchmark(self.context).run(
            bundle=bundle,
            model_config=model_config,
            run_id=run_id,
            draws=draws,
            repeats=repeats,
        )
        return result.payload

    def compare_results(
        self,
        forecast_run_id: str,
        comparison_id: str | None = None,
        cycle: int | None = None,
        office_type: str | None = None,
        race_id: str | None = None,
    ) -> dict[str, Any]:
        comparison_id = comparison_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        self.build_features()
        forecast_run_dir = self.context.artifacts_dir / "runs" / forecast_run_id
        if not forecast_run_dir.exists():
            raise FileNotFoundError(f"Forecast run not found: {forecast_run_dir}")
        results = self._read_optional_curated("results")
        return ResultComparator().compare(
            forecast_run_dir=forecast_run_dir,
            curated_results=results,
            comparison_id=comparison_id,
            cycle=cycle,
            office_type=office_type,
            race_id=race_id,
        )

    def run_cycle_eval(
        self,
        cycles: list[int],
        as_of_mm_dd: str,
        run_id: str | None = None,
        scenario_template: str = "president_{cycle}_state",
        forecast_run_prefix: str = "eval",
        comparison_id: str = "actuals",
        office_type: str = "president",
        reuse_existing: bool = False,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("cycle-eval-%Y%m%dT%H%M%SZ")
        plan = self._cycle_eval_plan(cycles, as_of_mm_dd, scenario_template)
        output_dir = self.context.artifacts_dir / "cycle_evals" / run_id
        rows: list[dict[str, Any]] = []
        for item in plan:
            cycle = int(item["cycle"])
            scenario = str(item["scenario"])
            as_of = str(item["as_of"])
            as_of_slug = as_of_mm_dd.replace("-", "")
            forecast_run_id = f"{forecast_run_prefix}-{cycle}-{as_of_slug}"
            forecast_run_dir = self.context.artifacts_dir / "runs" / forecast_run_id
            if not reuse_existing or not self._forecast_run_complete(forecast_run_dir):
                forecast_run_dir = self.run_forecast(
                    as_of=as_of, run_id=forecast_run_id, scenario=scenario
                )
            comparison = self._cycle_eval_comparison(
                forecast_run_id=forecast_run_id,
                comparison_id=comparison_id,
                cycle=cycle,
                office_type=office_type,
                reuse_existing=reuse_existing,
            )
            scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
            rows.append(
                self._cycle_eval_row(
                    cycle=cycle,
                    as_of=as_of,
                    forecast_run_id=forecast_run_id,
                    forecast_run_dir=forecast_run_dir,
                    comparison=comparison,
                    output_dir=output_dir,
                    scenario=scenario_obj,
                )
            )
        return CycleEvaluationReport().render(
            rows=rows, output_dir=output_dir, run_id=run_id, as_of_mm_dd=as_of_mm_dd
        )

    def rebuild_report(self, run_id: str) -> Path:
        out_dir = self.context.artifacts_dir / "runs" / run_id
        model_config = self.context.read_yaml("model.yaml")
        race_catalog = pl.read_parquet(out_dir / "race_catalog.parquet")
        race_forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
        source_manifest = pl.read_parquet(out_dir / "source_manifest.parquet")
        forecast_draws = pl.read_parquet(out_dir / "forecast_draws.parquet")
        control_forecasts = pl.read_parquet(out_dir / "control_forecasts.parquet")
        ecosystem_forecasts = pl.read_parquet(out_dir / "ecosystem_forecasts.parquet")
        poll_trajectory = (
            pl.read_parquet(out_dir / "poll_trajectory.parquet")
            if (out_dir / "poll_trajectory.parquet").exists()
            else pl.DataFrame()
        )
        backtest_artifacts = BacktestRunner(self.context)._evaluate()
        backtest_payload = backtest_artifacts.payload
        latest_covariance = self._read_latest_covariance(None)
        benchmark_covariance = (
            latest_covariance
            if latest_covariance is not None
            else backtest_artifacts.residual_covariance
        )
        silver_benchmark = SilverStyleBenchmark().evaluate(
            model_config=model_config,
            race_catalog=race_catalog,
            race_forecasts=race_forecasts,
            backtest_payload=backtest_payload,
            residual_covariance=benchmark_covariance,
            source_manifest=source_manifest,
            poll_trajectory=poll_trajectory,
        )
        backtest_predictions = (
            backtest_artifacts.rolling_predictions
            if not backtest_artifacts.rolling_predictions.is_empty()
            else self._read_optional_curated("backtest_predictions")
        )
        plot_generator = PlotGenerator()
        plot_manifest = plot_generator.render_all(
            out_dir,
            race_catalog,
            race_forecasts,
            forecast_draws,
            control_forecasts,
            ecosystem_forecasts,
            backtest_predictions,
            backtest_payload,
            silver_benchmark,
            poll_trajectory=poll_trajectory,
        )
        plot_generator.write_manifest(plot_manifest, out_dir)
        reward_card_path = out_dir / "reward_card.json"
        reward_card = None
        if reward_card_path.exists():
            import json

            reward_card = json.loads(reward_card_path.read_text(encoding="utf-8"))
        diagnostics = DiagnosticsReport().render(
            run_id,
            race_catalog,
            race_forecasts,
            source_manifest,
            backtest_payload,
            reward_card,
            plot_manifest,
            silver_benchmark,
            control_forecasts=control_forecasts,
            ecosystem_forecasts=ecosystem_forecasts,
        )
        write_text(diagnostics, out_dir / "diagnostics.html")
        methodology = MethodologySnapshot().render(
            run_id, "existing", model_config, source_manifest.height
        )
        write_text(methodology, out_dir / "methodology_snapshot.md")
        write_text(benchmark_to_json(silver_benchmark), out_dir / "silver_benchmark.json")
        write_text(SilverStyleBenchmark.html(silver_benchmark), out_dir / "silver_benchmark.html")
        write_text(
            ModelCard().render(
                run_id=run_id,
                scenario=None,
                model_config=model_config,
                backtest_payload=backtest_payload,
                component_admission=backtest_artifacts.component_admission,
                residual_covariance=latest_covariance,
                source_manifest=source_manifest,
            ),
            out_dir / "model_card.md",
        )
        return out_dir

    def _read_optional_curated(self, name: str) -> pl.DataFrame:
        path = self.context.curated_dir / f"{name}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    def _cycle_eval_plan(
        self, cycles: list[int], as_of_mm_dd: str, scenario_template: str
    ) -> list[dict[str, Any]]:
        if not cycles:
            raise ValueError("cycles must include at least one cycle")
        parts = as_of_mm_dd.split("-")
        if len(parts) != 2:
            raise ValueError("as_of_mm_dd must use MM-DD format")
        registry = ScenarioRegistry.from_context(self.context)
        plan: list[dict[str, Any]] = []
        errors: list[str] = []
        for cycle in cycles:
            try:
                requested_as_of = date.fromisoformat(f"{cycle}-{as_of_mm_dd}")
            except ValueError:
                errors.append(f"{cycle}-{as_of_mm_dd} is not a valid date")
                continue
            try:
                scenario = scenario_template.format(cycle=cycle)
            except Exception as exc:  # pragma: no cover - defensive formatting detail
                errors.append(f"scenario_template failed for cycle {cycle}: {exc}")
                continue
            try:
                scenario_obj = registry.get(scenario)
            except ValueError:
                errors.append(f"unknown scenario {scenario!r} for cycle {cycle}")
                continue
            # If the requested as_of falls after the scenario's election day, use the
            # scenario default_as_of (typically election_date - 1 day) so the active
            # bundle filter (election_date >= cutoff) still includes the cycle's races.
            as_of = requested_as_of
            if scenario_obj is not None:
                election_str = (
                    scenario_obj.payload.get("election_date") if scenario_obj.payload else None
                )
                if election_str:
                    election_date = date.fromisoformat(str(election_str))
                    if as_of > election_date:
                        default_as_of = scenario_obj.default_as_of
                        if default_as_of:
                            as_of = date.fromisoformat(default_as_of)
                        else:
                            as_of = election_date
            plan.append({"cycle": cycle, "scenario": scenario, "as_of": as_of.isoformat()})
        if errors:
            raise ValueError("Invalid cycle evaluation request: " + "; ".join(errors))
        return plan

    @staticmethod
    def _forecast_run_complete(forecast_run_dir: Path) -> bool:
        required = [
            "control_forecasts.parquet",
            "diagnostics.html",
            "race_catalog.parquet",
            "race_forecasts.parquet",
            "reproducibility_fingerprint.json",
        ]
        return forecast_run_dir.exists() and all(
            (forecast_run_dir / name).exists() for name in required
        )

    def _cycle_eval_comparison(
        self,
        forecast_run_id: str,
        comparison_id: str,
        cycle: int,
        office_type: str,
        reuse_existing: bool,
    ) -> dict[str, Any]:
        comparison_dir = (
            self.context.artifacts_dir / "runs" / forecast_run_id / "comparisons" / comparison_id
        )
        summary_path = comparison_dir / "result_comparison_summary.json"
        if reuse_existing and summary_path.exists():
            payload = read_json(summary_path)
            payload["output_dir"] = str(comparison_dir)
            return payload
        return self.compare_results(
            forecast_run_id=forecast_run_id,
            comparison_id=comparison_id,
            cycle=cycle,
            office_type=office_type,
        )

    @staticmethod
    def _cycle_eval_row(
        cycle: int,
        as_of: str,
        forecast_run_id: str,
        forecast_run_dir: Path,
        comparison: dict[str, Any],
        output_dir: Path,
        scenario: Scenario | None = None,
    ) -> dict[str, Any]:
        control = pl.read_parquet(forecast_run_dir / "control_forecasts.parquet")
        winner = (
            control.sort("control_probability", descending=True).row(0, named=True)
            if not control.is_empty()
            else {}
        )
        actual_majority_party = ForecastPipeline._actual_majority_winner(
            comparison=comparison,
            control=control,
            scenario=scenario,
        )
        dem_row = (
            control.filter(pl.col("party") == "DEM").row(0, named=True)
            if not control.is_empty() and (control["party"] == "DEM").any()
            else {}
        )
        rep_row = (
            control.filter(pl.col("party") == "REP").row(0, named=True)
            if not control.is_empty() and (control["party"] == "REP").any()
            else {}
        )
        actual_winner_probabilities = comparison.get("actual_winner_probabilities") or []
        missed_states = [
            str(row.get("geography"))
            for row in actual_winner_probabilities
            if not bool(row.get("race_winner_correct"))
        ]
        electoral_college = comparison.get("electoral_college") or {}
        actual_ec_winner = electoral_college.get("actual_winner_party") or actual_majority_party
        forecast_ec_winner = winner.get("party")
        ec_winner_accuracy = (
            float(forecast_ec_winner == actual_ec_winner)
            if forecast_ec_winner is not None and actual_ec_winner is not None
            else None
        )
        comparison_dir = Path(str(comparison["output_dir"]))
        return {
            "cycle": cycle,
            "as_of": as_of,
            "forecast_run_id": forecast_run_id,
            "control_body": winner.get("control_body"),
            "majority_threshold": winner.get("control_threshold"),
            "forecast_ec_winner_party": winner.get("party"),
            "actual_ec_winner_party": actual_ec_winner,
            "state_topline_ec_winner_party": electoral_college.get("predicted_winner_party"),
            "state_topline_ec_winner_accuracy": comparison.get("ec_winner_accuracy"),
            "forecast_ec_win_probability": winner.get("control_probability"),
            "forecast_ec_p10": winner.get("seat_count_p10"),
            "forecast_ec_p50": winner.get("seat_count_p50"),
            "forecast_ec_p90": winner.get("seat_count_p90"),
            "dem_seat_count_mean": dem_row.get("seat_count_mean"),
            "rep_seat_count_mean": rep_row.get("seat_count_mean"),
            "dem_majority_probability": dem_row.get("majority_probability"),
            "rep_majority_probability": rep_row.get("majority_probability"),
            "ec_winner_accuracy": ec_winner_accuracy,
            "state_accuracy": comparison.get("state_accuracy")
            or comparison.get("winner_accuracy"),
            "state_accuracy_n": comparison.get("state_accuracy_n")
            or comparison.get("race_count"),
            "brier_score": comparison.get("brier_score"),
            "mean_absolute_vote_share_error": comparison.get("mean_absolute_vote_share_error"),
            "upset_count": comparison.get("upset_count"),
            "missed_states": ", ".join(missed_states),
            "race_count": comparison.get("race_count"),
            "diagnostics_path": os.path.relpath(forecast_run_dir / "diagnostics.html", output_dir),
            "comparison_path": os.path.relpath(
                comparison_dir / "result_comparison.html", output_dir
            ),
        }

    @staticmethod
    def _actual_majority_winner(
        comparison: dict[str, Any],
        control: pl.DataFrame,
        scenario: Scenario | None,
    ) -> str | None:
        if scenario is None or control.is_empty():
            return None
        race_outcomes = comparison.get("race_outcomes") or []
        if not race_outcomes:
            return None
        seats_by_party: dict[str, int] = {}
        for row in race_outcomes:
            party = row.get("actual_winner_party")
            if not party:
                continue
            seats_by_party[str(party).upper()] = seats_by_party.get(
                str(party).upper(), 0
            ) + int(row.get("seats") or 1)
        for party, holdovers in scenario.holdovers.items():
            seats_by_party[party] = seats_by_party.get(party, 0) + int(holdovers)
        threshold = (
            int(control["control_threshold"][0])
            if "control_threshold" in control.columns and control.height > 0
            else None
        )
        if threshold is None:
            return None
        winners = [party for party, seats in seats_by_party.items() if seats >= threshold]
        if len(winners) == 1:
            return winners[0]
        if not seats_by_party:
            return None
        return max(seats_by_party.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _active_bundle(bundle: FeatureBundle, as_of: str) -> FeatureBundle:
        cutoff = date.fromisoformat(as_of)
        active_catalog = bundle.race_catalog.filter(pl.col("election_date") >= cutoff)
        return filter_bundle_by_date(subset_bundle(bundle, active_catalog), as_of)

    @staticmethod
    def _scenario_bundle(
        bundle: FeatureBundle, scenario: Scenario | None, include_cycle: bool
    ) -> FeatureBundle:
        if scenario is None:
            return bundle
        return subset_bundle(bundle, scenario.filter_catalog(bundle.race_catalog, include_cycle))

    @staticmethod
    def _target_cycle(bundle: FeatureBundle, scenario: Scenario | None) -> int:
        if scenario and scenario.cycle is not None:
            return scenario.cycle
        cycles = sorted(int(value) for value in bundle.race_catalog["cycle"].unique().to_list())
        if not cycles:
            raise ValueError("No active races available for target cycle selection")
        return cycles[0]

    def _apply_component_admission(
        self, model_config: dict[str, Any], scenario: Scenario | None
    ) -> dict[str, Any]:
        updated = json.loads(json.dumps(model_config))
        key = scenario.family if scenario else "default"
        path = (
            self.context.artifacts_dir / "backtests" / "latest" / f"component_admission_{key}.json"
        )
        if not path.exists():
            updated["component_admission_source"] = {
                "status": "missing",
                "key": key,
                "path": str(path),
                "engine_using": "config_defaults",
            }
            return updated
        with path.open("r", encoding="utf-8") as handle:
            admission = json.load(handle)
        if isinstance(admission, dict):
            if isinstance(admission.get("trusted_components"), dict):
                updated["trusted_components"] = admission["trusted_components"]
            if isinstance(admission.get("component_weights"), dict):
                updated["component_weights"] = admission["component_weights"]
            updated["component_admission"] = admission
            updated["component_admission_source"] = {
                "status": "learned",
                "key": key,
                "path": str(path),
                "engine_using": str(admission.get("engine_using", "learned_admission")),
                "admission_status": str(admission.get("admission_status", "unknown")),
            }
        return updated

    def _load_residual_covariance(self, scenario: Scenario | None) -> pl.DataFrame | None:
        return self._read_latest_covariance(scenario.family if scenario else None)

    def _read_latest_covariance(self, key: str | None) -> pl.DataFrame | None:
        storage_key = key or "default"
        path = (
            self.context.artifacts_dir
            / "backtests"
            / "latest"
            / f"residual_covariance_{storage_key}.parquet"
        )
        return pl.read_parquet(path) if path.exists() else None

    @staticmethod
    def _attach_lineage(
        frame: pl.DataFrame, model_config: dict[str, Any], source_manifest: pl.DataFrame
    ) -> pl.DataFrame:
        model_hash = ForecastPipeline._config_hash(model_config)
        source_hashes = ",".join(sorted(source_manifest["content_hash"].drop_nulls().to_list()))
        return frame.with_columns(
            pl.lit(model_hash).alias("model_config_hash"),
            pl.lit(hashlib.sha256(source_hashes.encode()).hexdigest()).alias(
                "source_manifest_hash"
            ),
        )

    @staticmethod
    def _attach_model_hash(frame: pl.DataFrame, model_config: dict[str, Any]) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(ForecastPipeline._config_hash(model_config)).alias("model_config_hash")
        )

    @staticmethod
    def _config_hash(model_config: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _read_reproducibility_fingerprint(out_dir: Path) -> dict[str, Any] | None:
        path = out_dir / "reproducibility_fingerprint.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write_reproducibility_fingerprint(
        out_dir: Path, previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        # Scope: compares against `previous` from the same out_dir. Cross-environment
        # reproducibility (CI vs local) requires shipping the fingerprint as a baseline.
        stable_artifacts = {
            name: ForecastPipeline._stable_artifact_hash(out_dir / name)
            for name in [
                "race_catalog.parquet",
                "race_forecasts.parquet",
                "forecast_draws.parquet",
                "control_forecasts.parquet",
                "ecosystem_forecasts.parquet",
                "poll_trajectory.parquet",
                "source_manifest.parquet",
                "methodology_snapshot.md",
                "model_card.md",
                "silver_benchmark.html",
                "silver_benchmark.json",
                "plot_manifest.json",
                "stability_metrics.json",
                "performance.json",
            ]
        }
        combined_hash = hashlib.sha256(
            json.dumps(stable_artifacts, sort_keys=True).encode()
        ).hexdigest()
        previous_hash = str(previous.get("combined_hash")) if previous else None
        payload: dict[str, Any] = {
            "status": "fingerprint_generated",
            "excluded_fields": ["generated_at", "retrieved_at", "status"],
            "stable_artifacts": stable_artifacts,
            "combined_hash": combined_hash,
            "compared_to_previous": previous_hash is not None,
            "previous_combined_hash": previous_hash,
            "cross_run_verified": previous_hash == combined_hash if previous_hash else False,
        }
        write_json(payload, out_dir / "reproducibility_fingerprint.json")
        return payload

    @staticmethod
    def _stable_artifact_hash(path: Path) -> str:
        if path.suffix == ".parquet":
            frame = pl.read_parquet(path)
            ignored = [
                column
                for column in ("generated_at", "retrieved_at", "status")
                if column in frame.columns
            ]
            if ignored:
                frame = frame.drop(ignored)
            if frame.columns:
                frame = frame.sort(frame.columns)
            rows = frame.to_dicts()
            payload = json.dumps(rows, sort_keys=True, default=str)
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload_obj = json.load(handle)
            payload = json.dumps(payload_obj, sort_keys=True, default=str)
        else:
            payload = path.read_text(encoding="utf-8")
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _stability_metrics(
        rolling_predictions: pl.DataFrame, scenario_family: str | None
    ) -> dict[str, Any]:
        if (
            rolling_predictions.is_empty()
            or "ensemble_probability" not in rolling_predictions.columns
        ):
            return {
                "scenario_family": scenario_family,
                "available": False,
                "reason": "no rolling-origin probabilities available",
                "mean_absolute_probability_change": None,
                "max_probability_change": None,
                "row_count": rolling_predictions.height,
            }
        ordered = rolling_predictions.sort(["cycle", "race_id", "option_id", "as_of_offset_days"])
        changes = ordered.with_columns(
            pl.col("ensemble_probability")
            .diff()
            .over(["cycle", "race_id", "option_id"])
            .abs()
            .alias("absolute_probability_change")
        ).filter(pl.col("absolute_probability_change").is_not_null())
        if changes.is_empty():
            return {
                "scenario_family": scenario_family,
                "available": False,
                "reason": "only one as-of point per race/option",
                "mean_absolute_probability_change": None,
                "max_probability_change": None,
                "row_count": rolling_predictions.height,
            }
        by_offset = (
            changes.group_by("as_of_offset_days")
            .agg(
                pl.col("absolute_probability_change").mean().alias("mean_absolute_change"),
                pl.col("absolute_probability_change").max().alias("max_absolute_change"),
            )
            .sort("as_of_offset_days", descending=True)
            .to_dicts()
        )
        return {
            "scenario_family": scenario_family,
            "available": True,
            "row_count": rolling_predictions.height,
            "mean_absolute_probability_change": float(
                changes["absolute_probability_change"].mean()
            ),
            "max_probability_change": float(changes["absolute_probability_change"].max()),
            "by_as_of_offset_days": by_offset,
        }
