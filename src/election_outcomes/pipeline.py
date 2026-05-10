from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, date, datetime, timedelta
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
from election_outcomes.inference.daily_update import run_daily_update
from election_outcomes.inference.fundamentals_prior import build_fundamentals_prior
from election_outcomes.ingest import SyncRunner
from election_outcomes.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from election_outcomes.models.polling import resolve_inference_engine
from election_outcomes.normalize import CuratedDataBuilder
from election_outcomes.observability import get_reporter
from election_outcomes.performance.benchmark import PerformanceBenchmark
from election_outcomes.reports import (
    DiagnosticsReport,
    MethodologySnapshot,
    ModelCard,
    PlotGenerator,
    RaceDetailRenderer,
    SilverStyleBenchmark,
    benchmark_to_json,
)
from election_outcomes.scoring import (
    BacktestRunner,
    CycleEvaluationReport,
    ResultComparator,
    RewardEvaluator,
    score_predictions,
)
from election_outcomes.storage.io import read_json, write_json, write_parquet, write_text

_FINGERPRINT_EXCLUDED_FIELDS = frozenset(
    {"generated_at", "retrieved_at", "status", "elapsed_seconds"}
)


class ForecastPipeline:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def sync(self) -> pl.DataFrame:
        return SyncRunner(self.context).run().manifest

    def build_features(self) -> FeatureBundle:
        CuratedDataBuilder(self.context).run()
        return FeatureBuilder(self.context).run()

    def run_forecast(
        self,
        as_of: str | None,
        run_id: str | None = None,
        scenario: str | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
        quiet: bool = False,
    ) -> Path:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        reporter = get_reporter(quiet=quiet)
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
        backtest_artifacts = BacktestRunner(self.context)._evaluate(
            scenario_obj.family if scenario_obj else None,
            inference_engine="kalman",
        )
        model_config = self._apply_component_admission(
            model_config,
            scenario_obj,
            admission_override=backtest_artifacts.component_admission,
        )
        inference_engine = resolve_inference_engine(model_config, inference_engine)
        model_config["_inference_engine"] = inference_engine
        if bayesian_backend:
            model_config["_bayesian_backend"] = bayesian_backend.lower().strip()
        residual_covariance = self._load_residual_covariance(scenario_obj)
        recalibration_map = self._load_recalibration_map(scenario_obj)
        source_manifest = pl.read_parquet(self.context.curated_dir / "source_manifest.parquet")
        fundamentals_model = FundamentalsModel(model_config).fit(training_bundle)
        fundamentals_prior = None
        if inference_engine == "bayes":
            with reporter.phase("Fundamentals prior", total_steps=2):
                fundamentals_prior = build_fundamentals_prior(
                    fundamentals_model, bundle, model_config
                )
                reporter.posterior_summary(
                    fundamentals_prior.frame.select(
                        ["race_id", "option_id", "mean_share", "sd_logit", "prior_method"]
                    )
                )
                model_config["_fundamentals_prior_rows"] = fundamentals_prior.frame.to_dicts()
        polling_model = PollingModel(model_config, as_of=as_of, inference_engine=inference_engine)
        if inference_engine == "bayes":
            with reporter.phase("Bayesian polling posterior", total_steps=3):
                polling_estimates = polling_model.run(bundle)
                poll_trajectory = polling_model.trajectory(bundle)
                polling_diagnostics = polling_model.diagnostics(bundle)
                posterior_draws = polling_model.posterior_draws(bundle)
                reporter.status(
                    "posterior draws: "
                    f"{posterior_draws.height} rows; "
                    f"race_options={polling_diagnostics.get('race_option_count')}"
                )
        else:
            polling_estimates = polling_model.run(bundle)
            poll_trajectory = polling_model.trajectory(bundle)
            polling_diagnostics = polling_model.diagnostics(bundle)
            posterior_draws = None
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
            holdovers=scenario_obj.holdover_caucus_seats if scenario_obj else None,
        ).run(bundle, ensemble, posterior_draws=posterior_draws)
        race_forecasts = self._attach_lineage(outputs.race_forecasts, model_config, source_manifest)
        race_catalog = self._attach_model_hash(bundle.race_catalog, model_config)
        seat_posterior = self._seat_posterior(
            outputs.draws,
            bundle.race_catalog,
            model_config,
            scenario_obj.holdover_caucus_seats if scenario_obj else None,
        )
        out_dir = self.context.artifacts_dir / "runs" / run_id
        previous_fingerprint = self._read_reproducibility_fingerprint(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_parquet(race_catalog, out_dir / "race_catalog.parquet")
        write_parquet(race_forecasts, out_dir / "race_forecasts.parquet")
        write_parquet(outputs.draws, out_dir / "forecast_draws.parquet")
        write_parquet(outputs.control_forecasts, out_dir / "control_forecasts.parquet")
        write_parquet(outputs.ecosystem_forecasts, out_dir / "ecosystem_forecasts.parquet")
        write_parquet(poll_trajectory, out_dir / "poll_trajectory.parquet")
        if recalibration_map is not None and not recalibration_map.is_empty():
            write_parquet(recalibration_map, out_dir / "recalibration_map.parquet")
        if inference_engine == "bayes":
            posterior_artifact = self._attach_lineage(
                posterior_draws if posterior_draws is not None else pl.DataFrame(),
                model_config,
                source_manifest,
            )
            write_parquet(posterior_artifact, out_dir / "posterior_draws.parquet")
            state_space_trajectory = self._attach_lineage(
                poll_trajectory,
                model_config,
                source_manifest,
            )
            write_parquet(state_space_trajectory, out_dir / "state_space_trajectory.parquet")
            pollster_house_effects = self._attach_lineage(
                self._pollster_house_effect_frame(polling_model.cached_house_effects),
                model_config,
                source_manifest,
            )
            write_parquet(pollster_house_effects, out_dir / "pollster_house_effects.parquet")
            fundamentals_prior_artifact = self._attach_lineage(
                fundamentals_prior.frame if fundamentals_prior is not None else pl.DataFrame(),
                model_config,
                source_manifest,
            )
            write_parquet(fundamentals_prior_artifact, out_dir / "fundamentals_prior.parquet")
            seat_posterior_artifact = self._attach_lineage(
                seat_posterior,
                model_config,
                source_manifest,
            )
            write_parquet(seat_posterior_artifact, out_dir / "seat_posterior.parquet")
            for body in ["senate", "house", "governor"]:
                body_frame = seat_posterior_artifact.filter(pl.col("control_body") == body)
                if not body_frame.is_empty():
                    write_parquet(body_frame, out_dir / f"{body}_seat_posterior.parquet")
            office_methodology = self._write_bayesian_office_methodology(
                out_dir=out_dir,
                bundle=bundle,
                posterior_draws=posterior_draws if posterior_draws is not None else pl.DataFrame(),
                seat_posterior=seat_posterior,
                model_config=model_config,
                source_manifest=source_manifest,
                posterior_diagnostics=polling_diagnostics,
            )
            polling_diagnostics = {
                **polling_diagnostics,
                "office_methodology": office_methodology,
            }
            write_json(
                {
                    **polling_diagnostics,
                    "model_config_hash": self._config_hash(model_config),
                    "source_manifest_hash": self._source_manifest_hash(source_manifest),
                },
                out_dir / "posterior_diagnostics.json",
            )
        write_json(outputs.performance, out_dir / "performance.json")
        write_parquet(
            source_manifest.with_columns(pl.lit("forecast_artifacts").alias("downstream_usage")),
            out_dir / "source_manifest.parquet",
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
            posterior_draws=posterior_draws,
            posterior_diagnostics=polling_diagnostics,
            fundamentals_prior=fundamentals_prior.frame
            if fundamentals_prior is not None
            else pl.DataFrame(),
        )
        plot_generator.write_manifest(plot_manifest, out_dir)
        race_detail_paths = RaceDetailRenderer().render_all(
            artifact_dir=out_dir,
            race_catalog=race_catalog,
            race_forecasts=race_forecasts,
            forecast_draws=outputs.draws,
            poll_trajectory=poll_trajectory,
        )
        write_json(
            {"races": race_detail_paths},
            out_dir / "race_detail_index.json",
        )
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
            runtime_metadata={
                "fundamentals": fundamentals_model.fit_summary(),
                "polling": polling_diagnostics,
                "fundamentals_prior": self._fundamentals_prior_metadata(
                    fundamentals_prior.frame if fundamentals_prior is not None else pl.DataFrame()
                ),
                "recalibration_map": self._recalibration_map_metadata(recalibration_map),
                "office_methodology": polling_diagnostics.get("office_methodology", {}),
            },
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
            polling_diagnostics if inference_engine == "bayes" else None,
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
            posterior_diagnostics=polling_diagnostics,
            fundamentals_prior=fundamentals_prior.frame
            if fundamentals_prior is not None
            else pl.DataFrame(),
        )
        write_text(diagnostics, out_dir / "diagnostics.html")
        if inference_engine == "bayes":
            reporter.tree(
                "Bayesian artifacts written",
                [
                    "posterior_draws.parquet",
                    "state_space_trajectory.parquet",
                    "pollster_house_effects.parquet",
                    "posterior_diagnostics.json",
                    "fundamentals_prior.parquet",
                    "senate_joint_posterior.parquet",
                    "house_hierarchical_posterior.parquet",
                    "cross_office_posterior.parquet",
                    "diagnostics.html",
                ],
            )
            reporter.save(out_dir)
        return out_dir

    def run_backtest(
        self,
        run_id: str | None = None,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        self.build_features()
        return BacktestRunner(self.context).run(
            run_id,
            scenario=scenario,
            start_cycle=start_cycle,
            holdout_cycle=holdout_cycle,
            inference_engine=inference_engine,
            bayesian_backend=bayesian_backend,
        )

    def refresh_hyperpriors(
        self,
        run_id: str | None = None,
        scenarios: list[str] | None = None,
        inference_engine: str | None = None,
        holdout_cycle: int | None = None,
        schedule_days: int | None = None,
        bayesian_backend: str | None = None,
    ) -> dict[str, Any]:
        model_config = self.context.read_yaml("model.yaml")
        refresh_config = dict(model_config.get("hyperprior_refresh", {}))
        run_id = run_id or datetime.now(UTC).strftime("hyperprior-refresh-%Y%m%dT%H%M%SZ")
        configured_scenarios = refresh_config.get("scenarios", ["president_state"])
        if scenarios is None:
            scenarios = (
                [str(item) for item in configured_scenarios]
                if isinstance(configured_scenarios, list)
                else [str(configured_scenarios)]
            )
        engine = str(inference_engine or refresh_config.get("inference_engine", "bayes"))
        backend = bayesian_backend or refresh_config.get("bayesian_backend")
        cadence_days = int(schedule_days or refresh_config.get("schedule_days", 30))
        generated_at = datetime.now(UTC)
        out_dir = self.context.artifacts_dir / "hyperprior_refreshes" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        self.sync()
        self.build_features()
        scenario_rows = []
        runner = BacktestRunner(self.context)
        for scenario in scenarios:
            artifacts = runner._evaluate(
                scenario=scenario,
                holdout_cycle=holdout_cycle,
                inference_engine=engine,
                bayesian_backend=backend,
            )
            scenario_dir = out_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            hyperpriors = dict(artifacts.payload.get("bayesian_hyperpriors", {}))
            latest = self._read_latest_hyperpriors(scenario)
            comparison = self._hyperprior_refresh_comparison(
                scenario=scenario,
                candidate=hyperpriors,
                current=latest,
            )
            write_json(artifacts.payload, scenario_dir / "scorecard_candidate.json")
            write_json(artifacts.component_admission, scenario_dir / "component_admission.json")
            write_json(hyperpriors, scenario_dir / f"bayesian_hyperpriors_{scenario}.json")
            write_json(comparison, scenario_dir / "comparison_report.json")
            write_parquet(
                artifacts.rolling_predictions,
                scenario_dir / "rolling_predictions_candidate.parquet",
            )
            write_parquet(artifacts.recalibration_map, scenario_dir / "recalibration_map.parquet")
            write_parquet(
                artifacts.residual_covariance,
                scenario_dir / "residual_covariance.parquet",
            )
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "candidate_status": hyperpriors.get("status", "unknown"),
                    "row_count": artifacts.payload.get("row_count", 0),
                    "current_status": latest.get("status", "missing"),
                    "candidate_selected": hyperpriors.get("selected", {}),
                    "current_selected": latest.get("selected", {}),
                    "comparison": comparison,
                    "artifact_dir": str(scenario_dir),
                }
            )

        manifest = {
            "run_id": run_id,
            "status": "candidate_refresh",
            "promoted": False,
            "promotion_policy": "manual_explicit_review_required",
            "generated_at": generated_at.isoformat(),
            "schedule_days": cadence_days,
            "next_due_after": (generated_at + timedelta(days=cadence_days)).date().isoformat(),
            "inference_engine": engine,
            "holdout_cycle": holdout_cycle,
            "scenarios": scenario_rows,
            "output_dir": str(out_dir),
        }
        write_json(manifest, out_dir / "hyperprior_refresh_manifest.json")
        write_text(
            self._hyperprior_refresh_report(manifest),
            out_dir / "comparison_report.md",
        )
        return manifest

    def assess_methodology_readiness(
        self,
        *,
        run_id: str | None = None,
        forecast_run_id: str | None = None,
        bayes_backtest_run_id: str | None = None,
        legacy_backtest_run_id: str | None = None,
        scenario: str = "president_state",
    ) -> dict[str, Any]:
        from election_outcomes.verification import MethodologyReadinessAuditor

        return MethodologyReadinessAuditor(self.context).run(
            run_id=run_id,
            forecast_run_id=forecast_run_id,
            bayes_backtest_run_id=bayes_backtest_run_id,
            legacy_backtest_run_id=legacy_backtest_run_id,
            scenario=scenario,
        )

    def verify_historical_calibration(
        self,
        *,
        run_id: str | None = None,
        scenario: str = "2022-midterm-historical-calibration",
        as_of: str | None = None,
        inference_engine: str = "bayes",
        bayesian_backend: str | None = "nuts",
        quiet: bool = False,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("historical-calibration-%Y%m%dT%H%M%SZ")
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        as_of = as_of or (scenario_obj.default_as_of if scenario_obj else None)
        if as_of is None:
            raise ValueError("as_of is required unless the selected scenario defines default_as_of")

        forecast_run_id = f"{run_id}-forecast"
        forecast_dir = self.run_forecast(
            as_of=as_of,
            run_id=forecast_run_id,
            scenario=scenario,
            inference_engine=inference_engine,
            bayesian_backend=bayesian_backend,
            quiet=quiet,
        )
        results = self._read_optional_curated("results")
        comparison = self._historical_calibration_frame(
            forecast_run_dir=forecast_dir,
            curated_results=results,
            scenario=scenario_obj,
        )
        office_rows = self._historical_calibration_office_rows(comparison)
        gates = self._historical_calibration_gates(
            office_rows=office_rows,
            expected_offices=self._expected_calibration_offices(scenario_obj),
        )
        diagnostics_path = forecast_dir / "posterior_diagnostics.json"
        posterior_diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {}
        payload: dict[str, Any] = {
            "run_id": run_id,
            "scenario": scenario,
            "as_of": as_of,
            "forecast_run_id": forecast_run_id,
            "forecast_output_dir": str(forecast_dir),
            "output_dir": str(self.context.artifacts_dir / "historical_calibration" / run_id),
            "generated_at": datetime.now(UTC).isoformat(),
            "inference_engine": inference_engine,
            "bayesian_backend": bayesian_backend,
            "scope": "compact_fixture_historical_calibration",
            "row_count": comparison.height,
            "race_count": comparison["race_id"].n_unique()
            if not comparison.is_empty() and "race_id" in comparison.columns
            else 0,
            "office_calibration": office_rows,
            "gates": gates,
            "passed": all(bool(gate["passed"]) for gate in gates.values()),
            "posterior_diagnostics": self._historical_calibration_diagnostics(
                posterior_diagnostics
            ),
            "office_methodology": self._historical_calibration_office_methodology(
                posterior_diagnostics
            ),
            "note": (
                "This is a compact fixture audit for the Phase 4/5/7 calibration gates. "
                "It proves the gate is runnable; a production-sized historical panel is still "
                "required before claiming full historical calibration coverage."
            ),
        }
        output_dir = Path(str(payload["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(comparison, output_dir / "historical_calibration_comparison.parquet")
        write_parquet(pl.DataFrame(office_rows), output_dir / "office_calibration.parquet")
        write_json(payload, output_dir / "historical_calibration.json")
        write_text(
            self._historical_calibration_report(payload), output_dir / "historical_calibration.md"
        )
        return payload

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

    def run_daily_update(self, anchor_run_id: str, as_of: str) -> dict[str, Any]:
        model_config = self.context.read_yaml("model.yaml")
        anchor_run_dir = self.context.artifacts_dir / "runs" / anchor_run_id
        result = run_daily_update(
            anchor_run_dir=anchor_run_dir,
            as_of=as_of,
            config=model_config,
        )
        reward_status = self._refresh_daily_update_reward(anchor_run_dir)
        return {
            "anchor_run_id": anchor_run_id,
            "as_of": as_of,
            "strategy": result.strategy,
            "output_dir": str(result.output_dir),
            "posterior_summary_rows": result.posterior_summary.height,
            "fallback_used": result.fallback_used,
            "needs_full_refit": result.needs_full_refit,
            "diagnostics": result.diagnostics,
            "reward_status": reward_status,
        }

    def run_phase0_spike(
        self,
        run_id: str | None = None,
        scenario: str = "president_state",
        holdout_cycle: int = 2024,
        bayesian_backend: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("phase0-%Y%m%dT%H%M%SZ")
        self.sync()
        self.build_features()
        from election_outcomes.inference.spike import run_phase0_comparison

        result = run_phase0_comparison(
            self.context,
            run_id=run_id,
            scenario=scenario,
            holdout_cycle=holdout_cycle,
            bayesian_backend=bayesian_backend,
        )
        return {
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "go_no_go": result.payload["go_no_go"],
            "row_count": {row["engine"]: row["row_count"] for row in result.payload["comparison"]},
        }

    def run_phase0b_spike(self, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("phase0b-%Y%m%dT%H%M%SZ")
        from election_outcomes.inference.acceleration_spike import run_phase0b_acceleration

        result = run_phase0b_acceleration(self.context, run_id=run_id)
        gate = result.payload["acceleration_gate"]
        return {
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "global_smc_rejected": result.payload["global_smc_rejected"],
            "selected_strategy": result.payload["selected_strategy"],
            "combined_global_smc_ess_ratio": gate["combined_global_smc_ess_ratio"],
            "geometry_gate": result.payload["geometry_gate"],
            "acceleration_gate": gate,
        }

    def _write_bayesian_office_methodology(
        self,
        *,
        out_dir: Path,
        bundle: FeatureBundle,
        posterior_draws: pl.DataFrame,
        seat_posterior: pl.DataFrame,
        model_config: dict[str, Any],
        source_manifest: pl.DataFrame,
        posterior_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from election_outcomes.inference.cross_office import summarize_cross_office
        from election_outcomes.inference.house_hierarchical import summarize_house_hierarchical
        from election_outcomes.inference.senate_joint import summarize_senate_joint

        outputs = {
            "senate_joint": (
                "senate_joint_posterior.parquet",
                summarize_senate_joint(
                    bundle, posterior_draws, seat_posterior, posterior_diagnostics
                ),
            ),
            "house_hierarchical": (
                "house_hierarchical_posterior.parquet",
                summarize_house_hierarchical(
                    bundle, posterior_draws, model_config, posterior_diagnostics
                ),
            ),
            "cross_office": (
                "cross_office_posterior.parquet",
                summarize_cross_office(
                    bundle, posterior_draws, model_config, posterior_diagnostics
                ),
            ),
        }
        diagnostics: dict[str, Any] = {}
        for key, (artifact_name, result) in outputs.items():
            diagnostics[key] = result.diagnostics
            if result.posterior.is_empty():
                continue
            write_parquet(
                self._attach_lineage(result.posterior, model_config, source_manifest),
                out_dir / artifact_name,
            )
        return diagnostics

    @staticmethod
    def _refresh_daily_update_reward(run_dir: Path) -> bool | None:
        reward_path = run_dir / "reward_card.json"
        update_path = run_dir / "latest_daily_update.json"
        if not reward_path.exists() or not update_path.exists():
            return None
        reward_card = read_json(reward_path)
        latest_update = read_json(update_path)
        if not isinstance(reward_card, dict) or not isinstance(latest_update, dict):
            return None
        passed = bool(latest_update.get("quality_passed")) and not bool(
            latest_update.get("needs_full_refit")
        )
        rewards = reward_card.setdefault("rewards", {})
        rewards["R15_daily_update_quality"] = {
            "passed": passed,
            "metric": latest_update,
            "detail": (
                "Daily update, when present, must pass its strategy-specific quality gate "
                "and avoid full-refit triggers."
            ),
        }
        reward_card["generated_at"] = datetime.now(UTC).isoformat()
        write_json(reward_card, reward_path)
        return passed

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
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
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
                    as_of=as_of,
                    run_id=forecast_run_id,
                    scenario=scenario,
                    inference_engine=inference_engine,
                    bayesian_backend=bayesian_backend,
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
        backtest_artifacts = BacktestRunner(self.context)._evaluate(inference_engine="kalman")
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
        posterior_draws = (
            pl.read_parquet(out_dir / "posterior_draws.parquet")
            if (out_dir / "posterior_draws.parquet").exists()
            else pl.DataFrame()
        )
        posterior_diagnostics = (
            read_json(out_dir / "posterior_diagnostics.json")
            if (out_dir / "posterior_diagnostics.json").exists()
            else None
        )
        fundamentals_prior = (
            pl.read_parquet(out_dir / "fundamentals_prior.parquet")
            if (out_dir / "fundamentals_prior.parquet").exists()
            else pl.DataFrame()
        )
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
            posterior_draws=posterior_draws,
            posterior_diagnostics=posterior_diagnostics,
            fundamentals_prior=fundamentals_prior,
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
            posterior_diagnostics=posterior_diagnostics,
            fundamentals_prior=fundamentals_prior,
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

    def verify_run(self, run_id: str) -> dict[str, Any]:
        out_dir = self.context.artifacts_dir / "runs" / run_id
        checks: list[dict[str, Any]] = []
        if not out_dir.exists():
            payload = {
                "run_id": run_id,
                "passed": False,
                "checks": [
                    {
                        "name": "run_directory",
                        "passed": False,
                        "detail": f"Run directory not found: {out_dir}",
                    }
                ],
            }
            return payload
        required = [
            "race_catalog.parquet",
            "race_forecasts.parquet",
            "forecast_draws.parquet",
            "control_forecasts.parquet",
            "ecosystem_forecasts.parquet",
            "source_manifest.parquet",
            "diagnostics.html",
            "reward_card.json",
            "methodology_snapshot.md",
            "model_card.md",
            "silver_benchmark.json",
            "plot_manifest.json",
            "poll_trajectory.parquet",
            "stability_metrics.json",
            "performance.json",
            "reproducibility_fingerprint.json",
        ]
        missing = [name for name in required if not (out_dir / name).exists()]
        checks.append(
            {
                "name": "required_artifacts",
                "passed": not missing,
                "detail": {"missing": missing, "required_count": len(required)},
            }
        )
        if (out_dir / "posterior_draws.parquet").exists():
            checks.extend(self._verify_posterior_artifacts(out_dir))
        if (out_dir / "state_space_trajectory.parquet").exists():
            checks.append(self._verify_schema_artifact(out_dir, "state_space_trajectory.parquet"))
        if (out_dir / "pollster_house_effects.parquet").exists():
            checks.append(self._verify_schema_artifact(out_dir, "pollster_house_effects.parquet"))
        for artifact_name in [
            "senate_seat_posterior.parquet",
            "house_seat_posterior.parquet",
            "governor_seat_posterior.parquet",
            "senate_joint_posterior.parquet",
            "house_hierarchical_posterior.parquet",
            "cross_office_posterior.parquet",
        ]:
            if (out_dir / artifact_name).exists():
                checks.append(self._verify_schema_artifact(out_dir, artifact_name))
        if (out_dir / "posterior_history.parquet").exists():
            checks.append(self._verify_posterior_history(out_dir))
        if (out_dir / "timeout_failover_audit.json").exists():
            checks.append(self._verify_timeout_failover_audit(out_dir))
        if (out_dir / "recalibration_map.parquet").exists():
            checks.append(self._verify_recalibration_map(out_dir))
        checks.append(self._verify_plot_manifest(out_dir))
        checks.append(self._verify_reward_card(out_dir))
        payload = {
            "run_id": run_id,
            "passed": all(bool(check["passed"]) for check in checks),
            "checks": checks,
            "output_dir": str(out_dir),
        }
        write_json(payload, out_dir / "verification.json")
        return payload

    @staticmethod
    def _historical_calibration_frame(
        forecast_run_dir: Path,
        curated_results: pl.DataFrame,
        scenario: Scenario | None,
    ) -> pl.DataFrame:
        race_catalog = pl.read_parquet(forecast_run_dir / "race_catalog.parquet")
        race_forecasts = pl.read_parquet(forecast_run_dir / "race_forecasts.parquet")
        race_meta = race_catalog.select(
            [
                "race_id",
                "cycle",
                "election_date",
                "geography_type",
                "geography",
                "office_type",
                "race_type",
                "tier",
                "tier_reason",
            ]
        ).unique()
        metadata_columns = [
            "cycle",
            "election_date",
            "geography_type",
            "geography",
            "office_type",
            "race_type",
            "tier",
            "tier_reason",
        ]
        forecasts = race_forecasts.drop(metadata_columns, strict=False).join(
            race_meta, on="race_id", how="left"
        )
        if scenario and scenario.cycle is not None:
            forecasts = forecasts.filter(pl.col("cycle") == scenario.cycle)
        expected = ForecastPipeline._expected_calibration_offices(scenario)
        if expected:
            forecasts = forecasts.filter(pl.col("office_type").is_in(expected))
        actuals = curated_results.select(
            [
                "race_id",
                "option_id",
                pl.col("vote_share").alias("actual_vote_share"),
                pl.col("turnout").alias("actual_turnout"),
                pl.col("winner").alias("actual_winner"),
            ]
        )
        comparison = forecasts.join(actuals, on=["race_id", "option_id"], how="inner")
        if comparison.is_empty():
            return comparison
        return comparison.filter(pl.col("winner_probability").is_not_null()).with_columns(
            pl.col("vote_share_p05").alias("lower_90"),
            pl.col("vote_share_p95").alias("upper_90"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share")).alias("vote_share_error"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share"))
            .abs()
            .alias("absolute_vote_share_error"),
        )

    @staticmethod
    def _historical_calibration_office_rows(comparison: pl.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if comparison.is_empty():
            return rows
        for key, group in comparison.group_by("office_type", maintain_order=True):
            office = key[0] if isinstance(key, tuple) else key
            metrics = score_predictions(group, "winner_probability")
            rows.append(
                {
                    "office_type": str(office),
                    "row_count": group.height,
                    "race_count": group["race_id"].n_unique(),
                    "brier": ForecastPipeline._json_safe_float(metrics.get("brier")),
                    "log_score": ForecastPipeline._json_safe_float(metrics.get("log_score")),
                    "calibration_intercept": ForecastPipeline._json_safe_float(
                        metrics.get("calibration_intercept")
                    ),
                    "calibration_slope": ForecastPipeline._json_safe_float(
                        metrics.get("calibration_slope")
                    ),
                    "expected_calibration_error": ForecastPipeline._json_safe_float(
                        metrics.get("expected_calibration_error")
                    ),
                    "expected_calibration_error_bins": ForecastPipeline._json_safe_float(
                        metrics.get("expected_calibration_error_bins")
                    ),
                    "interval_90_coverage": ForecastPipeline._json_safe_float(
                        metrics.get("interval_90_coverage")
                    ),
                    "winner_accuracy": ForecastPipeline._json_safe_float(
                        group.group_by("race_id")
                        .agg(
                            pl.col("winner_probability")
                            .sort_by("winner_probability", descending=True)
                            .first()
                            .alias("top_probability"),
                            pl.col("actual_winner")
                            .sort_by("winner_probability", descending=True)
                            .first()
                            .alias("top_actual_winner"),
                        )["top_actual_winner"]
                        .cast(pl.Float64)
                        .mean()
                    ),
                }
            )
        return sorted(rows, key=lambda row: str(row["office_type"]))

    @staticmethod
    def _historical_calibration_gates(
        office_rows: list[dict[str, Any]],
        expected_offices: list[str],
    ) -> dict[str, dict[str, Any]]:
        by_office = {str(row["office_type"]): row for row in office_rows}

        def office_gate(name: str, max_ece: float, phase: str) -> dict[str, Any]:
            row = by_office.get(name)
            ece = row.get("expected_calibration_error") if row else None
            race_count = int(row.get("race_count") or 0) if row else 0
            passed = ece is not None and race_count > 0 and float(ece) <= max_ece
            return {
                "phase": phase,
                "office_type": name,
                "passed": passed,
                "max_expected_calibration_error": max_ece,
                "expected_calibration_error": ece,
                "race_count": race_count,
            }

        per_office = {
            office: office_gate(office, 0.06, "phase7_cross_office") for office in expected_offices
        }
        return {
            "phase4_senate": office_gate("senate", 0.05, "phase4_senate"),
            "phase5_house": office_gate("house", 0.05, "phase5_house"),
            "phase7_cross_office": {
                "phase": "phase7_cross_office",
                "passed": all(bool(row["passed"]) for row in per_office.values()),
                "expected_offices": expected_offices,
                "per_office": per_office,
            },
        }

    @staticmethod
    def _expected_calibration_offices(scenario: Scenario | None) -> list[str]:
        if scenario is None:
            return ["senate", "house", "governor"]
        raw = scenario.payload.get("expected_offices") or scenario.payload.get("office_types")
        if isinstance(raw, list):
            return [str(value) for value in raw]
        office = scenario.payload.get("office_type")
        return [str(office)] if office else ["senate", "house", "governor"]

    @staticmethod
    def _historical_calibration_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "engine",
            "backend",
            "num_chains",
            "num_warmup",
            "num_samples",
            "nuts_sample_count",
            "r_hat_max",
            "ess_min",
            "r_hat_available",
            "ess_available",
            "divergences",
            "fallback_used",
        ]
        return {key: diagnostics.get(key) for key in keys if key in diagnostics}

    @staticmethod
    def _historical_calibration_office_methodology(
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        raw = diagnostics.get("office_methodology")
        if not isinstance(raw, dict):
            return {}
        summary: dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            summary[str(key)] = {
                field: value.get(field)
                for field in [
                    "engine",
                    "status",
                    "state_space_nuts_fitted",
                    "r_hat_max",
                    "ess_min",
                    "divergences",
                ]
                if field in value
            }
        return summary

    @staticmethod
    def _historical_calibration_report(payload: dict[str, Any]) -> str:
        lines = [
            "# Historical Calibration Audit",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- Scenario: `{payload['scenario']}`",
            f"- Forecast run id: `{payload['forecast_run_id']}`",
            f"- As of: `{payload['as_of']}`",
            f"- Inference engine: `{payload['inference_engine']}`",
            f"- Bayesian backend: `{payload.get('bayesian_backend')}`",
            f"- Passed: `{payload['passed']}`",
            "",
            "## Office Calibration",
            "",
            "| Office | Races | Rows | ECE | Brier | Log score | 90% coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        row_template = (
            "| {office} | {races} | {rows} | {ece} | {brier} | {log_score} | {coverage} |"
        )
        for row in payload.get("office_calibration", []):
            lines.append(
                row_template.format(
                    office=row.get("office_type"),
                    races=row.get("race_count"),
                    rows=row.get("row_count"),
                    ece=ForecastPipeline._format_optional_float(
                        row.get("expected_calibration_error")
                    ),
                    brier=ForecastPipeline._format_optional_float(row.get("brier")),
                    log_score=ForecastPipeline._format_optional_float(row.get("log_score")),
                    coverage=ForecastPipeline._format_optional_float(
                        row.get("interval_90_coverage")
                    ),
                )
            )
        lines.extend(["", "## Gates", ""])
        for name, gate in payload.get("gates", {}).items():
            lines.append(f"- `{name}`: `{gate.get('passed')}`")
        lines.extend(["", payload.get("note", "")])
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _json_safe_float(value: object) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric

    @staticmethod
    def _format_optional_float(value: object, digits: int = 4) -> str:
        numeric = ForecastPipeline._json_safe_float(value)
        return "n/a" if numeric is None else f"{numeric:.{digits}f}"

    def _read_optional_curated(self, name: str) -> pl.DataFrame:
        path = self.context.curated_dir / f"{name}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    def _verify_posterior_artifacts(self, out_dir: Path) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        posterior_schema = read_json(
            self.context.root / "schemas" / "artifact_contracts" / "posterior_draws.schema.json"
        )
        posterior = pl.read_parquet(out_dir / "posterior_draws.parquet")
        required_columns = set(posterior_schema.get("required_columns", []))
        missing_columns = sorted(required_columns - set(posterior.columns))
        checks.append(
            {
                "name": "posterior_draws_schema",
                "passed": not missing_columns and not posterior.is_empty(),
                "detail": {
                    "missing_columns": missing_columns,
                    "row_count": posterior.height,
                },
            }
        )
        diagnostics_path = out_dir / "posterior_diagnostics.json"
        diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {}
        diagnostics_required = {
            "engine",
            "draw_count",
            "race_option_count",
            "poll_count",
            "divergences",
            "model_config_hash",
            "source_manifest_hash",
        }
        missing_diagnostics = sorted(diagnostics_required - set(diagnostics))
        checks.append(
            {
                "name": "posterior_diagnostics_schema",
                "passed": not missing_diagnostics and bool(diagnostics),
                "detail": {"missing_keys": missing_diagnostics},
            }
        )
        fundamentals_prior = out_dir / "fundamentals_prior.parquet"
        checks.append(
            {
                "name": "fundamentals_prior_artifact",
                "passed": fundamentals_prior.exists()
                and not pl.read_parquet(fundamentals_prior).is_empty(),
                "detail": {"path": fundamentals_prior.name},
            }
        )
        return checks

    @staticmethod
    def _verify_timeout_failover_audit(out_dir: Path) -> dict[str, Any]:
        payload = read_json(out_dir / "timeout_failover_audit.json")
        audit = payload.get("audit", {})
        passed = (
            bool(payload.get("passed"))
            and payload.get("status") == "exercised"
            and bool(audit.get("fallback_used"))
            and audit.get("status") == "fallback_used"
        )
        return {
            "name": "timeout_failover_audit",
            "passed": passed,
            "detail": {
                "status": payload.get("status"),
                "audit_scope": payload.get("audit_scope"),
                "fallback_used": audit.get("fallback_used"),
                "publication_blocked": audit.get("publication_blocked"),
            },
        }

    @staticmethod
    def _verify_plot_manifest(out_dir: Path) -> dict[str, Any]:
        path = out_dir / "plot_manifest.json"
        if not path.exists():
            return {"name": "plot_manifest", "passed": False, "detail": "missing"}
        manifest = read_json(path)
        plot_paths = [
            out_dir / entry["path"]
            for entries in manifest.values()
            if isinstance(entries, list)
            for entry in entries
            if isinstance(entry, dict) and "path" in entry
        ]
        missing = [str(path.relative_to(out_dir)) for path in plot_paths if not path.exists()]
        empty = [
            str(path.relative_to(out_dir))
            for path in plot_paths
            if path.exists() and path.stat().st_size == 0
        ]
        return {
            "name": "plot_manifest",
            "passed": not missing and not empty and bool(plot_paths),
            "detail": {"missing": missing, "empty": empty, "plot_count": len(plot_paths)},
        }

    def _verify_posterior_history(self, out_dir: Path) -> dict[str, Any]:
        schema = read_json(
            self.context.root / "schemas" / "artifact_contracts" / "posterior_history.schema.json"
        )
        history = pl.read_parquet(out_dir / "posterior_history.parquet")
        missing_columns = sorted(set(schema.get("required_columns", [])) - set(history.columns))
        return {
            "name": "posterior_history_schema",
            "passed": not missing_columns and not history.is_empty(),
            "detail": {"missing_columns": missing_columns, "row_count": history.height},
        }

    def _verify_recalibration_map(self, out_dir: Path) -> dict[str, Any]:
        return self._verify_schema_artifact(out_dir, "recalibration_map.parquet")

    def _verify_schema_artifact(self, out_dir: Path, artifact_name: str) -> dict[str, Any]:
        schema_name = artifact_name.replace(".parquet", ".schema.json")
        schema = read_json(self.context.root / "schemas" / "artifact_contracts" / schema_name)
        frame = pl.read_parquet(out_dir / artifact_name)
        missing_columns = sorted(set(schema.get("required_columns", [])) - set(frame.columns))
        return {
            "name": f"{artifact_name.removesuffix('.parquet')}_schema",
            "passed": not missing_columns and not frame.is_empty(),
            "detail": {"missing_columns": missing_columns, "row_count": frame.height},
        }

    @staticmethod
    def _verify_reward_card(out_dir: Path) -> dict[str, Any]:
        path = out_dir / "reward_card.json"
        if not path.exists():
            return {"name": "reward_card", "passed": False, "detail": "missing"}
        reward_card = read_json(path)
        rewards = reward_card.get("rewards", {})
        failed = [
            key
            for key, value in rewards.items()
            if isinstance(value, dict) and value.get("passed") is False
        ]
        hard_gate_failures = [
            key
            for key in (
                "R13_posterior_quality",
                "R14_calibrated_publication",
                "R15_daily_update_quality",
            )
            if key in failed
        ]
        return {
            "name": "reward_card",
            "passed": bool(rewards) and not hard_gate_failures,
            "detail": {
                "failed": failed,
                "hard_gate_failures": hard_gate_failures,
                "reward_count": len(rewards),
            },
        }

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
            "state_accuracy": (
                comparison["state_accuracy"]
                if comparison.get("state_accuracy") is not None
                else comparison.get("winner_accuracy")
            ),
            "state_accuracy_n": (
                comparison["state_accuracy_n"]
                if comparison.get("state_accuracy_n") is not None
                else comparison.get("race_count")
            ),
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
        caucus_map = scenario.caucus_with
        seats_by_party: dict[str, int] = {}
        for row in race_outcomes:
            party = row.get("actual_winner_party")
            if not party:
                continue
            party = str(party).upper()
            caucus = caucus_map.get(party, party)
            seats_by_party[caucus] = seats_by_party.get(caucus, 0) + int(row.get("seats") or 1)
        for party, holdovers in scenario.holdover_caucus_seats.items():
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
        self,
        model_config: dict[str, Any],
        scenario: Scenario | None,
        admission_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        updated = json.loads(json.dumps(model_config))
        key = scenario.family if scenario else "default"
        path = (
            self.context.artifacts_dir / "backtests" / "latest" / f"component_admission_{key}.json"
        )
        if isinstance(admission_override, dict):
            return self._apply_component_admission_payload(
                updated,
                admission_override,
                {
                    "status": "inline",
                    "key": key,
                    "path": str(path),
                    "engine_using": str(
                        admission_override.get("engine_using", "inline_backtest_admission")
                    ),
                    "admission_status": str(admission_override.get("admission_status", "unknown")),
                },
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
            return self._apply_component_admission_payload(
                updated,
                admission,
                {
                    "status": "learned",
                    "key": key,
                    "path": str(path),
                    "recalibration_map": str(
                        self.context.artifacts_dir
                        / "backtests"
                        / "latest"
                        / f"recalibration_map_{key}.parquet"
                    ),
                    "engine_using": str(admission.get("engine_using", "learned_admission")),
                    "admission_status": str(admission.get("admission_status", "unknown")),
                    "weight_status": str(
                        dict(admission.get("ensemble_learning", {})).get("status", "unknown")
                    ),
                    "calibration_status": str(
                        dict(admission.get("probability_calibration", {})).get("status", "unknown")
                    ),
                },
            )
        return updated

    @staticmethod
    def _apply_component_admission_payload(
        model_config: dict[str, Any],
        admission: dict[str, Any],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(admission.get("trusted_components"), dict):
            model_config["trusted_components"] = admission["trusted_components"]
        if isinstance(admission.get("component_weights"), dict):
            model_config["component_weights"] = admission["component_weights"]
        if isinstance(admission.get("probability_calibration"), dict):
            model_config["probability_calibration"] = admission["probability_calibration"]
        if isinstance(admission.get("ensemble_learning"), dict):
            model_config["ensemble_learning_result"] = admission["ensemble_learning"]
        hyperpriors = admission.get("bayesian_hyperpriors")
        if isinstance(hyperpriors, dict) and isinstance(hyperpriors.get("selected"), dict):
            bayesian = dict(model_config.get("bayesian", {}))
            state_space = dict(bayesian.get("state_space", {}))
            prior_cfg = dict(bayesian.get("fundamentals_prior", {}))
            selected = hyperpriors["selected"]
            if selected.get("election_day_extra_sd") is not None:
                state_space["election_day_extra_sd"] = float(selected["election_day_extra_sd"])
            if selected.get("fundamentals_prior_strength") is not None:
                prior_cfg["prior_strength"] = float(selected["fundamentals_prior_strength"])
            bayesian["state_space"] = state_space
            bayesian["fundamentals_prior"] = prior_cfg
            bayesian["hyperpriors"] = hyperpriors
            model_config["bayesian"] = bayesian
        model_config["component_admission"] = admission
        model_config["component_admission_source"] = source
        return model_config

    def _load_residual_covariance(self, scenario: Scenario | None) -> pl.DataFrame | None:
        return self._read_latest_covariance(scenario.family if scenario else None)

    def _load_recalibration_map(self, scenario: Scenario | None) -> pl.DataFrame | None:
        return self._read_latest_recalibration_map(scenario.family if scenario else None)

    def _read_latest_covariance(self, key: str | None) -> pl.DataFrame | None:
        storage_key = key or "default"
        path = (
            self.context.artifacts_dir
            / "backtests"
            / "latest"
            / f"residual_covariance_{storage_key}.parquet"
        )
        return pl.read_parquet(path) if path.exists() else None

    def _read_latest_recalibration_map(self, key: str | None) -> pl.DataFrame | None:
        storage_key = key or "default"
        path = (
            self.context.artifacts_dir
            / "backtests"
            / "latest"
            / f"recalibration_map_{storage_key}.parquet"
        )
        return pl.read_parquet(path) if path.exists() else None

    def _read_latest_hyperpriors(self, key: str) -> dict[str, Any]:
        path = (
            self.context.artifacts_dir / "backtests" / "latest" / f"bayesian_hyperpriors_{key}.json"
        )
        return read_json(path) if path.exists() else {"status": "missing", "selected": {}}

    @staticmethod
    def _hyperprior_refresh_comparison(
        scenario: str,
        candidate: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_selected = dict(candidate.get("selected", {}))
        current_selected = dict(current.get("selected", {}))
        candidate_loss = candidate_selected.get("log_loss")
        current_loss = current_selected.get("log_loss")
        loss_delta = (
            round(float(candidate_loss) - float(current_loss), 12)
            if candidate_loss is not None and current_loss is not None
            else None
        )
        return {
            "scenario": scenario,
            "status": "candidate_ready" if candidate_selected else "candidate_unavailable",
            "candidate_status": candidate.get("status"),
            "current_status": current.get("status"),
            "candidate_log_loss": candidate_loss,
            "current_log_loss": current_loss,
            "candidate_minus_current_log_loss": loss_delta,
            "promotion_recommendation": "manual_review_required",
            "promotion_blocked": True,
            "reason": (
                "Scheduled refreshes write candidate hyperpriors only; production latest "
                "artifacts are unchanged until an explicit promotion review."
            ),
        }

    @staticmethod
    def _hyperprior_refresh_report(manifest: dict[str, Any]) -> str:
        lines = [
            "# Hyperprior Refresh Candidate Report",
            "",
            f"- Run id: `{manifest['run_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Promoted: `{manifest['promoted']}`",
            f"- Inference engine: `{manifest['inference_engine']}`",
            f"- Schedule days: `{manifest['schedule_days']}`",
            f"- Next due after: `{manifest['next_due_after']}`",
            "",
            "## Scenarios",
            "",
        ]
        for row in manifest.get("scenarios", []):
            comparison = dict(row.get("comparison", {}))
            lines.extend(
                [
                    f"### {row.get('scenario')}",
                    "",
                    f"- Candidate status: `{row.get('candidate_status')}`",
                    f"- Current status: `{row.get('current_status')}`",
                    f"- Row count: `{row.get('row_count')}`",
                    f"- Candidate log loss: `{comparison.get('candidate_log_loss')}`",
                    f"- Current log loss: `{comparison.get('current_log_loss')}`",
                    f"- Delta: `{comparison.get('candidate_minus_current_log_loss')}`",
                    f"- Promotion: `{comparison.get('promotion_recommendation')}`",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _attach_lineage(
        frame: pl.DataFrame, model_config: dict[str, Any], source_manifest: pl.DataFrame
    ) -> pl.DataFrame:
        model_hash = ForecastPipeline._config_hash(model_config)
        return frame.with_columns(
            pl.lit(model_hash).alias("model_config_hash"),
            pl.lit(ForecastPipeline._source_manifest_hash(source_manifest)).alias(
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
        stable_config = ForecastPipeline._stable_json_payload(model_config)
        return hashlib.sha256(
            json.dumps(stable_config, sort_keys=True, default=str).encode()
        ).hexdigest()

    @staticmethod
    def _source_manifest_hash(source_manifest: pl.DataFrame) -> str:
        source_hashes = ",".join(sorted(source_manifest["content_hash"].drop_nulls().to_list()))
        return hashlib.sha256(source_hashes.encode()).hexdigest()

    @staticmethod
    def _fundamentals_prior_metadata(frame: pl.DataFrame) -> dict[str, Any]:
        if frame.is_empty():
            return {"rows": 0, "methods": {}, "mean_sd_logit": None}
        methods = (
            {
                str(row["prior_method"]): int(row["count"])
                for row in frame.group_by("prior_method")
                .agg(pl.len().alias("count"))
                .iter_rows(named=True)
            }
            if "prior_method" in frame.columns
            else {}
        )
        return {
            "rows": frame.height,
            "methods": methods,
            "mean_sd_logit": float(frame["sd_logit"].mean())
            if "sd_logit" in frame.columns
            else None,
        }

    @staticmethod
    def _recalibration_map_metadata(frame: pl.DataFrame | None) -> dict[str, Any]:
        if frame is None or frame.is_empty():
            return {"rows": 0, "status": "missing"}
        row = frame.row(0, named=True)
        return {
            "rows": frame.height,
            "status": row.get("status"),
            "method": row.get("method"),
            "sample_size": row.get("sample_size"),
            "fit_cycles": row.get("fit_cycles"),
            "as_of_cuts": row.get("as_of_cuts"),
        }

    @staticmethod
    def _pollster_house_effect_frame(
        house_effects: dict[tuple[str, str | None], Any],
    ) -> pl.DataFrame:
        schema = {
            "pollster": pl.String,
            "option_id": pl.String,
            "effect": pl.Float64,
            "raw_effect": pl.Float64,
            "prior_effect": pl.Float64,
            "shrinkage": pl.Float64,
            "poll_count": pl.Int64,
        }
        rows = []
        for (_pollster, _option_id), estimate in house_effects.items():
            rows.append(
                {
                    "pollster": str(getattr(estimate, "pollster", _pollster)),
                    "option_id": getattr(estimate, "option_id", _option_id),
                    "effect": float(getattr(estimate, "effect", 0.0)),
                    "raw_effect": float(getattr(estimate, "raw_effect", 0.0)),
                    "prior_effect": float(getattr(estimate, "prior_effect", 0.0)),
                    "shrinkage": float(getattr(estimate, "shrinkage", 0.0)),
                    "poll_count": int(getattr(estimate, "poll_count", 0)),
                }
            )
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema).sort(["pollster", "option_id"])

    @staticmethod
    def _seat_posterior(
        draws: pl.DataFrame,
        race_catalog: pl.DataFrame,
        model_config: dict[str, Any],
        holdovers: dict[str, int] | None = None,
    ) -> pl.DataFrame:
        schema = {
            "draw_id": pl.Int64,
            "control_body": pl.String,
            "party": pl.String,
            "seat_count_modeled": pl.Float64,
            "holdover_seats": pl.Int64,
            "seat_count_total": pl.Float64,
            "majority_threshold": pl.Int64,
            "majority": pl.Boolean,
        }
        if draws.is_empty() or race_catalog.is_empty():
            return pl.DataFrame(schema=schema)
        catalog = race_catalog.select(["race_id", "control_body", "seats"]).filter(
            pl.col("control_body").is_not_null()
        )
        if catalog.is_empty():
            return pl.DataFrame(schema=schema)
        joined = draws.join(catalog, on="race_id", how="inner")
        if joined.is_empty():
            return pl.DataFrame(schema=schema)
        winners = joined.filter(pl.col("winner"))
        parties = sorted(str(value) for value in joined["party"].drop_nulls().unique().to_list())
        bodies = sorted(str(value) for value in joined["control_body"].unique().to_list())
        draw_ids = sorted(int(value) for value in joined["draw_id"].unique().to_list())
        counts = {
            (int(row["draw_id"]), str(row["control_body"]), str(row["party"])): float(
                row["seat_count_modeled"]
            )
            for row in winners.group_by(["draw_id", "control_body", "party"])
            .agg(pl.col("seats").sum().alias("seat_count_modeled"))
            .iter_rows(named=True)
        }
        thresholds = {
            str(key): int(value)
            for key, value in dict(model_config.get("control_thresholds", {})).items()
        }
        holdover_lookup = {str(key).upper(): int(value) for key, value in (holdovers or {}).items()}
        rows: list[dict[str, object]] = []
        for draw_id in draw_ids:
            for body in bodies:
                modeled_seats = int(
                    catalog.filter(pl.col("control_body") == body)
                    .select(pl.col("seats").sum())
                    .item()
                    or 0
                )
                threshold = thresholds.get(body, max(modeled_seats // 2 + 1, 1))
                for party in parties:
                    modeled = counts.get((draw_id, body, party), 0.0)
                    holdover = holdover_lookup.get(party.upper(), 0)
                    total = modeled + holdover
                    rows.append(
                        {
                            "draw_id": draw_id,
                            "control_body": body,
                            "party": party,
                            "seat_count_modeled": modeled,
                            "holdover_seats": holdover,
                            "seat_count_total": total,
                            "majority_threshold": threshold,
                            "majority": total >= threshold,
                        }
                    )
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

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
        artifact_names = [
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
        if (out_dir / "recalibration_map.parquet").exists():
            artifact_names.append("recalibration_map.parquet")
        artifact_names.extend(
            name
            for name in [
                "posterior_draws.parquet",
                "state_space_trajectory.parquet",
                "pollster_house_effects.parquet",
                "posterior_diagnostics.json",
                "fundamentals_prior.parquet",
                "seat_posterior.parquet",
                "senate_seat_posterior.parquet",
                "house_seat_posterior.parquet",
                "governor_seat_posterior.parquet",
                "senate_joint_posterior.parquet",
                "house_hierarchical_posterior.parquet",
                "cross_office_posterior.parquet",
            ]
            if (out_dir / name).exists()
        )
        stable_artifacts = {
            name: ForecastPipeline._stable_artifact_hash(out_dir / name) for name in artifact_names
        }
        combined_hash = hashlib.sha256(
            json.dumps(stable_artifacts, sort_keys=True).encode()
        ).hexdigest()
        previous_hash = str(previous.get("combined_hash")) if previous else None
        payload: dict[str, Any] = {
            "status": "fingerprint_generated",
            "excluded_fields": sorted(_FINGERPRINT_EXCLUDED_FIELDS),
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
            ignored = [column for column in _FINGERPRINT_EXCLUDED_FIELDS if column in frame.columns]
            if ignored:
                frame = frame.drop(ignored)
            if frame.columns:
                frame = frame.sort(frame.columns)
            rows = frame.to_dicts()
            payload = json.dumps(rows, sort_keys=True, default=str)
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload_obj = json.load(handle)
            payload_obj = ForecastPipeline._stable_json_payload(payload_obj)
            payload = json.dumps(payload_obj, sort_keys=True, default=str)
        else:
            payload = path.read_text(encoding="utf-8")
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _stable_json_payload(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ForecastPipeline._stable_json_payload(item)
                for key, item in value.items()
                if key not in _FINGERPRINT_EXCLUDED_FIELDS
            }
        if isinstance(value, list):
            return [ForecastPipeline._stable_json_payload(item) for item in value]
        return value

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
        by_offset = [
            {
                **row,
                "mean_absolute_change": ForecastPipeline._stable_metric_float(
                    row["mean_absolute_change"]
                ),
                "max_absolute_change": ForecastPipeline._stable_metric_float(
                    row["max_absolute_change"]
                ),
            }
            for row in by_offset
        ]
        return {
            "scenario_family": scenario_family,
            "available": True,
            "row_count": rolling_predictions.height,
            "mean_absolute_probability_change": ForecastPipeline._stable_metric_float(
                changes["absolute_probability_change"].mean()
            ),
            "max_probability_change": ForecastPipeline._stable_metric_float(
                changes["absolute_probability_change"].max()
            ),
            "by_as_of_offset_days": by_offset,
        }

    @staticmethod
    def _stable_metric_float(value: float | int | None) -> float | None:
        return None if value is None else round(float(value), 12)
