from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from election_outcomes.config import ProjectContext, ScenarioRegistry
from election_outcomes.features import FeatureBuilder
from election_outcomes.inference.cross_office import summarize_cross_office
from election_outcomes.inference.failover import (
    FailoverPolicy,
    execute_with_failover,
    exercise_timeout_failover,
)
from election_outcomes.inference.fundamentals_prior import (
    DiagonalNormalPrior,
    build_fundamentals_prior,
    to_numpyro_prior,
)
from election_outcomes.inference.house_hierarchical import summarize_house_hierarchical
from election_outcomes.inference.recalibration import RecalibrationMap, fit_recalibration
from election_outcomes.inference.seed import derive_seed, jax_prng_key
from election_outcomes.inference.senate_joint import summarize_senate_joint
from election_outcomes.inference.state_space import build_state_space_data
from election_outcomes.ingest import SyncRunner
from election_outcomes.ingest.sources import SourceDefinition, SourceRegistry
from election_outcomes.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from election_outcomes.models.common import logit
from election_outcomes.models.polling import resolve_inference_engine
from election_outcomes.models.polling_bayes import BayesianPollingModel
from election_outcomes.normalize import CuratedDataBuilder
from election_outcomes.normalize.builder import CuratedDataBuilder as _CuratedDataBuilderClass
from election_outcomes.performance import simulate_binary_draw_arrays
from election_outcomes.pipeline import ForecastPipeline
from election_outcomes.reports.plots import PlotGenerator
from election_outcomes.scoring import BacktestRunner, CycleEvaluationReport, score_predictions
from election_outcomes.scoring.results import ResultComparator
from election_outcomes.scoring.rewards import RewardEvaluator
from election_outcomes.storage.io import read_json
from election_outcomes.verification import Phase8VerificationRunner

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def build_bundle(tmp_path: Path):
    ctx = context(tmp_path)
    first = SyncRunner(ctx).run()
    CuratedDataBuilder(ctx).run()
    bundle = FeatureBuilder(ctx).run()
    return ctx, first, bundle


def test_sync_is_incremental_and_records_manifest(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    first = SyncRunner(ctx).run()
    second = SyncRunner(ctx).run()

    assert first.fetched_sources == 13
    assert first.failed_sources == 0
    assert second.fetched_sources == 0
    assert second.skipped_sources == 13
    assert (ctx.raw_dir / "source_manifest.parquet").exists()
    assert first.manifest.filter(pl.col("content_hash") == "").is_empty()


def test_source_overlay_and_failed_sync_preserve_previous_state(
    tmp_path: Path, monkeypatch
) -> None:
    from election_outcomes.ingest import sync as sync_module

    live_ctx = ProjectContext.create(
        root=ROOT,
        sources_config="sources_live.yaml",
        data_dir=tmp_path / "live_data",
        artifacts_dir=tmp_path / "live_artifacts",
    )
    live_registry = SourceRegistry.from_context(live_ctx)
    assert len(live_registry.sources) == 21
    assert live_registry.sources[-1].id == "wikipedia_house_2026_race_presence"

    ctx = context(tmp_path)
    ok_source = SourceDefinition(
        id="temporary_source",
        table="polls",
        type="fixture",
        path=ROOT / "fixtures" / "polls.csv",
        parser_version="fixture-v1",
        license="state preservation fixture",
        url="file://fixtures/polls.csv",
    )
    SyncRunner(ctx, registry=SourceRegistry([ok_source])).run()
    previous = read_json(ctx.state_dir / "sync_state.json")
    assert "temporary_source" in previous

    failing_source = SourceDefinition(
        id="temporary_source",
        table="polls",
        type="http_csv",
        path=None,
        parser_version="fixture-v1",
        license="state preservation fixture",
        url="http://127.0.0.1:1/missing.csv",
    )
    monkeypatch.setattr(sync_module, "HTTP_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    result = SyncRunner(ctx, registry=SourceRegistry([failing_source])).run()
    state = read_json(ctx.state_dir / "sync_state.json")

    assert result.failed_sources == 1
    assert state["temporary_source"] == previous["temporary_source"]


def test_http_sync_reuses_cached_snapshot_when_same_source_refresh_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from election_outcomes.ingest import sync as sync_module

    source_path = tmp_path / "remote.csv"
    source_path.write_text("poll_id,race_id,option_id,pct\n1,R,O,50\n", encoding="utf-8")
    ctx = context(tmp_path)
    source = SourceDefinition(
        id="cached_http_source",
        table="polls",
        type="http_csv",
        path=Path("remote.csv"),
        parser_version="fixture-v1",
        license="cache fallback fixture",
        url=source_path.resolve().as_uri(),
    )
    first = SyncRunner(ctx, registry=SourceRegistry([source])).run()
    previous = read_json(ctx.state_dir / "sync_state.json")
    assert first.failed_sources == 0

    def fail_refresh(url: str) -> bytes:
        raise RuntimeError(f"blocked test refresh: {url}")

    monkeypatch.setattr(
        sync_module.SyncRunner,
        "_http_get_with_retry",
        staticmethod(fail_refresh),
    )
    second = SyncRunner(ctx, registry=SourceRegistry([source])).run()
    state = read_json(ctx.state_dir / "sync_state.json")
    row = second.manifest.row(0, named=True)

    assert second.failed_sources == 0
    assert second.skipped_sources == 1
    assert row["status"] == "stale_reused"
    assert row["content_hash"] == previous["cached_http_source"]
    assert state["cached_http_source"] == previous["cached_http_source"]


def test_feature_builder_assigns_tiers_and_filters_blank_rows(tmp_path: Path) -> None:
    _ctx, _sync, bundle = build_bundle(tmp_path)
    tiers = dict(zip(bundle.race_catalog["race_id"], bundle.race_catalog["tier"], strict=True))

    assert tiers["US-SEN-GA-2026"] == "A"
    assert tiers["US-HOUSE-CA45-2026"] == "B"
    assert tiers["MAYOR-SPRINGFIELD-2026"] == "C"
    assert bundle.race_catalog["race_id"].null_count() == 0


def test_component_models_and_ensemble_respect_admission(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")
    polling_model = PollingModel(model_config, as_of="2026-05-08")

    estimates = [
        polling_model.run(active),
        FundamentalsModel(model_config).fit(bundle).run(active),
        MarketModel(model_config).run(active),
        PublicSignalModel(trusted=False).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)

    assert not ensemble.is_empty()
    assert "MAYOR-SPRINGFIELD-2026" not in ensemble["race_id"].to_list()
    public = estimates[-1]
    assert public.filter(pl.col("race_id") == "US-SEN-GA-2026")["admitted"].sum() == 0
    trajectory = polling_model.trajectory(active)
    assert "initial_vote_share_prior" in trajectory.columns
    prior_rows = trajectory.join(
        active.options.select(["race_id", "option_id", "previous_vote_share"]),
        on=["race_id", "option_id"],
        how="inner",
    ).filter(pl.col("previous_vote_share").is_not_null())
    assert not prior_rows.is_empty()
    assert (
        prior_rows.select(
            (pl.col("initial_vote_share_prior") - pl.col("previous_vote_share")).abs().max()
        ).item()
        < 1e-12
    )
    copied = replace(active, polls=active.polls.clone(), options=active.options.clone())
    assert polling_model._bundle_fingerprint(copied) == polling_model._bundle_fingerprint(active)
    changed = replace(active, polls=active.polls.with_columns((pl.col("pct") + 0.01).alias("pct")))
    assert polling_model._bundle_fingerprint(changed) != polling_model._bundle_fingerprint(active)


def test_polling_default_resolves_from_model_config() -> None:
    config = {
        "bayesian": {"enabled": True},
    }

    assert resolve_inference_engine(config) == "bayes"
    assert resolve_inference_engine(config, "kalman") == "kalman"
    assert resolve_inference_engine({"bayesian": {"enabled": False}}) == "kalman"


def test_bayesian_polling_exports_deterministic_posterior_draws(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = json.loads(json.dumps(ctx.read_yaml("model.yaml")))
    model_config["_bayesian_backend"] = "analytic"

    first_model = PollingModel(model_config, as_of="2026-05-08", inference_engine="bayes")
    first_estimates = first_model.run(active)
    first_draws = first_model.posterior_draws(active)
    first_diagnostics = first_model.diagnostics(active)

    second_model = PollingModel(model_config, as_of="2026-05-08", inference_engine="bayes")
    second_model.run(active)
    second_draws = second_model.posterior_draws(active)

    assert not first_estimates.is_empty()
    assert not first_draws.is_empty()
    assert {
        "draw_id",
        "chain_id",
        "race_id",
        "option_id",
        "latent_logit",
        "latent_share",
        "systematic_error",
        "pollster_effect",
    }.issubset(first_draws.columns)
    assert first_diagnostics["engine"] == "bayes-analytic-logit-normal"
    assert first_diagnostics["parameterization"] == "noncentered"
    assert first_diagnostics["divergences"] == 0
    assert first_diagnostics["forecast_horizon_inflation"]["method"] == (
        "random_walk_logit_inflation"
    )
    posterior_sums = (
        first_draws.join(
            active.options.group_by("race_id").agg(
                pl.col("option_id").n_unique().alias("option_count")
            ),
            on="race_id",
            how="inner",
        )
        .filter(pl.col("option_count") > 1)
        .group_by(["race_id", "draw_id"])
        .agg(pl.col("latent_share").sum().alias("share_sum"))
    )
    assert posterior_sums["share_sum"].to_list() == pytest.approx([1.0] * posterior_sums.height)
    assert first_draws.to_dicts() == second_draws.to_dicts()


def test_bayesian_polling_adapts_numpyro_nuts_result(tmp_path: Path, monkeypatch) -> None:
    from election_outcomes.inference import nuts as nuts_module
    from election_outcomes.inference.nuts import InferenceResult

    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = json.loads(json.dumps(ctx.read_yaml("model.yaml")))
    model_config["_bayesian_backend"] = "nuts"
    model_config["simulation_count"] = 20
    model_config["bayesian"]["posterior_draw_count"] = 20
    model_config["bayesian"]["nuts"]["num_warmup"] = 10
    model_config["bayesian"]["nuts"]["num_samples"] = 10
    model_config["bayesian"]["nuts"]["wall_clock_timeout_seconds"] = 30
    fundamentals_prior = build_fundamentals_prior(
        FundamentalsModel(model_config).fit(bundle), active, model_config
    )
    model_config["_fundamentals_prior_rows"] = fundamentals_prior.frame.to_dicts()

    def fake_fit_nuts(data, hyperpriors=None, config=None, seed=0):
        del hyperpriors, seed
        sample_count = int(config.num_samples * config.num_chains)
        offsets = np.linspace(-0.05, 0.05, sample_count, dtype=np.float64).reshape(-1, 1)
        state_logit = np.asarray(data.prior_logit, dtype=np.float64).reshape(1, -1) + offsets
        diagnostics = {
            "engine": "numpyro-nuts",
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "num_chains": config.num_chains,
            "chain_method": config.chain_method,
            "target_accept_prob": config.target_accept_prob,
            "parameterization": config.parameterization,
            "elapsed_seconds": 0.01,
            "divergences": 0,
            "r_hat_max": 1.0,
            "ess_min": float(sample_count),
            "r_hat_available": True,
            "ess_available": True,
            "hierarchy": {
                "office_count": len(data.office_ids),
                "geography_count": len(data.geography_ids),
                "race_count": len(data.race_ids),
                "office_ids": list(data.office_ids),
                "geography_ids": list(data.geography_ids),
            },
            "failover_audit": {
                "status": "completed",
                "primary_engine": "numpyro-nuts",
                "fallback_used": None,
                "reason": None,
                "elapsed_seconds": 0.01,
                "timeout_seconds": config.wall_clock_timeout_seconds,
                "fallback_order": [],
                "publication_blocked": False,
            },
        }
        return InferenceResult(
            samples={
                "state_logit": state_logit,
                "pollster_effect": np.zeros((sample_count, max(len(data.pollster_ids), 1))),
            },
            diagnostics=diagnostics,
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(nuts_module, "fit_nuts", fake_fit_nuts)
    model = PollingModel(model_config, as_of="2026-05-08", inference_engine="bayes")
    estimates = model.run(active)
    diagnostics = model.diagnostics(active)
    draws = model.posterior_draws(active)
    senate = summarize_senate_joint(active, draws, posterior_diagnostics=diagnostics)
    house = summarize_house_hierarchical(active, draws, model_config, diagnostics)
    cross_office = summarize_cross_office(active, draws, model_config, diagnostics)

    assert diagnostics["engine"] == "numpyro-nuts"
    assert diagnostics["fallback_used"] is None
    assert diagnostics["fundamentals_prior_used"] is True
    assert diagnostics["fundamentals_prior_rows"] == fundamentals_prior.frame.height
    assert diagnostics["hierarchy"]["office_count"] >= 1
    assert "house" in diagnostics["hierarchy"]["office_ids"]
    assert diagnostics["hierarchical_effects"]["office_count"] >= 1
    assert diagnostics["hierarchical_effects"]["geography_count"] >= 1
    assert diagnostics["num_chains"] == 2
    assert diagnostics["nuts_sample_count"] == 20
    assert diagnostics["r_hat_available"] is True
    assert diagnostics["ess_available"] is True
    assert diagnostics["draw_count"] == 100
    assert diagnostics["posterior_sample_resampling"] == "with_replacement"
    assert diagnostics["forecast_horizon_inflation"]["mean_horizon_sd_logit"] > 0
    probability_sums = estimates.group_by("race_id").agg(
        pl.len().alias("option_count"),
        pl.col("marginal_win_probability").sum().alias("probability_sum"),
    )
    assert probability_sums.filter(pl.col("option_count") > 1)[
        "probability_sum"
    ].to_list() == pytest.approx([1.0] * probability_sums.filter(pl.col("option_count") > 1).height)
    assert estimates.height == diagnostics["race_option_count"]
    assert diagnostics["prior_only_race_option_count"] >= 0
    assert diagnostics["race_option_count"] >= diagnostics["polling_observed_race_option_count"]
    assert draws.height == diagnostics["draw_count"] * diagnostics["race_option_count"]
    assert senate.diagnostics["state_space_nuts_fitted"] is True
    assert senate.diagnostics["engine"] == "numpyro-nuts-senate-joint-decomposition"
    assert house.diagnostics["state_space_nuts_fitted"] is True
    assert house.diagnostics["engine"] == "numpyro-nuts-house-hierarchical-decomposition"
    assert cross_office.diagnostics["state_space_nuts_fitted"] is True
    assert cross_office.diagnostics["engine"] == "numpyro-nuts-cross-office-decomposition"


def test_bayesian_nuts_backend_fallback_is_visible(tmp_path: Path, monkeypatch) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = json.loads(json.dumps(ctx.read_yaml("model.yaml")))
    model_config["_bayesian_backend"] = "nuts"

    def fail_nuts(self, bundle, as_of):
        raise RuntimeError("forced nuts failure")

    monkeypatch.setattr(BayesianPollingModel, "_fit_nuts_backend", fail_nuts)
    model = PollingModel(model_config, as_of="2026-05-08", inference_engine="bayes")
    estimates = model.run(active)
    diagnostics = model.diagnostics(active)

    assert not estimates.is_empty()
    assert diagnostics["engine"] == "bayes-analytic-logit-normal"
    assert diagnostics["fallback_used"] == "previous_posterior_reuse"
    assert diagnostics["failover_audit"]["primary_engine"] == "numpyro-nuts"
    assert diagnostics["failover_audit"]["reason"] == "forced nuts failure"


def test_fundamentals_predictive_distribution_builds_prior(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    model_config = ctx.read_yaml("model.yaml")
    model = FundamentalsModel(model_config).fit(bundle)
    predictive = model.predictive_distribution(bundle)
    prior = build_fundamentals_prior(model, bundle, model_config)
    distribution = to_numpyro_prior(
        prior,
        {race_id: index for index, race_id in enumerate(sorted(set(prior.race_ids)))},
    )

    assert not predictive.is_empty()
    assert {
        "race_id",
        "option_id",
        "mean_logit",
        "sd_logit",
        "prior_method",
    }.issubset(predictive.columns)
    assert predictive["sd_logit"].min() > 0
    assert prior.frame.height == predictive.height
    assert prior.prior_strength == model_config["bayesian"]["fundamentals_prior"]["prior_strength"]
    assert isinstance(distribution, DiagonalNormalPrior) or hasattr(distribution, "sample")


def test_recalibration_map_round_trip_and_apply(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "learned_ensemble_probability": [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90] * 2,
            "actual_winner": [False, False, False, True, True, True, True, True] * 2,
            "cycle": [2020] * 8 + [2024] * 8,
            "as_of_offset_days": [30, 7] * 8,
        }
    )
    recalibration = fit_recalibration(
        frame,
        config={"minimum_rows_for_trust": 8, "ensemble_learning": {"calibration_ridge": 0.001}},
    )
    path = tmp_path / "recalibration_map.parquet"
    recalibration.to_parquet(path)
    loaded = RecalibrationMap.from_parquet(path)
    adjusted = loaded.apply([0.25, 0.75])

    assert loaded.sample_size == frame.height
    assert loaded.fit_cycles == (2020, 2024)
    assert loaded.as_of_cuts == (7, 30)
    assert loaded.status == "fitted"
    assert adjusted.shape == (2,)
    assert adjusted[0] < adjusted[1]


def test_state_space_data_and_seed_helpers(tmp_path: Path) -> None:
    _ctx, _sync, bundle = build_bundle(tmp_path)
    data = build_state_space_data(
        bundle,
        as_of="2024-10-01",
        office_type="president",
        cycle=2024,
    )
    midterm_data = build_state_space_data(
        bundle,
        as_of="2026-05-08",
        office_type=None,
        cycle=2026,
    )
    first_seed = derive_seed("abc", "phase0")
    second_seed = derive_seed("abc", "phase0")

    assert data.poll_logit_y.size > 0
    assert data.poll_s.max() < data.dims[0]
    assert data.poll_j.max() < data.dims[2]
    assert data.prior_logit.shape == (data.dims[0],)
    assert data.option_office.shape == (data.dims[0],)
    assert data.option_geography.shape == (data.dims[0],)
    assert data.option_race.shape == (data.dims[0],)
    assert data.metadata["office_count"] == len(data.office_ids)
    assert data.metadata["geography_count"] == len(data.geography_ids)
    assert data.metadata["race_count"] == len(data.race_ids)
    assert data.office_ids == ["president"]
    assert data.metadata["poll_count"] == data.poll_logit_y.size
    assert data.metadata["poll_half_life_days"] == pytest.approx(21.0)
    assert data.metadata["pollster_house_effect_adjustment_mean_abs"] == pytest.approx(0.0)
    assert 0 < data.metadata["observation_weight_min"] <= data.metadata["observation_weight_max"]
    override_key = data.race_option_keys[0]
    prior_override = build_state_space_data(
        bundle,
        as_of="2024-10-01",
        office_type="president",
        cycle=2024,
        prior_logit_by_key={override_key: logit(0.62)},
    )
    assert prior_override.prior_logit[0] == pytest.approx(logit(0.62))
    adjusted_data = build_state_space_data(
        bundle,
        as_of="2024-10-01",
        office_type="president",
        cycle=2024,
        pollster_house_effects={(data.pollster_ids[0], None): 0.02},
    )
    drift_data = build_state_space_data(
        bundle,
        as_of="2024-10-01",
        office_type="president",
        cycle=2024,
        process_drift_sd_per_sqrt_day=0.01,
    )
    assert adjusted_data.metadata["pollster_house_effect_adjustment_mean_abs"] > 0
    assert not np.allclose(adjusted_data.poll_logit_y, data.poll_logit_y)
    assert drift_data.metadata["process_drift_sd_per_sqrt_day"] == pytest.approx(0.01)
    assert drift_data.metadata["temporal_process_variance"] == "poll_age_logit_variance"
    assert float(drift_data.poll_kappa.mean()) > float(data.poll_kappa.mean())
    assert {"governor", "house", "president", "senate"}.issubset(set(midterm_data.office_ids))
    assert "CA" in set(midterm_data.geography_ids)
    assert first_seed == second_seed
    try:
        key = jax_prng_key(first_seed)
    except RuntimeError as exc:
        assert "uv sync" in str(exc)
    else:
        assert key is not None


def test_office_methodology_summaries_use_shared_draw_stream(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = json.loads(json.dumps(ctx.read_yaml("model.yaml")))
    model_config["_bayesian_backend"] = "analytic"
    fundamentals_prior = build_fundamentals_prior(
        FundamentalsModel(model_config).fit(bundle), active, model_config
    )
    model_config = {**model_config, "_fundamentals_prior_rows": fundamentals_prior.frame.to_dicts()}
    polling_model = PollingModel(model_config, as_of="2026-05-08", inference_engine="bayes")
    polling_model.run(active)
    posterior = polling_model.posterior_draws(active)

    senate = summarize_senate_joint(active, posterior)
    house = summarize_house_hierarchical(active, posterior, model_config)
    cross_office = summarize_cross_office(active, posterior, model_config)

    assert senate.diagnostics["status"] == "fitted"
    assert senate.posterior["senate_class"].drop_nulls().n_unique() >= 1
    assert senate.posterior["class_effect_logit"].is_null().sum() == 0
    assert house.diagnostics["status"] == "fitted"
    assert house.diagnostics["dense_covariance_used"] is False
    assert "block_diagonal_state_era" in house.posterior["covariance_method"][0]
    assert cross_office.diagnostics["status"] == "fitted"
    assert set(cross_office.posterior["office_type"].unique().to_list()) >= {
        "governor",
        "house",
        "president",
        "senate",
    }
    assert cross_office.posterior["shared_draw_stream"].all()


def test_simulation_outputs_forecasts_control_and_ecosystem(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")
    model_config["probability_calibration"] = {
        "status": "fitted",
        "method": "platt_logistic_ridge",
        "intercept": 0.0,
        "slope": 0.0,
    }
    estimates = [
        PollingModel(model_config, as_of="2026-05-08").run(active),
        FundamentalsModel(model_config).fit(bundle).run(active),
        MarketModel(model_config).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)
    outputs = SimulationEngine(model_config).run(active, ensemble)

    simulated_options = ensemble.filter(pl.col("admitted")).height
    assert outputs.draws.height == int(model_config["simulation_count"]) * simulated_options
    assert outputs.control_forecasts.height > 0
    assert {"control_threshold", "pivotal_rates"}.issubset(outputs.control_forecasts.columns)
    assert outputs.ecosystem_forecasts.height >= 4
    assert outputs.ecosystem_forecasts["demographic_model_status"].unique().to_list() == [
        "placeholder_not_estimated"
    ]
    assert outputs.ecosystem_forecasts["close_margin_risk_status"].unique().to_list() == [
        "withheld_experimental"
    ]
    assert (
        outputs.ecosystem_forecasts["recount_probability"].null_count()
        == outputs.ecosystem_forecasts.height
    )
    tier_c = outputs.race_forecasts.filter(pl.col("race_id") == "MAYOR-SPRINGFIELD-2026")
    assert tier_c["winner_probability"].null_count() == tier_c.height
    trusted = outputs.race_forecasts.filter(pl.col("winner_probability").is_not_null())
    assert {"raw_winner_probability", "probability_calibration_status"}.issubset(
        outputs.race_forecasts.columns
    )
    assert trusted["probability_calibration_status"].unique().to_list() == ["fitted"]
    assert trusted["winner_probability"].round(8).unique().to_list() == [0.5]
    assert {"top_drivers", "component_contributions", "uncertainty_explanation"}.issubset(
        outputs.race_forecasts.columns
    )


def test_simulation_returns_typed_empty_ecosystem_forecasts(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    outputs = SimulationEngine(ctx.read_yaml("model.yaml")).run(active, pl.DataFrame())

    assert outputs.draws.is_empty()
    assert outputs.control_forecasts.is_empty()
    assert outputs.ecosystem_forecasts.is_empty()
    assert {
        "race_id",
        "turnout_mean",
        "demographic_model_status",
        "close_margin_risk_status",
        "ballot_measure_supported",
    }.issubset(outputs.ecosystem_forecasts.columns)

    manifest = PlotGenerator().render_all(
        tmp_path / "empty-draw-report",
        active.race_catalog,
        outputs.race_forecasts,
        outputs.draws,
        outputs.control_forecasts,
        outputs.ecosystem_forecasts,
        pl.DataFrame(),
        {"metrics": {}},
    )
    assert manifest["projection"]


def test_forecast_run_writes_required_artifacts_and_rewards(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    out_dir = ForecastPipeline(ctx).run_forecast(
        as_of="2026-05-08", run_id="unit", inference_engine="kalman"
    )
    required = {
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
        "silver_benchmark.html",
        "silver_benchmark.json",
        "plot_manifest.json",
        "poll_trajectory.parquet",
        "stability_metrics.json",
        "performance.json",
        "recalibration_map.parquet",
        "reproducibility_fingerprint.json",
        "plots",
        "race_detail_index.json",
        "races",
    }

    assert {path.name for path in out_dir.iterdir()} == required
    plot_manifest = json.loads((out_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    plot_paths = [
        out_dir / entry["path"] for entries in plot_manifest.values() for entry in entries
    ]
    assert len(plot_manifest["calibration"]) >= 3
    assert len(plot_manifest["projection"]) >= 4
    assert len(plot_manifest["trajectory"]) >= 1
    assert len(plot_manifest["stability"]) >= 1
    assert len(plot_manifest["model_quality"]) >= 1
    assert len(plot_manifest["benchmark"]) >= 1
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths)
    assert (out_dir / "plots" / "kalman_posterior_uncertainty.png").stat().st_size > 0
    benchmark = json.loads((out_dir / "silver_benchmark.json").read_text(encoding="utf-8"))
    assert "Silver/FiveThirtyEight" in benchmark["benchmark_name"]
    assert benchmark["tier_scale"] == {
        "absent": 0.0,
        "scaffold": 0.33,
        "functional": 0.66,
        "production": 1.0,
    }
    assert 0.0 <= benchmark["summary_score"] <= 0.85
    assert {row["tier"] for row in benchmark["rows"]}.issubset(
        {"absent", "scaffold", "functional", "production"}
    )
    trajectory_row = next(
        row for row in benchmark["rows"] if row["dimension"] == "Polling trajectory/Kalman support"
    )
    assert trajectory_row["tier"] == "functional"
    diagnostics = (out_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "Scenario Scope" in diagnostics
    assert "Distribution And Probability View" in diagnostics
    assert "Model Quality" in diagnostics
    assert "MCMC-style split posterior simulation draws" in diagnostics
    assert "Model Drivers" in diagnostics
    assert "Silver/FiveThirtyEight Benchmark" in diagnostics
    forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
    assert {"model_config_hash", "source_manifest_hash"}.issubset(forecasts.columns)
    reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    rewards = reward_card["rewards"]
    assert rewards["R0_build"]["passed"] is None
    assert rewards["R1_reproducibility"]["passed"] is False
    assert rewards["R2_provenance"]["passed"] is True
    assert rewards["R3_sync_integrity"]["passed"] is True
    assert isinstance(rewards["R5_baseline_competition"]["passed"], bool)
    assert isinstance(rewards["R6_component_admission"]["passed"], bool)
    assert isinstance(rewards["R8_uncertainty_quality"]["passed"], bool)
    assert rewards["R12_performance_contract"]["passed"] is True
    assert "R14_calibrated_publication" in rewards
    performance = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    assert performance["engine"] in {"numba", "python"}
    assert performance["simulation_count"] == 1000
    model_card = (out_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Admission source" in model_card
    assert "Pollster House Effects" in model_card
    assert "standardized_ridge_fit" in model_card or "handpicked_default" in model_card


def test_reward_component_admission_checks_only_trusted_components() -> None:
    ablations = {
        "ensemble": {"beats_or_matches_baseline": True},
        "polling": {"beats_or_matches_baseline": True},
        "fundamentals": {"beats_or_matches_baseline": False},
        "markets": {"beats_or_matches_baseline": False},
        "public_signals": {"beats_or_matches_baseline": True},
    }
    selective = RewardEvaluator(
        {
            "trusted_components": {
                "polling": True,
                "fundamentals": False,
                "markets": False,
                "public_signals": False,
            }
        }
    )._component_admission_metric(ablations)
    strict = RewardEvaluator(
        {
            "trusted_components": {
                "polling": True,
                "fundamentals": True,
                "markets": False,
                "public_signals": False,
            }
        }
    )._component_admission_metric(ablations)

    assert selective["passed"] is True
    assert selective["failed_trusted_components"] == []
    assert strict["passed"] is False
    assert strict["failed_trusted_components"] == ["fundamentals"]


def test_model_config_hash_ignores_volatile_admission_timestamp() -> None:
    first = {
        "bayesian": {"enabled": True},
        "component_admission": {
            "generated_at": "2026-05-10T00:00:00+00:00",
            "trusted_components": {"polling": True},
        },
    }
    second = {
        "bayesian": {"enabled": True},
        "component_admission": {
            "generated_at": "2026-05-10T01:00:00+00:00",
            "trusted_components": {"polling": True},
        },
    }

    assert ForecastPipeline._config_hash(first) == ForecastPipeline._config_hash(second)


def test_publication_probability_calibration_caps_sharpening_slope() -> None:
    bounded = ForecastPipeline._publication_probability_calibration(
        {"ensemble_learning": {"calibration_max_slope": 1.0}},
        {
            "status": "fitted",
            "method": "platt_logistic_ridge",
            "intercept": 0.0,
            "slope": 2.0,
            "max_slope": 2.0,
        },
    )

    assert bounded["slope"] == pytest.approx(1.0)
    assert bounded["source_slope"] == pytest.approx(2.0)
    assert bounded["publication_slope_capped"] is True


def test_component_admission_falls_back_when_trusted_component_unavailable() -> None:
    model_config = {
        "trusted_components": {"markets": True, "polling": False, "fundamentals": False},
        "component_weights": {"markets": 1.0, "polling": 0.0, "fundamentals": 0.0},
    }
    polling = pl.DataFrame(
        [{"component": "polling", "admitted": True, "race_id": "R", "option_id": "D"}]
    )

    updated = ForecastPipeline._ensure_available_component_admission(model_config, [polling])

    assert updated["trusted_components"]["polling"] is True
    assert updated["component_weights"]["polling"] == pytest.approx(1.0)
    assert updated["component_admission_runtime_fallback"]["fallback_component"] == "polling"


def test_forecast_run_with_bayes_writes_posterior_artifacts(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    out_dir = ForecastPipeline(ctx).run_forecast(
        as_of="2026-05-08",
        run_id="bayes-unit",
        inference_engine="bayes",
        bayesian_backend="analytic",
    )

    posterior_path = out_dir / "posterior_draws.parquet"
    diagnostics_path = out_dir / "posterior_diagnostics.json"
    fundamentals_prior_path = out_dir / "fundamentals_prior.parquet"
    seat_posterior_path = out_dir / "seat_posterior.parquet"
    state_space_trajectory_path = out_dir / "state_space_trajectory.parquet"
    pollster_house_effects_path = out_dir / "pollster_house_effects.parquet"
    senate_joint_path = out_dir / "senate_joint_posterior.parquet"
    house_hierarchical_path = out_dir / "house_hierarchical_posterior.parquet"
    cross_office_path = out_dir / "cross_office_posterior.parquet"
    posterior = pl.read_parquet(posterior_path)
    fundamentals_prior = pl.read_parquet(fundamentals_prior_path)
    seat_posterior = pl.read_parquet(seat_posterior_path)
    state_space_trajectory = pl.read_parquet(state_space_trajectory_path)
    pollster_house_effects = pl.read_parquet(pollster_house_effects_path)
    senate_joint = pl.read_parquet(senate_joint_path)
    house_hierarchical = pl.read_parquet(house_hierarchical_path)
    cross_office = pl.read_parquet(cross_office_path)
    forecast_draws = pl.read_parquet(out_dir / "forecast_draws.parquet")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    html = (out_dir / "diagnostics.html").read_text(encoding="utf-8")
    plot_manifest = json.loads((out_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    performance = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    verification = ForecastPipeline(ctx).verify_run("bayes-unit")
    update = ForecastPipeline(ctx).run_daily_update(anchor_run_id="bayes-unit", as_of="2026-05-09")
    verification_after_update = ForecastPipeline(ctx).verify_run("bayes-unit")
    model_card = (out_dir / "model_card.md").read_text(encoding="utf-8")
    fingerprint = json.loads(
        (out_dir / "reproducibility_fingerprint.json").read_text(encoding="utf-8")
    )
    posterior_schema = json.loads(
        (ROOT / "schemas" / "artifact_contracts" / "posterior_draws.schema.json").read_text(
            encoding="utf-8"
        )
    )
    senate_schema = json.loads(
        (ROOT / "schemas" / "artifact_contracts" / "senate_joint_posterior.schema.json").read_text(
            encoding="utf-8"
        )
    )
    house_schema = json.loads(
        (
            ROOT / "schemas" / "artifact_contracts" / "house_hierarchical_posterior.schema.json"
        ).read_text(encoding="utf-8")
    )
    cross_schema = json.loads(
        (ROOT / "schemas" / "artifact_contracts" / "cross_office_posterior.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert posterior_path.exists()
    assert diagnostics_path.exists()
    assert fundamentals_prior_path.exists()
    assert seat_posterior_path.exists()
    assert state_space_trajectory_path.exists()
    assert pollster_house_effects_path.exists()
    assert senate_joint_path.exists()
    assert house_hierarchical_path.exists()
    assert cross_office_path.exists()
    assert not posterior.is_empty()
    assert not fundamentals_prior.is_empty()
    assert not seat_posterior.is_empty()
    assert not state_space_trajectory.is_empty()
    assert not pollster_house_effects.is_empty()
    assert not senate_joint.is_empty()
    assert not house_hierarchical.is_empty()
    assert not cross_office.is_empty()
    assert {"model_config_hash", "source_manifest_hash"}.issubset(posterior.columns)
    assert {"model_config_hash", "source_manifest_hash"}.issubset(state_space_trajectory.columns)
    assert {"model_config_hash", "source_manifest_hash"}.issubset(pollster_house_effects.columns)
    assert diagnostics["engine"] == "bayes-analytic-logit-normal"
    assert diagnostics["fundamentals_prior_used"] is True
    assert diagnostics["model_config_hash"] == posterior["model_config_hash"][0]
    assert diagnostics["source_manifest_hash"] == posterior["source_manifest_hash"][0]
    assert "posterior_draws.parquet" in fingerprint["stable_artifacts"]
    assert "state_space_trajectory.parquet" in fingerprint["stable_artifacts"]
    assert "pollster_house_effects.parquet" in fingerprint["stable_artifacts"]
    assert "posterior_diagnostics.json" in fingerprint["stable_artifacts"]
    assert "fundamentals_prior.parquet" in fingerprint["stable_artifacts"]
    assert "seat_posterior.parquet" in fingerprint["stable_artifacts"]
    assert "senate_joint_posterior.parquet" in fingerprint["stable_artifacts"]
    assert "house_hierarchical_posterior.parquet" in fingerprint["stable_artifacts"]
    assert "cross_office_posterior.parquet" in fingerprint["stable_artifacts"]
    assert (out_dir / "senate_seat_posterior.parquet").exists()
    assert (out_dir / "house_seat_posterior.parquet").exists()
    assert (out_dir / "governor_seat_posterior.parquet").exists()
    assert {"draw_id", "control_body", "seat_count_total", "majority"}.issubset(
        seat_posterior.columns
    )
    assert set(posterior_schema["required_columns"]).issubset(posterior.columns)
    assert set(senate_schema["required_columns"]).issubset(senate_joint.columns)
    assert set(house_schema["required_columns"]).issubset(house_hierarchical.columns)
    assert set(cross_schema["required_columns"]).issubset(cross_office.columns)
    assert diagnostics["office_methodology"]["senate_joint"]["status"] == "fitted"
    assert diagnostics["office_methodology"]["house_hierarchical"]["dense_covariance_used"] is False
    assert diagnostics["office_methodology"]["cross_office"]["shared_draw_stream"] is True
    assert senate_joint["senate_class"].drop_nulls().n_unique() >= 1
    assert house_hierarchical["redistricting_era"].drop_nulls().n_unique() >= 1
    assert cross_office["shared_draw_stream"].all()
    assert plot_manifest["posterior"]
    assert plot_manifest["fundamentals_prior"]
    assert 'id="posterior_diagnostics"' in html
    assert 'id="fundamentals_prior"' in html
    assert performance["posterior_draws_used"] is True
    assert performance["posterior_draw_uncertainty_mode"] == "posterior_plus_simulation_error"
    assert reward_card["rewards"]["R13_posterior_quality"]["passed"] is True
    assert verification["passed"] is True
    assert any(check["name"] == "state_space_trajectory_schema" for check in verification["checks"])
    assert any(check["name"] == "pollster_house_effects_schema" for check in verification["checks"])
    assert any(check["name"] == "senate_seat_posterior_schema" for check in verification["checks"])
    assert any(check["name"] == "house_seat_posterior_schema" for check in verification["checks"])
    assert any(
        check["name"] == "governor_seat_posterior_schema" for check in verification["checks"]
    )
    assert any(check["name"] == "senate_joint_posterior_schema" for check in verification["checks"])
    assert any(
        check["name"] == "house_hierarchical_posterior_schema" for check in verification["checks"]
    )
    assert any(check["name"] == "cross_office_posterior_schema" for check in verification["checks"])
    assert update["strategy"] == "reweighting"
    assert update["needs_full_refit"] is False
    assert update["reward_status"] is True
    assert (out_dir / "posterior_history.parquet").exists()
    assert (out_dir / "latest_daily_update.json").exists()
    assert (out_dir / "updates" / "2026-05-09" / "daily_update_diagnostics.json").exists()
    updated_reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    assert updated_reward_card["rewards"]["R15_daily_update_quality"]["passed"] is True
    assert updated_reward_card["rewards"]["R15_daily_update_quality"]["metric"]["status"] == (
        "updated"
    )
    assert verification_after_update["passed"] is True
    assert (out_dir / "verification.json").exists()
    assert "posterior draws" in (out_dir / "inference.log").read_text(encoding="utf-8")
    assert (out_dir / "inference.html").stat().st_size > 0
    assert "bayes-analytic-logit-normal" in model_card
    assert "fundamentals_prior" in model_card
    assert "office_methodology" in model_card

    posterior_sums = (
        posterior.group_by(["race_id", "draw_id"])
        .agg(
            pl.len().alias("option_count"),
            pl.col("latent_share").sum().alias("share_sum"),
        )
        .filter(pl.col("option_count") > 1)
    )
    forecast_sums = forecast_draws.group_by(["race_id", "draw_id"]).agg(
        pl.col("vote_share").sum().alias("share_sum")
    )
    joined_draws = posterior.select(["draw_id", "race_id", "option_id", "latent_share"]).join(
        forecast_draws.select(["draw_id", "race_id", "option_id", "vote_share"]),
        on=["draw_id", "race_id", "option_id"],
        how="inner",
    )
    assert posterior_sums["share_sum"].to_list() == pytest.approx([1.0] * posterior_sums.height)
    assert forecast_sums["share_sum"].to_list() == pytest.approx([1.0] * forecast_sums.height)
    assert not joined_draws.is_empty()
    assert (
        joined_draws.select((pl.col("latent_share") - pl.col("vote_share")).abs().max()).item()
        > 1e-4
    )


def test_multioffice_verification_scenario_filters_expected_offices(
    tmp_path: Path,
) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    scenario = ScenarioRegistry.from_context(ctx).get("2026-multioffice-verification")
    assert scenario is not None
    filtered = scenario.filter_catalog(bundle.race_catalog)

    assert set(filtered["office_type"].unique().to_list()) == {
        "governor",
        "house",
        "president",
        "senate",
    }
    assert filtered["cycle"].unique().to_list() == [2026]
    tracker = filtered.filter(pl.col("race_id") == "US-PRES-TRACKER-2026").row(0, named=True)
    assert tracker["control_body"] is None
    assert tracker["seats"] == 0
    assert set(scenario.metadata()["expected_offices"]) == {
        "governor",
        "house",
        "president",
        "senate",
    }


def test_historical_calibration_audit_writes_phase_gates(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).verify_historical_calibration(
        run_id="historical-calibration-unit",
        bayesian_backend="analytic",
        quiet=True,
    )
    out_dir = Path(payload["output_dir"])
    office_frame = pl.read_parquet(out_dir / "office_calibration.parquet")
    persisted = json.loads((out_dir / "historical_calibration.json").read_text(encoding="utf-8"))

    assert payload["passed"] is False
    assert persisted["passed"] is False
    assert {row["office_type"] for row in payload["office_calibration"]} == {
        "governor",
        "house",
        "senate",
    }
    assert set(office_frame["office_type"].to_list()) == {"governor", "house", "senate"}
    assert payload["gates"]["phase4_senate"]["passed"] is True
    assert payload["gates"]["phase5_house"]["passed"] is True
    assert payload["gates"]["phase7_cross_office"]["passed"] is False
    assert (
        payload["gates"]["phase7_cross_office"]["per_office"]["governor"][
            "expected_calibration_error"
        ]
        > 0.06
    )
    assert (out_dir / "historical_calibration_comparison.parquet").exists()
    assert (out_dir / "historical_calibration.md").exists()
    assert (
        ctx.artifacts_dir
        / "runs"
        / "historical-calibration-unit-forecast"
        / "posterior_draws.parquet"
    ).exists()


def test_historical_panel_source_config_builds_production_dimension_scope(
    tmp_path: Path,
) -> None:
    ctx = ProjectContext.create(
        root=ROOT,
        sources_config="sources_historical_panels.yaml",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    SyncRunner(ctx).run()
    CuratedDataBuilder(ctx).run()
    bundle = FeatureBuilder(ctx).run()
    counts = {
        (int(row["cycle"]), str(row["office_type"])): int(row["len"])
        for row in bundle.race_catalog.group_by(["cycle", "office_type"])
        .len()
        .iter_rows(named=True)
    }

    assert counts[(2022, "senate")] >= 33
    assert counts[(2022, "house")] >= 435
    assert counts[(2026, "senate")] >= 33
    assert counts[(2026, "house")] >= 435
    assert bundle.polls.height > 9000
    assert bundle.fundamentals.height > 16000
    assert bundle.results.height > 5600


def test_phase8_verification_runner_smoke(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = Phase8VerificationRunner(ctx).run(
        run_id="phase8-unit",
        scenario="2026-multioffice-verification",
        as_of="2026-05-08",
        bayesian_backend="analytic",
        quiet=True,
    )
    out_dir = Path(payload["output_dir"])
    fingerprint = json.loads(
        (out_dir / "reproducibility_fingerprint.json").read_text(encoding="utf-8")
    )

    assert payload["passed"] is True
    assert payload["bayesian_backend"] == "analytic"
    assert payload["fixture_scope"]["expected_offices"] == [
        "president",
        "senate",
        "house",
        "governor",
    ]
    assert payload["fixture_scope"]["governor_status"] == "fixture_covered"
    assert payload["fixture_scope"]["president_tracker_status"] == "fixture_non_control_tracker"
    assert payload["fixture_scope"]["live_2026_status"] == "not_claimed"
    assert payload["fixture_scope"]["live_source_scope"]["live_2026_rows"] == 0
    assert payload["fixture_scope"]["live_source_scope"]["expected_offices"] == [
        "governor",
        "house",
        "senate",
    ]
    assert payload["artifact_verification"]["passed"] is True
    assert payload["visual_qa"]["passed"] is True
    assert payload["daily_update"]["reward_status"] is True
    assert payload["timeout_failover_audit"]["passed"] is True
    assert fingerprint["cross_run_verified"] is True
    assert (out_dir / "phase8_verification.json").exists()
    assert (out_dir / "visual_qa_checklist.json").exists()
    assert (out_dir / "verification.json").exists()
    assert (out_dir / "timeout_failover_audit.json").exists()
    assert (out_dir / "cross_office_posterior.parquet").exists()
    assert (out_dir / "governor_seat_posterior.parquet").exists()
    seat_posterior = pl.read_parquet(out_dir / "seat_posterior.parquet")
    cross_office = pl.read_parquet(out_dir / "cross_office_posterior.parquet")
    assert "president" not in set(seat_posterior["control_body"].drop_nulls().to_list())
    assert "president" in set(cross_office["office_type"].to_list())
    verification = json.loads((out_dir / "verification.json").read_text(encoding="utf-8"))
    assert any(check["name"] == "timeout_failover_audit" for check in verification["checks"])


def test_reproducibility_hash_ignores_json_runtime_timing(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "engine": "numpyro-nuts",
                "elapsed_seconds": 0.11,
                "generated_at": "2026-05-08T00:00:00+00:00",
                "failover_audit": {"elapsed_seconds": 0.01, "status": "completed"},
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "engine": "numpyro-nuts",
                "elapsed_seconds": 0.42,
                "generated_at": "2026-05-08T00:01:00+00:00",
                "failover_audit": {"elapsed_seconds": 0.02, "status": "completed"},
            }
        ),
        encoding="utf-8",
    )

    assert ForecastPipeline._stable_artifact_hash(first) == ForecastPipeline._stable_artifact_hash(
        second
    )


def test_reward_card_verification_fails_hard_posterior_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "reward_card.json").write_text(
        json.dumps(
            {
                "rewards": {
                    "R1_reproducibility": {"passed": False},
                    "R13_posterior_quality": {
                        "passed": False,
                        "metric": {"divergences": 3},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    check = ForecastPipeline._verify_reward_card(run_dir)

    assert check["passed"] is False
    assert check["detail"]["hard_gate_failures"] == ["R13_posterior_quality"]


def test_methodology_readiness_audit_blocks_default_switch_without_live_scope(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    run_dir = ctx.artifacts_dir / "runs" / "phase8-readiness"
    run_dir.mkdir(parents=True)
    (run_dir / "fundamentals_prior.parquet").write_bytes(b"placeholder")
    (run_dir / "posterior_diagnostics.json").write_text(
        json.dumps(
            {
                "engine": "numpyro-nuts",
                "fundamentals_prior_used": True,
                "fundamentals_prior_rows": 8,
                "divergences": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reward_card.json").write_text(
        json.dumps(
            {
                "rewards": {
                    "R13_posterior_quality": {"passed": True},
                    "R14_calibrated_publication": {"passed": True},
                    "R15_daily_update_quality": {"passed": True},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "phase8_verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "scenario": "2026-multioffice-verification",
                "inference_engine": "bayes",
                "bayesian_backend": "nuts",
                "fixture_scope": {
                    "status": "fixture_verification",
                    "live_2026_status": "not_claimed",
                },
            }
        ),
        encoding="utf-8",
    )
    bayes_dir = ctx.artifacts_dir / "backtests" / "bayes-bt"
    legacy_dir = ctx.artifacts_dir / "backtests" / "legacy-bt"
    bayes_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    (bayes_dir / "scorecard.json").write_text(
        json.dumps(
            {
                "inference_engine": "bayes",
                "metrics": {"ensemble": {"log_score": 0.10, "interval_90_coverage": 0.92}},
            }
        ),
        encoding="utf-8",
    )
    (legacy_dir / "scorecard.json").write_text(
        json.dumps(
            {
                "inference_engine": "kalman",
                "metrics": {"ensemble": {"log_score": 0.20, "interval_90_coverage": 0.90}},
            }
        ),
        encoding="utf-8",
    )

    payload = ForecastPipeline(ctx).assess_methodology_readiness(
        run_id="readiness-unit",
        forecast_run_id="phase8-readiness",
        bayes_backtest_run_id="bayes-bt",
        legacy_backtest_run_id="legacy-bt",
    )
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["status"] == "blocked"
    assert payload["eligible_for_default_switch"] is False
    assert checks["fundamentals_prior_production_path"]["passed"] is True
    assert checks["reward_hard_gates"]["passed"] is True
    assert checks["phase8_verification_passed"]["passed"] is True
    assert checks["rolling_origin_beats_legacy"]["passed"] is True
    assert checks["bayes_dependencies_in_base"]["passed"] is True
    assert checks["bayesian_config_default"]["passed"] is True
    assert checks["docs_declare_bayes_default"]["passed"] is True
    assert checks["live_2026_source_scope"]["passed"] is False
    assert "model-bearing rows" in checks["live_2026_source_scope"]["detail"]["policy"]
    assert (
        ctx.artifacts_dir / "readiness" / "readiness-unit" / "methodology_readiness.json"
    ).exists()
    assert (
        ctx.artifacts_dir / "readiness" / "readiness-unit" / "methodology_readiness.md"
    ).exists()


def test_nuts_timeout_uses_visible_fallback() -> None:
    policy = FailoverPolicy(timeout_seconds=0.01)

    result = execute_with_failover(
        primary=lambda: time.sleep(0.05),
        fallback=lambda: "cached-posterior",
        policy=policy,
        primary_engine="numpyro-nuts",
    )

    assert result.result == "cached-posterior"
    assert result.audit["status"] == "fallback_used"
    assert result.audit["fallback_used"] == "previous_posterior_reuse"
    assert result.audit["publication_blocked"] is True

    success = execute_with_failover(
        primary=lambda: "nuts-complete",
        fallback=None,
        policy=FailoverPolicy(timeout_seconds=1.0),
        primary_engine="numpyro-nuts",
    )
    assert success.result == "nuts-complete"
    assert success.audit["status"] == "completed"
    assert success.audit["fallback_used"] is None

    parsed = FailoverPolicy.from_config(
        {
            "bayesian": {
                "nuts": {
                    "wall_clock_timeout_seconds": 12,
                    "failover": {
                        "fallback_order": "cached_posterior,kalman_legacy_fallback",
                        "block_publication_on_fallback": False,
                    },
                }
            }
        }
    )
    assert parsed.timeout_seconds == 12
    assert parsed.fallback_order == ("cached_posterior", "kalman_legacy_fallback")
    assert parsed.block_publication_on_fallback is False
    assert exercise_timeout_failover(policy)["passed"] is True


def test_forecast_requires_as_of_without_scenario(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    try:
        ForecastPipeline(ctx).run_forecast(as_of=None, run_id="missing-date")
    except ValueError as exc:
        assert "as_of is required" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("forecast without as_of or scenario default should fail")


def test_backtest_and_report_rebuild(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2026-05-08", run_id="reportable", inference_engine="kalman")
    payload = pipeline.run_backtest(run_id="bt", inference_engine="kalman")
    report_dir = pipeline.rebuild_report("reportable")
    backtest_dir = ctx.artifacts_dir / "backtests" / "bt"

    assert payload["row_count"] >= 30
    assert payload["inference_engine"] == "kalman"
    assert payload["rolling_origin_executed"] is True
    assert payload["sample_size_too_small"] is False
    assert (backtest_dir / "scorecard.json").exists()
    assert (backtest_dir / "rolling_predictions.parquet").exists()
    assert (backtest_dir / "component_admission.json").exists()
    assert (backtest_dir / "ensemble_learning.json").exists()
    assert (backtest_dir / "probability_calibration.json").exists()
    assert (backtest_dir / "recalibration_map.parquet").exists()
    assert (backtest_dir / "bayesian_hyperpriors.json").exists()
    assert (backtest_dir / "residual_covariance.parquet").exists()
    rolling = pl.read_parquet(backtest_dir / "rolling_predictions.parquet")
    assert set(rolling["as_of_offset_days"].unique().to_list()) == {1, 7, 30, 60, 90}
    assert rolling["polling_inference_engine"].unique().to_list() == ["kalman"]
    assert {
        "configured_ensemble_probability",
        "learned_ensemble_probability",
        "ensemble_probability",
    }.issubset(rolling.columns)
    admission = json.loads((backtest_dir / "component_admission.json").read_text())
    learned_weights = admission["ensemble_learning"]["component_weights"]
    assert abs(sum(learned_weights.values()) - 1.0) < 1e-9
    assert admission["probability_calibration"]["status"] == "fitted"
    assert admission["bayesian_hyperpriors"]["status"] == "fitted"
    recalibration_map = pl.read_parquet(backtest_dir / "recalibration_map.parquet")
    assert recalibration_map["status"].to_list() == ["fitted"]
    assert recalibration_map["sample_size"].to_list() == [rolling.height]
    covariance = pl.read_parquet(backtest_dir / "residual_covariance.parquet")
    assert {"matrix_rank", "covariance_method"}.issubset(covariance.columns)
    assert (report_dir / "model_card.md").exists()
    assert (
        (report_dir / "diagnostics.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    )


def test_backtest_can_score_bayesian_polling_bridge(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_backtest(
        run_id="bayes-bt",
        scenario="president_state",
        holdout_cycle=2024,
        inference_engine="bayes",
        bayesian_backend="analytic",
    )
    backtest_dir = ctx.artifacts_dir / "backtests" / "bayes-bt"
    rolling = pl.read_parquet(backtest_dir / "rolling_predictions.parquet")
    scorecard = json.loads((backtest_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert payload["inference_engine"] == "bayes"
    assert scorecard["inference_engine"] == "bayes"
    assert rolling["polling_inference_engine"].unique().to_list() == ["bayes"]
    assert rolling.height == payload["row_count"]
    assert "polls_probability" in rolling.columns
    assert not rolling.select("polls_probability").drop_nulls().is_empty()
    assert (backtest_dir / "recalibration_map.parquet").exists()
    assert (backtest_dir / "bayesian_hyperpriors.json").exists()


def test_hyperprior_refresh_writes_candidates_without_promoting_latest(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).refresh_hyperpriors(
        run_id="hyper-refresh",
        scenarios=["president_state"],
        holdout_cycle=2024,
        inference_engine="bayes",
        bayesian_backend="analytic",
    )
    out_dir = Path(payload["output_dir"])
    scenario_dir = out_dir / "president_state"
    manifest = json.loads((out_dir / "hyperprior_refresh_manifest.json").read_text())
    comparison = json.loads((scenario_dir / "comparison_report.json").read_text())

    assert payload["status"] == "candidate_refresh"
    assert payload["promoted"] is False
    assert manifest["scenarios"][0]["scenario"] == "president_state"
    assert comparison["promotion_blocked"] is True
    assert comparison["promotion_recommendation"] == "manual_review_required"
    assert (scenario_dir / "bayesian_hyperpriors_president_state.json").exists()
    assert (scenario_dir / "rolling_predictions_candidate.parquet").exists()
    assert (out_dir / "comparison_report.md").exists()
    assert not (ctx.artifacts_dir / "backtests" / "latest" / "index.json").exists()


def test_phase0_spike_writes_comparison_artifacts(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_phase0_spike(
        run_id="phase0-unit",
        scenario="president_state",
        holdout_cycle=2024,
        bayesian_backend="analytic",
    )
    out_dir = Path(payload["output_dir"])
    comparison = pl.read_parquet(out_dir / "phase0_comparison.parquet")
    comparison_json = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))

    assert payload["run_id"] == "phase0-unit"
    assert set(comparison["engine"].to_list()) == {"kalman", "bayes"}
    assert comparison_json["go_no_go"]["metric"] == "ensemble_log_score"
    assert (out_dir / "rolling_predictions_kalman.parquet").exists()
    assert (out_dir / "rolling_predictions_bayes.parquet").exists()
    assert (out_dir / "scorecard_kalman.json").exists()
    assert (out_dir / "scorecard_bayes.json").exists()


def test_phase0b_spike_rejects_global_smc_and_selects_strategy(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_phase0b_spike(run_id="phase0b-unit")
    out_dir = Path(payload["output_dir"])
    summary = json.loads((out_dir / "phase0b_summary.json").read_text(encoding="utf-8"))
    geometry = pl.read_parquet(out_dir / "geometry_comparison.parquet")
    bakeoff = pl.read_parquet(out_dir / "acceleration_bakeoff.parquet")

    assert payload["run_id"] == "phase0b-unit"
    assert payload["global_smc_rejected"] is True
    assert payload["selected_strategy"] == "reweighting"
    assert summary["geometry_gate"]["default_parameterization"] == "noncentered"
    assert summary["geometry_gate"]["divergences"] == 0
    assert set(geometry["parameterization"].to_list()) == {"centered", "noncentered"}
    assert "combined" in bakeoff["scope"].to_list()
    combined_global = bakeoff.filter(
        (pl.col("scope") == "combined") & (pl.col("strategy") == "global_smc")
    )
    assert combined_global.select("accepted").item() is False
    assert combined_global.select("failure_mode").item() == "weight_collapse"
    assert summary["fallback_semantics"]["reweighting"].startswith("trigger full NUTS refit")


def test_presidential_result_comparison(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2024-10-01", run_id="pres-2024", inference_engine="kalman")
    payload = pipeline.compare_results(
        forecast_run_id="pres-2024",
        comparison_id="pres-actuals",
        cycle=2024,
        office_type="president",
    )
    comparison_dir = Path(payload["output_dir"])
    comparison = pl.read_parquet(comparison_dir / "result_comparison.parquet")

    assert payload["race_count"] >= 1
    assert payload["row_count"] >= payload["race_count"] * 2
    assert 0.0 <= payload["winner_accuracy"] <= 1.0
    assert 0.0 <= payload["state_accuracy"] <= 1.0
    assert payload["state_accuracy_n"] == payload["race_count"]
    assert payload["ec_winner_accuracy"] in {0.0, 1.0}
    assert payload["electoral_college"]["scope"] in {
        "full_electoral_college",
        "modeled_state_slice",
    }
    assert payload["actual_winner_probabilities"]
    assert payload["largest_misses"]
    assert (comparison_dir / "result_comparison_summary.json").exists()
    assert (comparison_dir / "result_comparison.html").exists()
    assert (comparison_dir / "narrative.md").exists()
    assert (comparison_dir / "race_outcomes.parquet").exists()
    assert (comparison_dir / "largest_misses.parquet").exists()
    html_doc = (comparison_dir / "result_comparison.html").read_text(encoding="utf-8")
    assert "result-plot-grid" in html_doc
    assert "Race-By-Race Outcomes" in html_doc
    assert (comparison_dir / "plots" / "vote_share_forecast_vs_actual.png").stat().st_size > 0
    assert (comparison_dir / "plots" / "actual_winner_probabilities.png").stat().st_size > 0
    assert (comparison_dir / "plots" / "actual_winner_probability_swarm.png").stat().st_size > 0
    assert (comparison_dir / "plots" / "largest_vote_share_misses.png").stat().st_size > 0
    assert {
        "actual_winner_probability",
        "race_winner_correct",
        "predicted_winner_party",
        "actual_winner_party",
    }.issubset(comparison.columns)
    assert comparison.filter(pl.col("actual_winner")).height == payload["race_count"]


def test_result_comparison_handles_withheld_vote_share_projection(tmp_path: Path) -> None:
    race_catalog = pl.DataFrame(
        [
            {
                "race_id": "US-HOUSE-XX-01-2026",
                "cycle": 2026,
                "election_date": "2026-11-03",
                "geography_type": "district",
                "geography": "XX-01",
                "office_type": "house",
                "race_type": "general",
                "seats": 1,
                "control_body": "house",
                "tier": "C",
                "tier_reason": "sparse fixture",
            }
        ]
    )
    race_forecasts = pl.DataFrame(
        [
            {
                "race_id": "US-HOUSE-XX-01-2026",
                "option_id": "US-HOUSE-XX-01-2026-DEM",
                "winner_probability": None,
            },
            {
                "race_id": "US-HOUSE-XX-01-2026",
                "option_id": "US-HOUSE-XX-01-2026-REP",
                "winner_probability": None,
            },
        ]
    )
    curated_results = pl.DataFrame(
        [
            {
                "race_id": "US-HOUSE-XX-01-2026",
                "option_id": "US-HOUSE-XX-01-2026-DEM",
                "vote_share": 0.48,
                "turnout": 100_000,
                "winner": False,
            },
            {
                "race_id": "US-HOUSE-XX-01-2026",
                "option_id": "US-HOUSE-XX-01-2026-REP",
                "vote_share": 0.52,
                "turnout": 100_000,
                "winner": True,
            },
        ]
    )

    comparison = ResultComparator()._comparison_frame(
        race_catalog=race_catalog,
        race_forecasts=race_forecasts,
        curated_results=curated_results,
        cycle=2026,
        office_type="house",
        race_id=None,
    )
    race_outcomes = ResultComparator._race_outcome_frame(comparison)

    assert "vote_share_mean" in comparison.columns
    assert comparison["absolute_vote_share_error"].null_count() == comparison.height
    assert race_outcomes["predicted_winner_option_id"].null_count() == race_outcomes.height
    assert race_outcomes["race_winner_correct"].null_count() == race_outcomes.height

    empty_comparison = ResultComparator()._comparison_frame(
        race_catalog=race_catalog,
        race_forecasts=race_forecasts,
        curated_results=curated_results.head(0),
        cycle=2026,
        office_type="house",
        race_id=None,
    )
    manifest = ResultComparator()._write_plots(empty_comparison, tmp_path / "empty-comparison")

    assert empty_comparison.is_empty()
    assert manifest == {"comparison": []}


def test_cycle_eval_writes_consolidated_dashboard(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_cycle_eval(
        cycles=[2020, 2024],
        as_of_mm_dd="10-05",
        run_id="cycle-smoke",
        inference_engine="kalman",
    )
    out_dir = Path(payload["output_dir"])
    summary = pl.read_parquet(out_dir / "cycle_summary.parquet")

    assert payload["cycle_count"] == 2
    assert summary.height == 2
    assert {
        "actual_ec_winner_party",
        "forecast_ec_winner_party",
        "state_accuracy",
        "state_topline_ec_winner_party",
        "brier_score",
    }.issubset(summary.columns)
    assert payload["aggregate"]["ec_winner_accuracy"] in {0.0, 0.5, 1.0}
    assert (out_dir / "cycle_summary.json").exists()
    assert (out_dir / "cycle_eval.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (out_dir / "narrative.md").exists()
    assert (out_dir / "plots" / "ec_winner_probability_by_cycle.png").stat().st_size > 0
    assert (out_dir / "plots" / "accuracy_brier_by_cycle.png").stat().st_size > 0
    assert (out_dir / "plots" / "error_upsets_by_cycle.png").stat().st_size > 0
    assert (ctx.artifacts_dir / "runs" / "eval-2024-1005" / "diagnostics.html").exists()
    assert (
        ctx.artifacts_dir
        / "runs"
        / "eval-2024-1005"
        / "comparisons"
        / "actuals"
        / "result_comparison.html"
    ).exists()
    reuse_payload = ForecastPipeline(ctx).run_cycle_eval(
        cycles=[2020, 2024],
        as_of_mm_dd="10-05",
        run_id="cycle-smoke-reuse",
        reuse_existing=True,
        inference_engine="kalman",
    )
    assert reuse_payload["aggregate"] == payload["aggregate"]


def test_cycle_eval_report_handles_null_comparison_metrics(tmp_path: Path) -> None:
    rows = [
        {
            "cycle": 2024,
            "as_of": "2024-10-05",
            "forecast_run_id": "senate-cycle-2024-1005",
            "control_body": "senate",
            "majority_threshold": 51,
            "forecast_ec_winner_party": "DEM",
            "actual_ec_winner_party": "DEM",
            "state_topline_ec_winner_party": None,
            "state_topline_ec_winner_accuracy": None,
            "forecast_ec_win_probability": 0.62,
            "forecast_ec_p10": 48,
            "forecast_ec_p50": 51,
            "forecast_ec_p90": 54,
            "dem_seat_count_mean": 51.2,
            "rep_seat_count_mean": 48.8,
            "dem_majority_probability": 0.62,
            "rep_majority_probability": 0.38,
            "ec_winner_accuracy": 1.0,
            "state_accuracy": 0.8,
            "state_accuracy_n": 33,
            "brier_score": 0.12,
            "mean_absolute_vote_share_error": 0.03,
            "upset_count": 1,
            "missed_states": "OH",
            "race_count": 33,
            "diagnostics_path": "runs/senate-cycle-2024-1005/diagnostics.html",
            "comparison_path": "runs/senate-2024/comparisons/actuals/result_comparison.html",
        },
        {
            "cycle": 2026,
            "as_of": "2026-10-05",
            "forecast_run_id": "senate-cycle-2026-1005",
            "control_body": "senate",
            "majority_threshold": 51,
            "forecast_ec_winner_party": "REP",
            "actual_ec_winner_party": None,
            "state_topline_ec_winner_party": None,
            "state_topline_ec_winner_accuracy": None,
            "forecast_ec_win_probability": 0.58,
            "forecast_ec_p10": 49,
            "forecast_ec_p50": 52,
            "forecast_ec_p90": 55,
            "dem_seat_count_mean": 48.7,
            "rep_seat_count_mean": 51.3,
            "dem_majority_probability": 0.42,
            "rep_majority_probability": 0.58,
            "ec_winner_accuracy": None,
            "state_accuracy": None,
            "state_accuracy_n": 0,
            "brier_score": None,
            "mean_absolute_vote_share_error": None,
            "upset_count": None,
            "missed_states": "",
            "race_count": 0,
            "diagnostics_path": "runs/senate-cycle-2026-1005/diagnostics.html",
            "comparison_path": "runs/senate-2026/comparisons/actuals/result_comparison.html",
        },
    ]

    payload = CycleEvaluationReport().render(
        rows=rows,
        output_dir=tmp_path / "cycle-null-metrics",
        run_id="cycle-null-metrics",
        as_of_mm_dd="10-05",
    )
    out_dir = Path(payload["output_dir"])

    assert payload["aggregate"]["mean_vote_share_mae"] == 0.03
    assert (out_dir / "plots" / "accuracy_brier_by_cycle.png").stat().st_size > 0
    assert (out_dir / "plots" / "error_upsets_by_cycle.png").stat().st_size > 0
    assert "n/a" in (out_dir / "cycle_eval.html").read_text(encoding="utf-8")


def test_cycle_eval_preflights_dates_and_scenarios(tmp_path: Path) -> None:
    pipeline = ForecastPipeline(context(tmp_path))
    try:
        pipeline.run_cycle_eval(cycles=[2023], as_of_mm_dd="02-29", run_id="bad-date")
    except ValueError as exc:
        assert "2023-02-29 is not a valid date" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("invalid per-cycle date should fail before forecasting")

    try:
        pipeline.run_cycle_eval(cycles=[2026], as_of_mm_dd="10-05", run_id="bad-scenario")
    except ValueError as exc:
        assert "unknown scenario 'president_2026_state'" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("unknown scenario should fail before forecasting")


def test_presidential_scenario_writes_ec_plot_and_latest_backtest_artifacts(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    payload = pipeline.run_backtest(
        run_id="pres-bt",
        scenario="president_state",
        holdout_cycle=2024,
        inference_engine="kalman",
    )
    out_dir = pipeline.run_forecast(
        as_of=None,
        run_id="pres-scenario",
        scenario="president_2024_state",
        inference_engine="kalman",
    )
    race_catalog = pl.read_parquet(out_dir / "race_catalog.parquet")

    assert race_catalog["cycle"].unique().to_list() == [2024]
    assert race_catalog["office_type"].unique().to_list() == ["president"]
    assert race_catalog.height == 51
    assert race_catalog["seats"].sum() == 538
    assert (out_dir / "plots" / "electoral_college_distribution.png").stat().st_size > 0
    assert (out_dir / "plots" / "electoral_college_chain_traces.png").stat().st_size > 0
    assert (out_dir / "plots" / "topline_electoral_swarm.png").stat().st_size > 0
    diagnostics = (out_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "kpi-strip" in diagnostics
    assert "overview-plot-grid" in diagnostics
    assert diagnostics.index("plots/topline_electoral_swarm.png") < diagnostics.index(
        "Where The Forecast Lives"
    )
    assert payload["row_count"] >= 30
    assert payload["sample_size_too_small"] is False
    assert (out_dir / "poll_trajectory.parquet").stat().st_size > 0
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "component_admission_president_state.json"
    ).exists()
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "ensemble_learning_president_state.json"
    ).exists()
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "probability_calibration_president_state.json"
    ).exists()
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "recalibration_map_president_state.parquet"
    ).exists()
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "bayesian_hyperpriors_president_state.json"
    ).exists()
    assert (
        ctx.artifacts_dir / "backtests" / "latest" / "residual_covariance_president_state.parquet"
    ).exists()
    assert (out_dir / "recalibration_map.parquet").exists()
    forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
    assert forecasts["probability_calibration_status"].drop_nulls().unique().to_list() == ["fitted"]
    reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    assert reward_card["rewards"]["R14_calibrated_publication"]["passed"] is True
    verification = pipeline.verify_run("pres-scenario")
    assert verification["passed"] is True
    assert any(check["name"] == "recalibration_map_schema" for check in verification["checks"])


def test_http_sync_and_538_polls_normalizer(tmp_path: Path) -> None:
    csv_payload = (
        "cycle,state,pollster,poll_id,question_id,start_date,end_date,sample_size,population,"
        "methodology,internal,partisan,stage,answer,candidate_party,pct\n"
        "2020,Wisconsin,Acme Poll,1001,77,10/01/20,10/03/20,800,LV,Live Phone,,,General,"
        "Smith,DEM,49.5\n"
        "2020,Wisconsin,Acme Poll,1001,77,10/01/20,10/03/20,800,LV,Live Phone,,,General,"
        "Jones,REP,48.1\n"
        "2020,Ohio,Other Poll,1002,88,10/04/20,10/05/20,700,RV,Online Panel,,,General,"
        "Lee,DEM,40.0\n"
    )
    source_path = tmp_path / "538_president.csv"
    source_path.write_text(csv_payload, encoding="utf-8")

    ctx = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    SyncRunner(ctx).run()
    fixture_registry = SourceRegistry.from_context(ctx)

    extra = SourceDefinition(
        id="fivethirtyeight_president_polls_test",
        table="polls",
        type="http_csv",
        path=source_path,
        parser_version="fivethirtyeight-president-polls-v1",
        license="Test fixture for 538-format normalization.",
        url=source_path.resolve().as_uri(),
        auth_mode="public",
        parser_args={
            "cycle": 2020,
            "state": "wisconsin",
            "stage": "general",
            "race_id": "US-PRES-WI-2020",
            "parties": ["DEM", "REP"],
        },
    )
    registry = SourceRegistry([*fixture_registry.sources, extra])
    SyncRunner(ctx, registry=registry).run()
    result = CuratedDataBuilder(ctx).run()

    polls = result.tables["polls"]
    wi_rows = polls.filter(
        (pl.col("race_id") == "US-PRES-WI-2020") & pl.col("poll_id").str.starts_with("538-")
    )
    assert wi_rows.height == 2
    assert {"D", "R"}.issubset(set(wi_rows["option_id"].str.slice(-1).to_list()))
    assert wi_rows["methodology"].unique().to_list() == ["live_phone"]
    assert _CuratedDataBuilderClass is CuratedDataBuilder


def test_538_general_polls_normalizer_covers_2026_offices(tmp_path: Path) -> None:
    csv_payload = (
        "cycle,state,pollster,poll_id,question_id,start_date,end_date,sample_size,population,"
        "population_full,methodology,office_type,seat_number,seat_name,internal,partisan,stage,"
        "candidate_party,pct\n"
        "2026,Georgia,Peach State Poll,2001,91,04/10/26,04/12/26,900,lv,lv,Online,"
        "U.S. Senate,2,Class II,,,"
        "General,DEM,50.0\n"
        "2026,Georgia,Peach State Poll,2001,91,04/10/26,04/12/26,900,lv,lv,Online,"
        "U.S. Senate,2,Class II,,,"
        "General,REP,47.0\n"
        "2026,Georgia,Peachtree Poll,2002,92,04/11/26,04/13/26,950,rv,rv,IVR/Text,"
        "Governor,0,,, ,"
        "General,DEM,48.0\n"
        "2026,Georgia,Peachtree Poll,2002,92,04/11/26,04/13/26,950,rv,rv,IVR/Text,"
        "Governor,0,,, ,"
        "General,REP,49.0\n"
        "2026,California,Coastal Poll,2003,93,04/14/26,04/16/26,760,lv,lv,Live Phone,"
        "U.S. House,45,District 45,,,"
        "General,DEM,49.0\n"
        "2026,California,Coastal Poll,2003,93,04/14/26,04/16/26,760,lv,lv,Live Phone,"
        "U.S. House,45,District 45,,,"
        "General,REP,48.0\n"
        "2024,California,Old Poll,2004,94,04/14/24,04/16/24,760,lv,lv,Online,"
        "U.S. House,45,District 45,,,"
        "General,DEM,51.0\n"
    )
    source_path = tmp_path / "538_general.csv"
    source_path.write_text(csv_payload, encoding="utf-8")

    ctx = context(tmp_path)
    fixture_registry = SourceRegistry.from_context(ctx)
    live_sources = [
        SourceDefinition(
            id=f"fivethirtyeight_{office}_polls_test",
            table="polls",
            type="http_csv",
            path=source_path,
            parser_version="fivethirtyeight-general-polls-v1",
            license="Test fixture for 538-format general normalization.",
            url=source_path.resolve().as_uri(),
            auth_mode="public",
            parser_args={
                "cycle": 2026,
                "office": office,
                "stage": "general",
                "parties": ["DEM", "REP"],
                **({"house_race_id_format": "compact"} if office == "house" else {}),
            },
        )
        for office in ["senate", "governor", "house"]
    ]
    SyncRunner(ctx, registry=SourceRegistry([*fixture_registry.sources, *live_sources])).run()
    result = CuratedDataBuilder(ctx).run()

    polls = result.tables["polls"].filter(pl.col("poll_id").str.starts_with("538-"))
    assert polls.height == 6
    assert set(polls["race_id"].to_list()) == {
        "US-SEN-GA-2026",
        "US-GOV-GA-2026",
        "US-HOUSE-CA45-2026",
    }
    assert set(polls["office_type"].to_list()) == {"senate", "governor", "house"}
    assert {"US-HOUSE-CA45-2026-D", "US-HOUSE-CA45-2026-R"}.issubset(
        set(polls["option_id"].to_list())
    )


def test_phase8_live_source_scope_claims_when_non_file_sources_cover_offices(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    ctx.curated_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_id": ["live_senate", "live_house", "live_governor"],
            "table": ["polls", "polls", "polls"],
            "url": [
                "https://example.test/senate.csv",
                "https://example.test/house.csv",
                "https://example.test/governor.csv",
            ],
            "status": ["fetched", "fetched", "fetched"],
        }
    ).write_parquet(ctx.curated_dir / "source_manifest.parquet")
    pl.DataFrame(
        {
            "source_id": ["live_senate", "live_house", "live_governor"],
            "race_id": ["US-SEN-GA-2026", "US-HOUSE-CA45-2026", "US-GOV-GA-2026"],
            "cycle": [2026, 2026, 2026],
            "office_type": ["senate", "house", "governor"],
        }
    ).write_parquet(ctx.curated_dir / "polls.parquet")

    scenario = ScenarioRegistry.from_context(ctx).get("2026-multioffice-verification")
    assert scenario is not None
    payload = Phase8VerificationRunner(ctx)._live_2026_source_scope(scenario, "2026-05-08")

    assert payload["status"] == "claimed"
    assert payload["covered_offices"] == ["governor", "house", "senate"]
    assert payload["live_2026_rows"] == 3
    assert payload["model_signal_2026_rows"] == 3


def test_phase8_live_source_scope_does_not_claim_neutral_metadata_only_rows(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    ctx.curated_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_id": ["wiki_senate", "wiki_house", "wiki_governor"],
            "table": ["public_signals", "public_signals", "public_signals"],
            "url": [
                "https://example.test/senate.wiki",
                "https://example.test/house.wiki",
                "https://example.test/governor.wiki",
            ],
            "status": ["fetched", "fetched", "fetched"],
        }
    ).write_parquet(ctx.curated_dir / "source_manifest.parquet")
    pl.DataFrame(
        {
            "source_id": ["wiki_senate", "wiki_house", "wiki_governor"],
            "race_id": ["US-SEN-GA-2026", "US-HOUSE-CA45-2026", "US-GOV-GA-2026"],
            "cycle": [2026, 2026, 2026],
            "office_type": ["senate", "house", "governor"],
            "z_score": [0.0, 0.0, 0.0],
        }
    ).write_parquet(ctx.curated_dir / "public_signals.parquet")

    scenario = ScenarioRegistry.from_context(ctx).get("2026-multioffice-verification")
    assert scenario is not None
    payload = Phase8VerificationRunner(ctx)._live_2026_source_scope(scenario, "2026-05-08")

    assert payload["status"] == "metadata_only"
    assert payload["covered_offices"] == ["governor", "house", "senate"]
    assert payload["model_signal_covered_offices"] == []
    assert payload["live_2026_rows"] == 3
    assert payload["model_signal_2026_rows"] == 0


def test_phase8_live_source_scope_fills_nullable_fundamental_scope_from_races(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    ctx.curated_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_id": ["live_fred"],
            "table": ["fundamentals"],
            "url": ["https://example.test/fred.csv"],
            "status": ["fetched"],
        }
    ).write_parquet(ctx.curated_dir / "source_manifest.parquet")
    pl.DataFrame(
        {
            "race_id": ["US-SEN-GA-2026", "US-HOUSE-CA45-2026", "US-GOV-GA-2026"],
            "cycle": [2026, 2026, 2026],
            "office_type": ["senate", "house", "governor"],
        }
    ).write_parquet(ctx.curated_dir / "races.parquet")
    pl.DataFrame(
        {
            "source_id": ["live_fred", "live_fred", "live_fred"],
            "race_id": ["US-SEN-GA-2026", "US-HOUSE-CA45-2026", "US-GOV-GA-2026"],
            "cycle": [None, None, None],
            "economic_index": [0.1, 0.1, 0.1],
        }
    ).write_parquet(ctx.curated_dir / "fundamentals.parquet")

    scenario = ScenarioRegistry.from_context(ctx).get("2026-multioffice-verification")
    assert scenario is not None
    payload = Phase8VerificationRunner(ctx)._live_2026_source_scope(scenario, "2026-05-08")

    assert payload["status"] == "claimed"
    assert payload["model_signal_covered_offices"] == ["governor", "house", "senate"]
    assert payload["model_signal_rows_by_table"] == {"fundamentals": 3}


def test_wikipedia_race_presence_parser_emits_neutral_public_signals(tmp_path: Path) -> None:
    wiki_path = tmp_path / "wiki.wikitext"
    wiki_path.write_text(
        "== Georgia ==\n"
        "{{Main|2026 United States Senate election in Georgia}}\n"
        "!{{ushr|CA|45|X}}\n",
        encoding="utf-8",
    )
    ctx = context(tmp_path)
    source = SourceDefinition(
        id="wikipedia_presence_test",
        table="public_signals",
        type="fixture",
        path=wiki_path,
        parser_version="wikipedia-race-presence-signals-v1",
        license="Test Wikipedia race-presence fixture.",
        url=wiki_path.resolve().as_uri(),
        parser_args={
            "cycle": 2026,
            "observed_at": "2026-05-10T00:00:00Z",
            "require_all": True,
            "races": [
                {
                    "race_id": "US-SEN-GA-2026",
                    "office_type": "senate",
                    "patterns": ["2026 United States Senate election in Georgia"],
                },
                {
                    "race_id": "US-HOUSE-CA45-2026",
                    "office_type": "house",
                    "patterns": ["\\{\\{ushr\\|CA\\|45\\|X\\}\\}"],
                },
            ],
        },
    )

    SyncRunner(ctx, registry=SourceRegistry([source])).run()
    result = CuratedDataBuilder(ctx).run()
    signals = result.tables["public_signals"]

    assert signals.height == 4
    assert set(signals["race_id"].to_list()) == {"US-SEN-GA-2026", "US-HOUSE-CA45-2026"}
    assert set(signals["office_type"].to_list()) == {"senate", "house"}
    assert signals["z_score"].sum() == 0.0
    assert signals["leakage_checked"].all()


def test_fred_national_fundamentals_parser_emits_model_bearing_rows(tmp_path: Path) -> None:
    fred_path = tmp_path / "fred_unrate.csv"
    fred_path.write_text(
        "observation_date,UNRATE\n2026-01-01,4.0\n2026-02-01,4.5\n2026-03-01,5.0\n",
        encoding="utf-8",
    )
    ctx = context(tmp_path)
    source = SourceDefinition(
        id="fred_unrate_test",
        table="fundamentals",
        type="fixture",
        path=fred_path,
        parser_version="fred-national-fundamentals-v1",
        license="Test FRED unemployment fixture.",
        url=fred_path.resolve().as_uri(),
        parser_args={
            "series_id": "UNRATE",
            "as_of": "2026-05-08",
            "lookback_observations": 3,
            "economic_index_scale": 0.1,
            "races": [
                {
                    "race_id": "US-SEN-GA-2026",
                    "partisan_lean": 0.4,
                    "incumbency_advantage": 0.5,
                    "demographic_turnout_index": 0.2,
                    "historical_turnout_rate": 0.642,
                    "registered_voters": 5200000,
                },
                {
                    "race_id": "US-HOUSE-CA45-2026",
                    "partisan_lean": -1.6,
                    "incumbency_advantage": -1.0,
                    "demographic_turnout_index": 0.1,
                    "historical_turnout_rate": 0.584,
                    "registered_voters": 510000,
                },
            ],
        },
    )

    SyncRunner(ctx, registry=SourceRegistry([source])).run()
    result = CuratedDataBuilder(ctx).run()
    fundamentals = result.tables["fundamentals"]

    assert fundamentals.height == 2
    assert set(fundamentals["race_id"].to_list()) == {"US-SEN-GA-2026", "US-HOUSE-CA45-2026"}
    assert fundamentals["source_id"].unique().to_list() == ["fred_unrate_test"]
    assert fundamentals["economic_series_id"].unique().to_list() == ["UNRATE"]
    assert fundamentals["economic_observation_date"].unique().to_list() == [date(2026, 3, 1)]
    assert fundamentals["economic_index"].to_list() == pytest.approx([-0.1, -0.1])


def test_president_state_panel_parsers_derive_curated_tables(tmp_path: Path) -> None:
    panel_path = ROOT / "fixtures" / "president_state_panel_sample.csv"
    parser_sources = [
        ("panel_races", "races", "president-state-panel-races-v1", {}),
        ("panel_options", "options", "president-state-panel-options-v1", {}),
        ("panel_results", "results", "president-state-panel-results-v1", {}),
        (
            "panel_fundamentals",
            "fundamentals",
            "president-state-panel-fundamentals-v1",
            {"as_of_offsets_days": [30, 7]},
        ),
        (
            "panel_polls",
            "polls",
            "president-state-panel-polls-v1",
            {"as_of_offsets_days": [30, 7], "poll_duration_days": 2},
        ),
    ]
    sources = [
        SourceDefinition(
            id=source_id,
            table=table,
            type="fixture",
            path=panel_path,
            parser_version=parser_version,
            license="Synthetic compact presidential-state panel parser fixture.",
            url="file://fixtures/president_state_panel_sample.csv",
            parser_args=parser_args,
        )
        for source_id, table, parser_version, parser_args in parser_sources
    ]
    ctx = context(tmp_path)
    sync_result = SyncRunner(ctx, registry=SourceRegistry(sources)).run()
    result = CuratedDataBuilder(ctx).run()

    assert sync_result.manifest["content_hash"].n_unique() == 1
    assert set(sync_result.manifest["parser_version"].to_list()) == {
        "president-state-panel-races-v1",
        "president-state-panel-options-v1",
        "president-state-panel-results-v1",
        "president-state-panel-fundamentals-v1",
        "president-state-panel-polls-v1",
    }

    races = result.tables["races"]
    assert races.select(["cycle", "state", "race_id", "seats"]).to_dicts() == [
        {"cycle": 2024, "state": "MI", "race_id": "US-PRES-MI-2024", "seats": 15},
        {"cycle": 2020, "state": "WI", "race_id": "US-PRES-WI-2020", "seats": 10},
    ]

    options = result.tables["options"]
    assert {"cycle", "state", "race_id", "option_id"}.issubset(options.columns)
    assert options.filter(pl.col("race_id") == "US-PRES-WI-2020").select(
        ["option_id", "party", "incumbent", "previous_vote_share"]
    ).to_dicts() == [
        {
            "option_id": "US-PRES-WI-2020-D",
            "party": "DEM",
            "incumbent": False,
            "previous_vote_share": 0.4645,
        },
        {
            "option_id": "US-PRES-WI-2020-R",
            "party": "REP",
            "incumbent": True,
            "previous_vote_share": 0.4722,
        },
    ]

    results = result.tables["results"]
    assert results.filter(pl.col("winner")).select(["race_id", "option_id"]).to_dicts() == [
        {"race_id": "US-PRES-MI-2024", "option_id": "US-PRES-MI-2024-D"},
        {"race_id": "US-PRES-WI-2020", "option_id": "US-PRES-WI-2020-D"},
    ]
    assert {"cycle", "state", "race_id", "option_id", "vote_share"}.issubset(results.columns)

    fundamentals = result.tables["fundamentals"]
    assert fundamentals.height == 4
    assert fundamentals.select(["cycle", "state", "race_id", "as_of_offset_days"]).to_dicts() == [
        {
            "cycle": 2024,
            "state": "MI",
            "race_id": "US-PRES-MI-2024",
            "as_of_offset_days": 30,
        },
        {
            "cycle": 2024,
            "state": "MI",
            "race_id": "US-PRES-MI-2024",
            "as_of_offset_days": 7,
        },
        {
            "cycle": 2020,
            "state": "WI",
            "race_id": "US-PRES-WI-2020",
            "as_of_offset_days": 30,
        },
        {
            "cycle": 2020,
            "state": "WI",
            "race_id": "US-PRES-WI-2020",
            "as_of_offset_days": 7,
        },
    ]

    polls = result.tables["polls"]
    assert polls.height == 8
    assert polls.filter(pl.col("poll_id") == "panel-US-PRES-WI-2020-t30-D").select(
        ["cycle", "state", "race_id", "option_id", "end_date", "pct"]
    ).to_dicts() == [
        {
            "cycle": 2020,
            "state": "WI",
            "race_id": "US-PRES-WI-2020",
            "option_id": "US-PRES-WI-2020-D",
            "end_date": date(2020, 10, 4),
            "pct": 50.0,
        }
    ]


def test_senate_and_house_panel_parsers_derive_curated_tables(tmp_path: Path) -> None:
    senate_path = tmp_path / "senate_panel.csv"
    senate_path.write_text(
        "\n".join(
            [
                (
                    "cycle,state,election_date,dem_name,rep_name,dem_incumbent,rep_incumbent,"
                    "dem_previous_vote_share,rep_previous_vote_share,dem_fundraising_usd,"
                    "rep_fundraising_usd,dem_vote_share,rep_vote_share,turnout,"
                    "partisan_lean,incumbency_advantage,economic_index,"
                    "demographic_turnout_index,historical_turnout_rate,registered_voters,"
                    "pollster,poll_sample_size,poll_population,poll_sponsor_class,"
                    "poll_methodology,dem_poll_pct,rep_poll_pct"
                ),
                (
                    "2024,AZ,2024-11-05,Dem Senate,Rep Senate,false,true,0.49,0.51,"
                    "9000000,8000000,0.505,0.495,3200000,-1.2,3.5,-0.1,0.4,0.63,"
                    "5200000,Panel Poll,850,lv,nonpartisan,mixed,50.4,49.6"
                ),
            ]
        ),
        encoding="utf-8",
    )
    house_path = tmp_path / "house_panel.csv"
    house_path.write_text(
        "\n".join(
            [
                (
                    "cycle,state,district,election_date,competitive,dem_name,rep_name,"
                    "dem_incumbent,rep_incumbent,dem_previous_vote_share,"
                    "rep_previous_vote_share,dem_fundraising_usd,rep_fundraising_usd,"
                    "dem_vote_share,rep_vote_share,turnout,partisan_lean,"
                    "incumbency_advantage,economic_index,demographic_turnout_index,"
                    "historical_turnout_rate,registered_voters,pollster,poll_sample_size,"
                    "poll_population,poll_sponsor_class,poll_methodology,dem_poll_pct,"
                    "rep_poll_pct"
                ),
                (
                    "2024,CA,CA-45,2024-11-05,true,Dem House,Rep House,false,true,"
                    "0.48,0.52,4500000,4700000,0.51,0.49,410000,1.0,4.0,-0.1,"
                    "0.2,0.55,760000,House Panel Poll,700,lv,nonpartisan,online,50.8,49.2"
                ),
                (
                    "2024,CA,CA-12,2024-11-05,false,Safe Dem,Safe Rep,true,false,"
                    "0.70,0.30,900000,200000,0.72,0.28,390000,22.0,4.0,-0.1,"
                    "0.3,0.55,740000,House Panel Poll,700,lv,nonpartisan,online,70.0,30.0"
                ),
            ]
        ),
        encoding="utf-8",
    )
    parser_sources = [
        ("senate_races", "races", "senate-state-panel-races-v1", senate_path, {}),
        ("senate_options", "options", "senate-state-panel-options-v1", senate_path, {}),
        ("senate_results", "results", "senate-state-panel-results-v1", senate_path, {}),
        (
            "senate_fundamentals",
            "fundamentals",
            "senate-state-panel-fundamentals-v1",
            senate_path,
            {"as_of_offsets_days": [30, 7]},
        ),
        (
            "senate_polls",
            "polls",
            "senate-state-panel-polls-v1",
            senate_path,
            {"as_of_offsets_days": [30, 7], "poll_duration_days": 2},
        ),
        ("house_races", "races", "house-district-panel-races-v1", house_path, {}),
        ("house_options", "options", "house-district-panel-options-v1", house_path, {}),
        ("house_results", "results", "house-district-panel-results-v1", house_path, {}),
        (
            "house_fundamentals",
            "fundamentals",
            "house-district-panel-fundamentals-v1",
            house_path,
            {"as_of_offsets_days": [30, 7]},
        ),
        (
            "house_polls",
            "polls",
            "house-district-panel-polls-v1",
            house_path,
            {"as_of_offsets_days": [30, 7], "poll_duration_days": 2},
        ),
    ]
    sources = [
        SourceDefinition(
            id=source_id,
            table=table,
            type="fixture",
            path=path,
            parser_version=parser_version,
            license="Synthetic congressional panel parser fixture.",
            url=path.resolve().as_uri(),
            parser_args=parser_args,
        )
        for source_id, table, parser_version, path, parser_args in parser_sources
    ]
    ctx = context(tmp_path)
    sync_result = SyncRunner(ctx, registry=SourceRegistry(sources)).run()
    result = CuratedDataBuilder(ctx).run()

    assert sync_result.failed_sources == 0
    races = result.tables["races"]
    actual_races = sorted(
        races.select(["race_id", "office_type", "control_body", "seats"]).to_dicts(),
        key=lambda row: row["race_id"],
    )
    assert actual_races == sorted(
        [
            {
                "race_id": "US-SEN-AZ-2024",
                "office_type": "senate",
                "control_body": "senate",
                "seats": 1,
            },
            {
                "race_id": "US-HOUSE-CA-45-2024",
                "office_type": "house",
                "control_body": "house",
                "seats": 1,
            },
            {
                "race_id": "US-HOUSE-CA-12-2024",
                "office_type": "house",
                "control_body": "house",
                "seats": 1,
            },
        ],
        key=lambda row: row["race_id"],
    )

    options = result.tables["options"]
    assert options.filter(pl.col("race_id") == "US-SEN-AZ-2024").select(
        ["option_id", "party", "incumbent", "previous_vote_share"]
    ).to_dicts() == [
        {
            "option_id": "US-SEN-AZ-2024-D",
            "party": "DEM",
            "incumbent": False,
            "previous_vote_share": 0.49,
        },
        {
            "option_id": "US-SEN-AZ-2024-R",
            "party": "REP",
            "incumbent": True,
            "previous_vote_share": 0.51,
        },
    ]

    results = result.tables["results"]
    actual_winners = sorted(
        results.filter(pl.col("winner")).select(["race_id", "party"]).to_dicts(),
        key=lambda row: row["race_id"],
    )
    assert actual_winners == sorted(
        [
            {"race_id": "US-SEN-AZ-2024", "party": "DEM"},
            {"race_id": "US-HOUSE-CA-45-2024", "party": "DEM"},
            {"race_id": "US-HOUSE-CA-12-2024", "party": "DEM"},
        ],
        key=lambda row: row["race_id"],
    )

    fundamentals = result.tables["fundamentals"]
    assert fundamentals.filter(pl.col("race_id") == "US-SEN-AZ-2024").height == 2
    assert fundamentals.filter(pl.col("race_id") == "US-HOUSE-CA-45-2024").height == 2

    polls = result.tables["polls"]
    assert polls.filter(pl.col("race_id") == "US-SEN-AZ-2024").height == 4
    assert polls.filter(pl.col("race_id") == "US-HOUSE-CA-45-2024").height == 4
    assert polls.filter(pl.col("race_id") == "US-HOUSE-CA-12-2024").is_empty()
    assert set(polls.select("methodology").unique().to_series().to_list()) == {
        "online",
        "mixed",
    }


def test_president_state_panel_parser_requires_declared_columns(tmp_path: Path) -> None:
    source_path = tmp_path / "bad_president_panel.csv"
    source_path.write_text(
        "cycle,state,election_date,pollster,poll_sample_size,poll_population,"
        "poll_sponsor_class,poll_methodology,dem_poll_pct\n"
        "2024,WI,2024-11-05,Panel Research,900,lv,nonpartisan,mixed,50.0\n",
        encoding="utf-8",
    )
    ctx = context(tmp_path)
    source = SourceDefinition(
        id="panel_polls_missing_column",
        table="polls",
        type="fixture",
        path=source_path,
        parser_version="president-state-panel-polls-v1",
        license="Strict parser test fixture.",
        url=source_path.resolve().as_uri(),
        parser_args={"as_of_offsets_days": [7]},
    )
    SyncRunner(ctx, registry=SourceRegistry([source])).run()

    try:
        CuratedDataBuilder(ctx).run()
    except ValueError as exc:
        assert "president-state-panel-polls-v1 missing columns" in str(exc)
        assert "rep_poll_pct" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("president state panel parser should fail on missing columns")


def test_538_parser_args_are_required(tmp_path: Path) -> None:
    csv_payload = (
        "cycle,state,pollster,poll_id,question_id,start_date,end_date,sample_size,population,"
        "methodology,internal,partisan,stage,answer,candidate_party,pct\n"
        "2020,Wisconsin,Acme Poll,1001,77,10/01/20,10/03/20,800,LV,Live Phone,,,General,"
        "Smith,DEM,49.5\n"
    )
    source_path = tmp_path / "538_president.csv"
    source_path.write_text(csv_payload, encoding="utf-8")
    ctx = context(tmp_path)
    extra = SourceDefinition(
        id="fivethirtyeight_president_polls_missing_args",
        table="polls",
        type="http_csv",
        path=source_path,
        parser_version="fivethirtyeight-president-polls-v1",
        license="Test fixture for strict parser args.",
        url=source_path.resolve().as_uri(),
        auth_mode="public",
    )
    SyncRunner(ctx, registry=SourceRegistry([extra])).run()
    try:
        CuratedDataBuilder(ctx).run()
    except ValueError as exc:
        assert "parser_args missing required keys" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("strict 538 parser args should fail without race identity")


def test_http_sync_retries_then_fails_on_unreachable_url(tmp_path: Path, monkeypatch) -> None:
    from election_outcomes.ingest import sync as sync_module

    ctx = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    bogus = SourceDefinition(
        id="unreachable_csv",
        table="polls",
        type="http_csv",
        path=None,
        parser_version="fivethirtyeight-president-polls-v1",
        license="Test fixture for retry path.",
        url="http://127.0.0.1:1/missing.csv",
        auth_mode="public",
    )
    monkeypatch.setattr(sync_module, "HTTP_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    fixture_registry = SourceRegistry.from_context(ctx)
    registry = SourceRegistry([*fixture_registry.sources, bogus])
    result = SyncRunner(ctx, registry=registry).run()

    failed_rows = result.manifest.filter(pl.col("source_id") == "unreachable_csv")
    assert failed_rows.height == 1
    assert failed_rows["status"].to_list() == ["failed"]
    assert failed_rows["auth_mode"].to_list() == ["public"]
    assert "HTTP fetch failed" in failed_rows["error"].to_list()[0]


def test_fundamentals_ridge_fits_when_training_rows_meet_threshold(tmp_path: Path) -> None:
    _ctx, _sync, bundle = build_bundle(tmp_path)
    model_config = {"fundamentals": {"min_training_rows": 1, "ridge_alpha": 0.5}}
    model = FundamentalsModel(model_config).fit(bundle)
    estimates = model.run(bundle)

    assert model.fit_status.startswith("standardized_ridge_fit")
    assert model.training_rows >= 1
    assert model.feature_stds
    assert not estimates.is_empty()
    assert estimates["explanation"].str.contains("standardized_ridge_fit").all()


def test_residual_covariance_requires_multiple_observations() -> None:
    one_observation = pl.DataFrame(
        {
            "cycle": [2024, 2024],
            "as_of": ["2024-11-04", "2024-11-04"],
            "geography": ["WI", "WI"],
            "predicted_vote_share": [0.51, 0.49],
            "actual_vote_share": [0.495, 0.501],
        }
    )
    assert BacktestRunner._residual_covariance(one_observation).is_empty()

    two_observations = pl.DataFrame(
        {
            "cycle": [2024, 2024, 2024, 2024],
            "as_of": ["2024-10-29", "2024-10-29", "2024-11-04", "2024-11-04"],
            "geography": ["WI", "WI", "WI", "WI"],
            "predicted_vote_share": [0.51, 0.49, 0.50, 0.50],
            "actual_vote_share": [0.495, 0.501, 0.495, 0.501],
        }
    )
    covariance = BacktestRunner._residual_covariance(two_observations)
    assert covariance.height == 1
    assert covariance["sample_size"].to_list() == [2]
    assert covariance["covariance_method"].to_list() == ["structured_shrinkage_by_geography"]


def test_score_predictions_handles_empty_and_real_rows(tmp_path: Path) -> None:
    _ctx, _sync, _bundle = build_bundle(tmp_path)
    empty_scores = score_predictions(pl.DataFrame())
    real_scores = BacktestRunner(_ctx).evaluate(inference_engine="kalman")["metrics"]["ensemble"]

    assert str(empty_scores["brier"]) == "nan"
    assert 0.0 <= real_scores["brier"] <= 1.0
    assert 0.0 <= real_scores["expected_calibration_error"] <= 1.0
    assert real_scores["expected_calibration_error_bins"] >= 1


def test_performance_benchmark_and_python_kernel_fallback(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_benchmark(
        as_of="2026-05-08", run_id="perf", draws=100, repeats=1
    )
    assert payload["forecast_draw_rows"] == payload["draws"] * 10
    assert payload["rows_per_second"] > 0
    assert (ctx.artifacts_dir / "benchmarks" / "perf" / "performance_benchmark.json").exists()

    arrays = simulate_binary_draw_arrays(
        first_shares=pl.Series([0.52]).to_numpy(),
        turnout_bases=pl.Series([1000.0]).to_numpy(),
        national_errors=pl.Series([0.0, 0.01]).to_numpy(),
        local_errors=pl.DataFrame({"a": [0.0, -0.02]}).to_numpy().T,
        use_numba=False,
    )
    assert len(arrays[0]) == 4
    assert arrays[5][0] == 0.52
