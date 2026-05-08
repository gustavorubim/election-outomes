from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.features import FeatureBuilder
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
from election_outcomes.normalize import CuratedDataBuilder
from election_outcomes.normalize.builder import CuratedDataBuilder as _CuratedDataBuilderClass
from election_outcomes.performance import simulate_binary_draw_arrays
from election_outcomes.pipeline import ForecastPipeline
from election_outcomes.scoring import BacktestRunner, score_predictions

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

    assert first.fetched_sources == 8
    assert first.failed_sources == 0
    assert second.fetched_sources == 0
    assert second.skipped_sources == 8
    assert (ctx.raw_dir / "source_manifest.parquet").exists()
    assert first.manifest.filter(pl.col("content_hash") == "").is_empty()


def test_feature_builder_assigns_tiers_and_filters_blank_rows(tmp_path: Path) -> None:
    _ctx, _sync, bundle = build_bundle(tmp_path)
    tiers = dict(zip(bundle.race_catalog["race_id"], bundle.race_catalog["tier"], strict=True))

    assert tiers["US-SEN-AZ-2026"] == "A"
    assert tiers["US-HOUSE-CA45-2026"] == "B"
    assert tiers["MAYOR-SPRINGFIELD-2026"] == "C"
    assert bundle.race_catalog["race_id"].null_count() == 0


def test_component_models_and_ensemble_respect_admission(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")

    estimates = [
        PollingModel(model_config, as_of="2026-05-08").run(active),
        FundamentalsModel(model_config).fit(bundle).run(active),
        MarketModel(model_config).run(active),
        PublicSignalModel(trusted=False).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)

    assert not ensemble.is_empty()
    assert "MAYOR-SPRINGFIELD-2026" not in ensemble["race_id"].to_list()
    public = estimates[-1]
    assert public.filter(pl.col("race_id") == "US-SEN-AZ-2026")["admitted"].sum() == 0


def test_simulation_outputs_forecasts_control_and_ecosystem(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")
    estimates = [
        PollingModel(model_config, as_of="2026-05-08").run(active),
        FundamentalsModel(model_config).fit(bundle).run(active),
        MarketModel(model_config).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)
    outputs = SimulationEngine(model_config).run(active, ensemble)

    assert outputs.draws.height == 6000
    assert outputs.control_forecasts.height > 0
    assert {"control_threshold", "pivotal_rates"}.issubset(outputs.control_forecasts.columns)
    assert outputs.ecosystem_forecasts.height == 3
    assert outputs.ecosystem_forecasts["demographic_model_status"].unique().to_list() == [
        "placeholder_not_estimated"
    ]
    tier_c = outputs.race_forecasts.filter(pl.col("race_id") == "MAYOR-SPRINGFIELD-2026")
    assert tier_c["winner_probability"].null_count() == tier_c.height
    assert {"top_drivers", "component_contributions", "uncertainty_explanation"}.issubset(
        outputs.race_forecasts.columns
    )


def test_forecast_run_writes_required_artifacts_and_rewards(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    out_dir = ForecastPipeline(ctx).run_forecast(as_of="2026-05-08", run_id="unit")
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
        "plot_manifest.json",
        "performance.json",
        "reproducibility_fingerprint.json",
        "plots",
    }

    assert {path.name for path in out_dir.iterdir()} == required
    plot_manifest = json.loads((out_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    plot_paths = [
        out_dir / entry["path"] for entries in plot_manifest.values() for entry in entries
    ]
    assert len(plot_manifest["calibration"]) >= 3
    assert len(plot_manifest["projection"]) >= 4
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths)
    forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
    assert {"model_config_hash", "source_manifest_hash"}.issubset(forecasts.columns)
    reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    rewards = reward_card["rewards"]
    assert rewards["R0_build"]["passed"] is None
    assert rewards["R1_reproducibility"]["passed"] is False
    assert rewards["R2_provenance"]["passed"] is True
    assert rewards["R3_sync_integrity"]["passed"] is True
    assert rewards["R5_baseline_competition"]["passed"] is False
    assert rewards["R6_component_admission"]["passed"] is False
    assert rewards["R8_uncertainty_quality"]["passed"] is False
    assert rewards["R12_performance_contract"]["passed"] is True
    performance = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    assert performance["engine"] in {"numba", "python"}
    assert performance["simulation_count"] == 1000


def test_backtest_and_report_rebuild(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2026-05-08", run_id="reportable")
    payload = pipeline.run_backtest(run_id="bt")
    report_dir = pipeline.rebuild_report("reportable")

    assert payload["row_count"] == 4
    assert payload["ablations"]["ensemble"]["beats_or_matches_baseline"]
    assert (ctx.artifacts_dir / "backtests" / "bt" / "scorecard.json").exists()
    assert (
        (report_dir / "diagnostics.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    )


def test_presidential_result_comparison(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2024-10-01", run_id="pres-2024")
    payload = pipeline.compare_results(
        forecast_run_id="pres-2024",
        comparison_id="pres-actuals",
        cycle=2024,
        office_type="president",
    )
    comparison_dir = Path(payload["output_dir"])
    comparison = pl.read_parquet(comparison_dir / "result_comparison.parquet")

    assert payload["race_count"] == 1
    assert payload["row_count"] == 2
    assert payload["winner_accuracy"] in {0.0, 1.0}
    assert (comparison_dir / "result_comparison_summary.json").exists()
    assert (comparison_dir / "result_comparison.html").exists()
    assert (comparison_dir / "plots" / "vote_share_forecast_vs_actual.png").stat().st_size > 0
    assert comparison.filter(pl.col("actual_winner")).height == 1


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
    )
    registry = SourceRegistry([*fixture_registry.sources, extra])
    SyncRunner(ctx, registry=registry).run()
    result = CuratedDataBuilder(ctx).run()

    polls = result.tables["polls"]
    assert polls.filter(pl.col("poll_id").str.starts_with("538-")).height == 2
    wi_rows = polls.filter(pl.col("race_id") == "US-PRES-WI-2020")
    assert {"D", "R"}.issubset(set(wi_rows["option_id"].str.slice(-1).to_list()))
    assert wi_rows["methodology"].unique().to_list() == ["live_phone"]
    assert _CuratedDataBuilderClass is CuratedDataBuilder


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

    assert model.fit_status.startswith("ridge_fit")
    assert model.training_rows >= 1
    assert not estimates.is_empty()
    assert estimates["explanation"].str.contains("ridge_fit").all()


def test_score_predictions_handles_empty_and_real_rows(tmp_path: Path) -> None:
    _ctx, _sync, _bundle = build_bundle(tmp_path)
    empty_scores = score_predictions(pl.DataFrame())
    real_scores = BacktestRunner(_ctx).evaluate()["metrics"]["ensemble"]

    assert str(empty_scores["brier"]) == "nan"
    assert real_scores["brier"] < 0.25
    assert 0.0 <= real_scores["expected_calibration_error"] <= 1.0


def test_performance_benchmark_and_python_kernel_fallback(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_benchmark(
        as_of="2026-05-08", run_id="perf", draws=100, repeats=1
    )
    assert payload["forecast_draw_rows"] == 600
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
