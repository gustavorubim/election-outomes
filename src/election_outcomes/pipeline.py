from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.features import FeatureBuilder, FeatureBundle
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
from election_outcomes.reports import DiagnosticsReport, MethodologySnapshot, PlotGenerator
from election_outcomes.scoring import BacktestRunner, ResultComparator, RewardEvaluator
from election_outcomes.storage.io import write_json, write_parquet, write_text


class ForecastPipeline:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def sync(self) -> pl.DataFrame:
        return SyncRunner(self.context).run().manifest

    def build_features(self) -> FeatureBundle:
        CuratedDataBuilder(self.context).run()
        return FeatureBuilder(self.context).run()

    def run_forecast(self, as_of: str, run_id: str | None = None) -> Path:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        full_bundle = self.build_features()
        bundle = self._active_bundle(full_bundle, as_of)
        model_config = self.context.read_yaml("model.yaml")
        source_manifest = pl.read_parquet(self.context.curated_dir / "source_manifest.parquet")
        component_estimates = [
            PollingModel(model_config, as_of=as_of).run(bundle),
            FundamentalsModel(model_config).fit(full_bundle).run(bundle),
            MarketModel(model_config).run(bundle),
            PublicSignalModel(
                trusted=bool(
                    model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(bundle),
        ]
        ensemble = EnsembleModel(model_config).run(bundle, component_estimates)
        outputs = SimulationEngine(model_config).run(bundle, ensemble)
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
        write_json(outputs.performance, out_dir / "performance.json")
        write_parquet(
            source_manifest.with_columns(pl.lit("forecast_artifacts").alias("downstream_usage")),
            out_dir / "source_manifest.parquet",
        )
        backtest_payload = BacktestRunner(self.context).evaluate()
        plot_generator = PlotGenerator()
        plot_manifest = plot_generator.render_all(
            out_dir,
            race_catalog,
            race_forecasts,
            outputs.draws,
            outputs.control_forecasts,
            outputs.ecosystem_forecasts,
            full_bundle.backtest_predictions,
            backtest_payload,
        )
        plot_generator.write_manifest(plot_manifest, out_dir)
        methodology = MethodologySnapshot().render(
            run_id, as_of, model_config, source_manifest.height
        )
        write_text(methodology, out_dir / "methodology_snapshot.md")
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
        )
        write_text(diagnostics, out_dir / "diagnostics.html")
        return out_dir

    def run_backtest(self, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.sync()
        self.build_features()
        return BacktestRunner(self.context).run(run_id)

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

    def rebuild_report(self, run_id: str) -> Path:
        out_dir = self.context.artifacts_dir / "runs" / run_id
        model_config = self.context.read_yaml("model.yaml")
        race_catalog = pl.read_parquet(out_dir / "race_catalog.parquet")
        race_forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
        source_manifest = pl.read_parquet(out_dir / "source_manifest.parquet")
        forecast_draws = pl.read_parquet(out_dir / "forecast_draws.parquet")
        control_forecasts = pl.read_parquet(out_dir / "control_forecasts.parquet")
        ecosystem_forecasts = pl.read_parquet(out_dir / "ecosystem_forecasts.parquet")
        backtest_payload = BacktestRunner(self.context).evaluate()
        backtest_predictions = self._read_optional_curated("backtest_predictions")
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
        )
        write_text(diagnostics, out_dir / "diagnostics.html")
        methodology = MethodologySnapshot().render(
            run_id, "existing", model_config, source_manifest.height
        )
        write_text(methodology, out_dir / "methodology_snapshot.md")
        return out_dir

    def _read_optional_curated(self, name: str) -> pl.DataFrame:
        path = self.context.curated_dir / f"{name}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    @staticmethod
    def _active_bundle(bundle: FeatureBundle, as_of: str) -> FeatureBundle:
        cutoff = date.fromisoformat(as_of)
        active_catalog = bundle.race_catalog.filter(pl.col("election_date") >= cutoff)
        active_ids = active_catalog["race_id"].to_list()

        def by_race(frame: pl.DataFrame) -> pl.DataFrame:
            return (
                frame.filter(pl.col("race_id").is_in(active_ids))
                if "race_id" in frame.columns
                else frame
            )

        def by_race_and_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
            filtered = by_race(frame)
            if column not in filtered.columns:
                return filtered
            dates = pl.col(column)
            if filtered.schema[column] != pl.Date:
                dates = (
                    pl.col(column)
                    .cast(pl.Utf8)
                    .str.slice(0, 10)
                    .str.strptime(pl.Date, strict=False)
                )
            return filtered.filter(dates <= cutoff)

        return replace(
            bundle,
            races=by_race(bundle.races),
            options=by_race(bundle.options),
            polls=by_race_and_date(bundle.polls, "end_date"),
            markets=by_race_and_date(bundle.markets, "observed_at"),
            public_signals=by_race_and_date(bundle.public_signals, "observed_at"),
            fundamentals=by_race(bundle.fundamentals),
            results=by_race(bundle.results),
            race_catalog=active_catalog,
        )

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
                "source_manifest.parquet",
                "methodology_snapshot.md",
                "plot_manifest.json",
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
