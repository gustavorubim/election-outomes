from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


class RewardEvaluator:
    """Build machine-readable reward checks for a forecast run."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self.model_config = model_config

    def evaluate(
        self,
        run_id: str,
        artifact_dir: Path,
        race_forecasts: pl.DataFrame,
        race_catalog: pl.DataFrame,
        source_manifest: pl.DataFrame,
        backtest_payload: dict[str, Any],
        plot_manifest: dict[str, list[dict[str, str]]] | None = None,
        performance: dict[str, Any] | None = None,
        posterior_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provenance_share = self._provenance_share(race_forecasts, source_manifest)
        tier_c_ok = self._tier_c_withheld(race_forecasts, race_catalog)
        ensemble_metrics = backtest_payload.get("metrics", {}).get("ensemble", {})
        ablations = backtest_payload.get("ablations", {})
        reproducibility = self._reproducibility_status(artifact_dir)
        trustworthy_backtest = self._trustworthy_backtest(backtest_payload)
        sync_breakdown = self._sync_breakdown(source_manifest)
        component_admission = self._component_admission_metric(ablations)
        return {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "rewards": {
                "R0_build": {
                    "passed": None,
                    "metric": None,
                    "detail": "External gate: uv sync, ruff, format check, pytest coverage.",
                },
                "R1_reproducibility": {
                    "passed": reproducibility["cross_run_verified"],
                    "metric": reproducibility,
                    "detail": (
                        "Stable artifact fingerprint is generated on every run. This reward "
                        "passes only after rerunning the same run id with unchanged inputs and "
                        "matching the previous fingerprint."
                    ),
                },
                "R2_provenance": {
                    "passed": provenance_share >= 1.0,
                    "metric": provenance_share,
                    "detail": "Forecast rows trace to non-empty source hashes.",
                },
                "R3_sync_integrity": {
                    "passed": (
                        source_manifest.filter(pl.col("status") == "failed").is_empty()
                        and sync_breakdown["failed_auth"] == 0
                    ),
                    "metric": sync_breakdown,
                    "detail": (
                        "All configured sources are recorded with status, hash, and auth mode."
                    ),
                },
                "R4_calibration": {
                    "passed": bool(ensemble_metrics),
                    "metric": ensemble_metrics,
                    "detail": (
                        "Historical backtest reports scoring metrics and a probability "
                        "calibration transform."
                    ),
                },
                "R5_baseline_competition": {
                    "passed": trustworthy_backtest
                    and ablations.get("ensemble", {}).get("beats_or_matches_baseline", False),
                    "metric": ablations.get("ensemble", {}),
                    "detail": (
                        "Trusted ensemble must beat or match baseline Brier score on a "
                        "rolling-origin backtest with enough rows."
                    ),
                },
                "R6_component_admission": {
                    "passed": trustworthy_backtest and component_admission["passed"],
                    "metric": component_admission,
                    "detail": "Trusted components are backed by ablation evidence.",
                },
                "R7_sparse_honesty": {
                    "passed": tier_c_ok,
                    "metric": int(tier_c_ok),
                    "detail": "Tier C races withhold trusted probabilities.",
                },
                "R8_uncertainty_quality": {
                    "passed": trustworthy_backtest and self._coverage_ok(ensemble_metrics),
                    "metric": ensemble_metrics.get("interval_90_coverage"),
                    "detail": (
                        "Historical intervals report empirical coverage near nominal on a "
                        "rolling-origin backtest with enough rows."
                    ),
                },
                "R9_public_signal_discipline": {
                    "passed": not self.model_config.get("trusted_components", {}).get(
                        "public_signals", False
                    ),
                    "metric": ablations.get("public_signals", {}),
                    "detail": (
                        "Public signals remain experimental until ablation and leakage checks pass."
                    ),
                },
                "R10_explainability": {
                    "passed": self._has_explainability(race_forecasts),
                    "metric": race_forecasts.height,
                    "detail": "Forecast rows include tier reason and quality flags.",
                },
                "R11_plot_contract": {
                    "passed": self._plot_contract_ok(artifact_dir, plot_manifest or {}),
                    "metric": {
                        "calibration": len((plot_manifest or {}).get("calibration", [])),
                        "projection": len((plot_manifest or {}).get("projection", [])),
                    },
                    "detail": "Forecast runs emit calibration and projection plot artifacts.",
                },
                "R12_performance_contract": {
                    "passed": self._performance_contract_ok(performance or {}),
                    "metric": performance or {},
                    "detail": (
                        "Forecast runs record acceleration engine, parallel mode, and draw count."
                    ),
                },
                "R13_posterior_quality": {
                    "passed": self._posterior_quality_ok(posterior_diagnostics),
                    "metric": posterior_diagnostics or {},
                    "detail": (
                        "Bayesian runs must emit posterior diagnostics with enough draws, no "
                        "divergences, and valid R-hat/ESS when those MCMC metrics are available."
                    ),
                },
                "R14_calibrated_publication": {
                    "passed": self._calibrated_publication_ok(
                        race_forecasts, artifact_dir, ensemble_metrics
                    ),
                    "metric": self._calibrated_publication_metric(
                        race_forecasts, artifact_dir, ensemble_metrics
                    ),
                    "detail": (
                        "Published probabilities must either use a persisted recalibration map "
                        "or show acceptable rolling-origin calibration without a map."
                    ),
                },
                "R15_daily_update_quality": {
                    "passed": self._daily_update_quality_ok(artifact_dir),
                    "metric": self._latest_daily_update(artifact_dir),
                    "detail": (
                        "Daily update, when present, must pass its strategy-specific quality "
                        "gate and avoid full-refit triggers."
                    ),
                },
            },
        }

    @staticmethod
    def _artifacts_exist(artifact_dir: Path) -> bool:
        required = {
            "race_catalog.parquet",
            "race_forecasts.parquet",
            "forecast_draws.parquet",
            "control_forecasts.parquet",
            "ecosystem_forecasts.parquet",
            "source_manifest.parquet",
            "diagnostics.html",
            "methodology_snapshot.md",
            "model_card.md",
            "silver_benchmark.html",
            "silver_benchmark.json",
            "plot_manifest.json",
            "performance.json",
            "reproducibility_fingerprint.json",
        }
        return all((artifact_dir / name).exists() for name in required)

    @staticmethod
    def _provenance_share(race_forecasts: pl.DataFrame, source_manifest: pl.DataFrame) -> float:
        if race_forecasts.is_empty() or source_manifest.is_empty():
            return 0.0
        required = {"model_config_hash", "source_manifest_hash"}
        if not required.issubset(set(race_forecasts.columns)):
            return 0.0
        forecast_rows = race_forecasts.filter(
            pl.col("model_config_hash").is_not_null()
            & (pl.col("model_config_hash") != "")
            & pl.col("source_manifest_hash").is_not_null()
            & (pl.col("source_manifest_hash") != "")
        ).height
        manifest_rows = source_manifest.filter(
            pl.col("content_hash").is_not_null() & (pl.col("content_hash") != "")
        ).height
        return min(forecast_rows / race_forecasts.height, manifest_rows / source_manifest.height)

    @staticmethod
    def _tier_c_withheld(race_forecasts: pl.DataFrame, race_catalog: pl.DataFrame) -> bool:
        tier_c_races = race_catalog.filter(pl.col("tier") == "C")["race_id"].to_list()
        if not tier_c_races:
            return True
        tier_c_forecasts = race_forecasts.filter(pl.col("race_id").is_in(tier_c_races))
        return tier_c_forecasts["winner_probability"].null_count() == tier_c_forecasts.height

    def _coverage_ok(self, metrics: dict[str, Any]) -> bool:
        coverage = metrics.get("interval_90_coverage")
        if coverage is None:
            return False
        tolerance = float(
            self.model_config.get("reward_thresholds", {}).get("interval_coverage_tolerance", 0.12)
        )
        return abs(float(coverage) - 0.90) <= tolerance

    @staticmethod
    def _has_explainability(race_forecasts: pl.DataFrame) -> bool:
        required = {
            "tier_reason",
            "data_quality_flags",
            "top_drivers",
            "component_contributions",
            "uncertainty_explanation",
        }
        return required.issubset(set(race_forecasts.columns)) and race_forecasts.height > 0

    def _component_admission_metric(self, ablations: dict[str, Any]) -> dict[str, Any]:
        trusted_components = {
            str(key): bool(value)
            for key, value in dict(self.model_config.get("trusted_components", {})).items()
        }
        component_keys = {"polling", "fundamentals", "markets", "public_signals"}
        trusted_results = {
            component: ablations.get(component, {})
            for component, trusted in trusted_components.items()
            if trusted and component in component_keys
        }
        failed_trusted = sorted(
            component
            for component, result in trusted_results.items()
            if not bool(result.get("beats_or_matches_baseline", False))
        )
        ensemble_result = dict(ablations.get("ensemble", {}))
        ensemble_passed = bool(ensemble_result.get("beats_or_matches_baseline", False))
        return {
            "passed": ensemble_passed and not failed_trusted,
            "trusted_components": trusted_components,
            "trusted_component_results": trusted_results,
            "failed_trusted_components": failed_trusted,
            "untrusted_components": sorted(
                component
                for component in component_keys
                if component in trusted_components and not trusted_components[component]
            ),
            "ensemble": ensemble_result,
            "all_ablations": ablations,
        }

    @staticmethod
    def _plot_contract_ok(
        artifact_dir: Path, plot_manifest: dict[str, list[dict[str, str]]]
    ) -> bool:
        if not plot_manifest.get("calibration") or not plot_manifest.get("projection"):
            return False
        paths = [
            artifact_dir / entry["path"] for entries in plot_manifest.values() for entry in entries
        ]
        return bool(paths) and all(path.exists() and path.stat().st_size > 0 for path in paths)

    @staticmethod
    def _performance_contract_ok(performance: dict[str, Any]) -> bool:
        required = {"requested_engine", "engine", "parallel", "numba_available", "simulation_count"}
        if not required.issubset(performance):
            return False
        if performance["requested_engine"] == "numba":
            if performance["numba_available"]:
                return performance["engine"] == "numba"
            return performance["engine"] == "python"
        return performance["engine"] in {"numba", "python"}

    @staticmethod
    def _posterior_quality_ok(diagnostics: dict[str, Any] | None) -> bool | None:
        if not diagnostics:
            return None
        if str(diagnostics.get("fallback_used") or "").strip():
            return False
        if int(diagnostics.get("divergences") or 0) != 0:
            return False
        if int(diagnostics.get("draw_count") or 0) < 100:
            return False
        if int(diagnostics.get("race_option_count") or 0) <= 0:
            return False
        r_hat = diagnostics.get("r_hat_max")
        ess = diagnostics.get("ess_min")
        if r_hat is not None and float(r_hat) > 1.05:
            return False
        if ess is not None and float(ess) < 400:
            return False
        return True

    def _calibrated_publication_ok(
        self,
        race_forecasts: pl.DataFrame,
        artifact_dir: Path,
        ensemble_metrics: dict[str, Any],
    ) -> bool | None:
        if race_forecasts.is_empty() or "winner_probability" not in race_forecasts.columns:
            return None
        published = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if published.is_empty():
            return None
        map_present = (artifact_dir / "recalibration_map.parquet").exists()
        statuses = self._forecast_calibration_statuses(race_forecasts)
        if map_present:
            return bool(statuses) and "not_configured" not in statuses
        if not ensemble_metrics:
            return None
        return self._already_calibrated(ensemble_metrics)

    def _calibrated_publication_metric(
        self,
        race_forecasts: pl.DataFrame,
        artifact_dir: Path,
        ensemble_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        statuses = self._forecast_calibration_statuses(race_forecasts)
        non_null = (
            race_forecasts.filter(pl.col("winner_probability").is_not_null())
            if "winner_probability" in race_forecasts.columns
            else pl.DataFrame()
        )
        max_delta = None
        if (
            not non_null.is_empty()
            and "raw_winner_probability" in non_null.columns
            and "winner_probability" in non_null.columns
        ):
            max_delta = float(
                non_null.select(
                    (pl.col("winner_probability") - pl.col("raw_winner_probability")).abs().max()
                ).item()
                or 0.0
            )
        map_path = artifact_dir / "recalibration_map.parquet"
        map_rows = None
        map_status = None
        if map_path.exists():
            frame = pl.read_parquet(map_path)
            map_rows = frame.height
            if not frame.is_empty() and "status" in frame.columns:
                map_status = str(frame["status"][0])
        return {
            "map_present": map_path.exists(),
            "map_rows": map_rows,
            "map_status": map_status,
            "forecast_calibration_statuses": statuses,
            "max_probability_delta": max_delta,
            "rolling_origin_calibration": {
                "intercept": ensemble_metrics.get("calibration_intercept"),
                "slope": ensemble_metrics.get("calibration_slope"),
                "expected_calibration_error": ensemble_metrics.get("expected_calibration_error"),
            },
            "already_calibrated_without_map": self._already_calibrated(ensemble_metrics)
            if ensemble_metrics
            else None,
        }

    @staticmethod
    def _forecast_calibration_statuses(race_forecasts: pl.DataFrame) -> list[str]:
        if (
            race_forecasts.is_empty()
            or "probability_calibration_status" not in race_forecasts.columns
        ):
            return []
        return sorted(
            str(value)
            for value in race_forecasts["probability_calibration_status"]
            .drop_nulls()
            .unique()
            .to_list()
        )

    def _already_calibrated(self, metrics: dict[str, Any]) -> bool:
        if not metrics:
            return False
        intercept = metrics.get("calibration_intercept")
        slope = metrics.get("calibration_slope")
        ece = metrics.get("expected_calibration_error")
        if intercept is None or slope is None or ece is None:
            return False
        thresholds = dict(self.model_config.get("reward_thresholds", {}))
        max_abs_intercept = float(thresholds.get("calibration_max_abs_intercept", 0.05))
        max_slope_delta = float(thresholds.get("calibration_max_slope_delta", 0.10))
        max_ece = float(thresholds.get("calibration_max_ece", 0.06))
        return (
            abs(float(intercept)) <= max_abs_intercept
            and abs(float(slope) - 1.0) <= max_slope_delta
            and float(ece) <= max_ece
        )

    @staticmethod
    def _daily_update_quality_ok(artifact_dir: Path) -> bool | None:
        latest = RewardEvaluator._latest_daily_update(artifact_dir)
        if not latest:
            return None
        return bool(latest.get("quality_passed")) and not bool(latest.get("needs_full_refit"))

    @staticmethod
    def _latest_daily_update(artifact_dir: Path) -> dict[str, Any]:
        path = artifact_dir / "latest_daily_update.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _reproducibility_status(artifact_dir: Path) -> dict[str, Any]:
        path = artifact_dir / "reproducibility_fingerprint.json"
        if not path.exists():
            return {
                "fingerprint_exists": False,
                "cross_run_verified": False,
                "combined_hash": None,
            }
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            "fingerprint_exists": True,
            "cross_run_verified": bool(payload.get("cross_run_verified")),
            "compared_to_previous": bool(payload.get("compared_to_previous")),
            "combined_hash": payload.get("combined_hash"),
        }

    @staticmethod
    def _trustworthy_backtest(backtest_payload: dict[str, Any]) -> bool:
        return bool(backtest_payload.get("rolling_origin_executed")) and not bool(
            backtest_payload.get("sample_size_too_small")
        )

    @staticmethod
    def _sync_breakdown(source_manifest: pl.DataFrame) -> dict[str, int]:
        breakdown = {
            "total": int(source_manifest.height),
            "fetched": 0,
            "unchanged": 0,
            "failed": 0,
            "failed_auth": 0,
            "by_auth_mode": {},
        }
        if source_manifest.is_empty():
            return breakdown
        for row in (
            source_manifest.group_by("status").agg(pl.len().alias("count")).iter_rows(named=True)
        ):
            status_value = row.get("status")
            key = str(status_value) if status_value else "unknown"
            breakdown[key] = breakdown.get(key, 0) + int(row["count"])
        if "auth_mode" in source_manifest.columns:
            failed_auth = source_manifest.filter(
                (pl.col("status") == "failed") & (pl.col("auth_mode") != "public")
            ).height
            breakdown["failed_auth"] = int(failed_auth)
            for row in (
                source_manifest.group_by("auth_mode")
                .agg(pl.len().alias("count"))
                .iter_rows(named=True)
            ):
                mode = row.get("auth_mode")
                breakdown["by_auth_mode"][str(mode) if mode else "none"] = int(row["count"])
        return breakdown
