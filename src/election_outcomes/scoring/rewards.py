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
    ) -> dict[str, Any]:
        provenance_share = self._provenance_share(race_forecasts, source_manifest)
        tier_c_ok = self._tier_c_withheld(race_forecasts, race_catalog)
        ensemble_metrics = backtest_payload.get("metrics", {}).get("ensemble", {})
        ablations = backtest_payload.get("ablations", {})
        reproducibility = self._reproducibility_status(artifact_dir)
        trustworthy_backtest = self._trustworthy_backtest(backtest_payload)
        sync_breakdown = self._sync_breakdown(source_manifest)
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
                    "detail": "Historical backtest reports calibration and scoring metrics.",
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
                    "passed": trustworthy_backtest
                    and all(
                        item.get("beats_or_matches_baseline", False)
                        for key, item in ablations.items()
                        if key in {"polling", "fundamentals", "markets", "ensemble"}
                    ),
                    "metric": ablations,
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
