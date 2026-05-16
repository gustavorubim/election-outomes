from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.storage.io import read_json, write_json, write_text


@dataclass(frozen=True)
class MethodologyReadinessAuditor:
    """Audit whether Bayesian methodology is eligible for a production default switch."""

    context: ProjectContext

    def run(
        self,
        *,
        run_id: str | None = None,
        forecast_run_id: str | None = None,
        bayes_backtest_run_id: str | None = None,
        legacy_backtest_run_id: str | None = None,
        scenario: str = "president_state",
    ) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("readiness-%Y%m%dT%H%M%SZ")
        out_dir = self.context.artifacts_dir / "readiness" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        checks = [
            self._base_dependencies(),
            self._production_default_config(),
            self._documentation_default(),
            self._fundamentals_prior(forecast_run_id),
            self._reward_gates(forecast_run_id),
            self._phase8_verification(forecast_run_id),
            self._live_2026_scope(forecast_run_id),
            self._rolling_origin_evidence(
                bayes_backtest_run_id=bayes_backtest_run_id,
                legacy_backtest_run_id=legacy_backtest_run_id,
            ),
        ]
        payload = {
            "run_id": run_id,
            "scenario": scenario,
            "forecast_run_id": forecast_run_id,
            "bayes_backtest_run_id": bayes_backtest_run_id,
            "legacy_backtest_run_id": legacy_backtest_run_id,
            "status": "eligible" if all(bool(check["passed"]) for check in checks) else "blocked",
            "eligible_for_default_switch": all(bool(check["passed"]) for check in checks),
            "checks": checks,
            "output_dir": str(out_dir),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        write_json(payload, out_dir / "methodology_readiness.json")
        write_text(self._report(payload), out_dir / "methodology_readiness.md")
        return payload

    def _base_dependencies(self) -> dict[str, Any]:
        project = self._pyproject()
        dependencies = [str(item).lower() for item in project.get("dependencies", [])]
        optional = {
            key: [str(item).lower() for item in values]
            for key, values in dict(project.get("optional-dependencies", {})).items()
            if isinstance(values, list)
        }
        required = ["jax", "jaxlib", "numpyro", "arviz"]
        base_present = [
            package
            for package in required
            if any(dep == package or dep.startswith(f"{package}>") for dep in dependencies)
        ]
        optional_present = [
            package
            for package in required
            if any(
                dep == package or dep.startswith(f"{package}>")
                for values in optional.values()
                for dep in values
            )
        ]
        missing = sorted(set(required) - set(base_present))
        return {
            "name": "bayes_dependencies_in_base",
            "passed": not missing,
            "detail": {
                "required": required,
                "base_present": base_present,
                "optional_present": optional_present,
                "missing_from_base": missing,
                "policy": "Bayes cannot be production default while required deps are optional.",
            },
        }

    def _production_default_config(self) -> dict[str, Any]:
        config = self.context.read_yaml("model.yaml")
        bayesian = dict(config.get("bayesian", {}))
        enabled = bool(bayesian.get("enabled"))
        backend = str(bayesian.get("backend", "")).lower().strip()
        passed = enabled and backend == "nuts"
        return {
            "name": "bayesian_config_default",
            "passed": passed,
            "detail": {
                "bayesian_enabled": enabled,
                "backend": backend,
                "required": {"bayesian.enabled": True, "bayesian.backend": "nuts"},
            },
        }

    def _documentation_default(self) -> dict[str, Any]:
        docs = {
            "README.md": self.context.root / "README.md",
            "SPEC.md": self.context.root / "SPEC.md",
            "docs/methodology.md": self.context.root / "docs" / "methodology.md",
        }
        required_phrase = "bayesian path is the production default"
        missing = []
        for label, path in docs.items():
            text = path.read_text(encoding="utf-8").lower() if path.exists() else ""
            if required_phrase not in text:
                missing.append(label)
        return {
            "name": "docs_declare_bayes_default",
            "passed": not missing,
            "detail": {
                "required_phrase": required_phrase,
                "missing": missing,
            },
        }

    def _fundamentals_prior(self, forecast_run_id: str | None) -> dict[str, Any]:
        run_dir = self._forecast_run_dir(forecast_run_id)
        if run_dir is None:
            return {
                "name": "fundamentals_prior_production_path",
                "passed": False,
                "detail": "forecast_run_id is required",
            }
        diagnostics = self._read_run_json(run_dir, "posterior_diagnostics.json")
        prior_path = run_dir / "fundamentals_prior.parquet"
        used = bool(diagnostics.get("fundamentals_prior_used"))
        rows = int(diagnostics.get("fundamentals_prior_rows") or 0)
        passed = prior_path.exists() and used and rows > 0
        return {
            "name": "fundamentals_prior_production_path",
            "passed": passed,
            "detail": {
                "fundamentals_prior_artifact": str(prior_path),
                "artifact_exists": prior_path.exists(),
                "fundamentals_prior_used": used,
                "fundamentals_prior_rows": rows,
            },
        }

    def _reward_gates(self, forecast_run_id: str | None) -> dict[str, Any]:
        run_dir = self._forecast_run_dir(forecast_run_id)
        if run_dir is None:
            return {
                "name": "reward_hard_gates",
                "passed": False,
                "detail": "forecast_run_id is required",
            }
        rewards = self._read_run_json(run_dir, "reward_card.json").get("rewards", {})
        required = [
            "R13_posterior_quality",
            "R14_calibrated_publication",
            "R15_daily_update_quality",
        ]
        statuses = {
            key: dict(rewards.get(key, {})).get("passed")
            for key in required
            if isinstance(rewards.get(key), dict)
        }
        missing = [key for key in required if key not in statuses]
        failed = [key for key, value in statuses.items() if value is not True]
        return {
            "name": "reward_hard_gates",
            "passed": not missing and not failed,
            "detail": {
                "required": required,
                "missing": missing,
                "failed": failed,
                "statuses": statuses,
            },
        }

    def _phase8_verification(self, forecast_run_id: str | None) -> dict[str, Any]:
        run_dir = self._forecast_run_dir(forecast_run_id)
        if run_dir is None:
            return {
                "name": "phase8_verification_passed",
                "passed": False,
                "detail": "forecast_run_id is required",
            }
        payload = self._read_run_json(run_dir, "phase8_verification.json")
        return {
            "name": "phase8_verification_passed",
            "passed": bool(payload.get("passed")),
            "detail": {
                "passed": payload.get("passed"),
                "scenario": payload.get("scenario"),
                "inference_engine": payload.get("inference_engine"),
                "bayesian_backend": payload.get("bayesian_backend"),
            },
        }

    def _live_2026_scope(self, forecast_run_id: str | None) -> dict[str, Any]:
        run_dir = self._forecast_run_dir(forecast_run_id)
        if run_dir is None:
            return {
                "name": "live_2026_source_scope",
                "passed": False,
                "detail": "forecast_run_id is required",
            }
        fixture_scope = dict(
            self._read_run_json(run_dir, "phase8_verification.json").get("fixture_scope", {})
        )
        live_status = str(fixture_scope.get("live_2026_status", "missing"))
        return {
            "name": "live_2026_source_scope",
            "passed": live_status == "claimed",
            "detail": {
                "live_2026_status": live_status,
                "fixture_scope_status": fixture_scope.get("status"),
                "live_source_scope": fixture_scope.get("live_source_scope", {}),
                "policy": (
                    "Production-default Phase 8 requires a live-source claim from "
                    "model-bearing rows, not only fixtures or metadata-only rows."
                ),
            },
        }

    def _rolling_origin_evidence(
        self,
        *,
        bayes_backtest_run_id: str | None,
        legacy_backtest_run_id: str | None,
    ) -> dict[str, Any]:
        bayes = self._read_backtest_scorecard(bayes_backtest_run_id)
        legacy = self._read_backtest_scorecard(legacy_backtest_run_id)
        if not bayes or not legacy:
            return {
                "name": "rolling_origin_beats_legacy",
                "passed": False,
                "detail": {
                    "bayes_backtest_run_id": bayes_backtest_run_id,
                    "legacy_backtest_run_id": legacy_backtest_run_id,
                    "reason": "both bayes and legacy Kalman backtest scorecards are required",
                },
            }
        bayes_all_metrics = dict(bayes.get("metrics", {}))
        legacy_all_metrics = dict(legacy.get("metrics", {}))
        bayes_metrics = dict(bayes_all_metrics.get("ensemble", {}))
        legacy_metrics = dict(legacy_all_metrics.get("ensemble", {}))
        bayes_polling_metrics = dict(bayes_all_metrics.get("polling", {}))
        legacy_polling_metrics = dict(legacy_all_metrics.get("polling", {}))
        bayes_log = bayes_metrics.get("log_score")
        legacy_log = legacy_metrics.get("log_score")
        bayes_polling_log = bayes_polling_metrics.get("log_score")
        legacy_polling_log = legacy_polling_metrics.get("log_score")
        bayes_coverage = bayes_metrics.get("interval_90_coverage")
        legacy_coverage = legacy_metrics.get("interval_90_coverage")
        beats_log = (
            bayes_log is not None
            and legacy_log is not None
            and float(bayes_log) < float(legacy_log)
        )
        polling_beats_log = (
            bayes_polling_log is not None
            and legacy_polling_log is not None
            and float(bayes_polling_log) < float(legacy_polling_log)
        )
        coverage_ok = (
            bayes_coverage is not None
            and legacy_coverage is not None
            and float(bayes_coverage) >= float(legacy_coverage)
        )
        return {
            "name": "rolling_origin_beats_legacy",
            "passed": bool(beats_log and coverage_ok),
            "detail": {
                "bayes_backtest_run_id": bayes_backtest_run_id,
                "legacy_backtest_run_id": legacy_backtest_run_id,
                "bayes_log_score": bayes_log,
                "legacy_log_score": legacy_log,
                "bayes_polling_log_score": bayes_polling_log,
                "legacy_polling_log_score": legacy_polling_log,
                "bayes_interval_90_coverage": bayes_coverage,
                "legacy_interval_90_coverage": legacy_coverage,
                "beats_log_score": beats_log,
                "polling_component_beats_log_score": polling_beats_log,
                "coverage_not_degraded": coverage_ok,
            },
        }

    def _forecast_run_dir(self, run_id: str | None) -> Path | None:
        if not run_id:
            return None
        return self.context.artifacts_dir / "runs" / run_id

    def _read_run_json(self, run_dir: Path, name: str) -> dict[str, Any]:
        path = run_dir / name
        return read_json(path) if path.exists() else {}

    def _read_backtest_scorecard(self, run_id: str | None) -> dict[str, Any]:
        if not run_id:
            return {}
        path = self.context.artifacts_dir / "backtests" / run_id / "scorecard.json"
        return read_json(path) if path.exists() else {}

    def _pyproject(self) -> dict[str, Any]:
        path = self.context.root / "pyproject.toml"
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        project = payload.get("project", {})
        return dict(project) if isinstance(project, dict) else {}

    @staticmethod
    def _report(payload: dict[str, Any]) -> str:
        lines = [
            "# Methodology Readiness",
            "",
            f"- Status: `{payload['status']}`",
            f"- Eligible for default switch: `{payload['eligible_for_default_switch']}`",
            f"- Forecast run: `{payload.get('forecast_run_id')}`",
            f"- Bayes backtest: `{payload.get('bayes_backtest_run_id')}`",
            f"- Legacy backtest: `{payload.get('legacy_backtest_run_id')}`",
            "",
            "## Checks",
            "",
        ]
        for check in payload["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- `{mark}` `{check['name']}`")
        lines.extend(
            [
                "",
                "This report is intentionally conservative: a fixture-backed Phase 8 pass is",
                "not treated as a production-default switch unless dependency placement,",
                "docs, live-source scope, posterior quality, calibration, daily updates, and",
                "rolling-origin legacy comparison evidence all pass.",
                "",
            ]
        )
        return "\n".join(lines)
