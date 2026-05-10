from __future__ import annotations

import json
from typing import Any

import polars as pl


class ModelCard:
    """Run-local model card focused on learned vs configured assumptions."""

    def render(
        self,
        run_id: str,
        scenario: dict[str, Any] | None,
        model_config: dict[str, Any],
        backtest_payload: dict[str, Any],
        component_admission: dict[str, Any],
        residual_covariance: pl.DataFrame | None,
        source_manifest: pl.DataFrame,
        runtime_metadata: dict[str, Any] | None = None,
        pollster_house_effects: dict[tuple[str, str | None], Any] | None = None,
    ) -> str:
        covariance_rows = 0 if residual_covariance is None else residual_covariance.height
        runtime_metadata = runtime_metadata or {}
        admission_source = dict(model_config.get("component_admission_source", {}))
        house_effect_rows = self._house_effect_rows(pollster_house_effects or {})
        fundamentals_status = runtime_metadata.get("fundamentals", {}).get(
            "fit_status", "not captured for rebuilt report"
        )
        fundamentals_prior = runtime_metadata.get("fundamentals_prior", {})
        polling_metadata = runtime_metadata.get("polling", {})
        polling_engine = polling_metadata.get("engine")
        polling_status = (
            "deterministic Gaussian state-space Kalman filter with empirical-Bayes "
            f"pollster house-effect shrinkage; learned_effects={len(house_effect_rows)}"
        )
        if polling_engine and polling_engine != "kalman":
            polling_status = (
                f"{polling_engine}; draw_count={polling_metadata.get('draw_count')}; "
                f"race_options={polling_metadata.get('race_option_count')}; "
                f"fallback={polling_metadata.get('fallback_used')}; "
                f"learned_effects={len(house_effect_rows)}"
            )
        covariance_status = "fixed config fallback"
        if covariance_rows:
            sample_size = (
                int(residual_covariance["sample_size"].max())
                if "sample_size" in residual_covariance.columns
                else None
            )
            matrix_rank = (
                int(residual_covariance["matrix_rank"].max())
                if "matrix_rank" in residual_covariance.columns
                else None
            )
            method_values = (
                residual_covariance["covariance_method"].drop_nulls().to_list()
                if "covariance_method" in residual_covariance.columns
                else []
            )
            method = str(method_values[0]) if method_values else "learned from rolling residuals"
            covariance_status = f"{method}; sample_size={sample_size}; matrix_rank={matrix_rank}"
        fit_status = {
            "polling": polling_status,
            "fundamentals": fundamentals_status,
            "fundamentals_prior": fundamentals_prior,
            "markets": (
                "configured public-market inversion; ensemble calibration applied downstream"
            ),
            "public_signals": "experimental unless admission artifact proves value",
            "covariance": covariance_status,
            "probability_calibration": model_config.get("probability_calibration", {}),
            "recalibration_map": runtime_metadata.get("recalibration_map", {}),
            "office_methodology": runtime_metadata.get("office_methodology", {}),
        }
        return f"""# Model Card

- Run id: `{run_id}`
- Scenario: `{(scenario or {}).get("name", "default")}`
- Model version: `{model_config.get("model_version")}`
- Admission status: `{component_admission.get("admission_status", "not_available")}`
- Admission source: `{admission_source.get("status", "not_available")}`
- Engine using: `{admission_source.get("engine_using", "unknown")}`
- Rolling-origin rows: `{backtest_payload.get("row_count", 0)}`
- Residual covariance rows: `{covariance_rows}`
- Source manifest rows: `{source_manifest.height}`

## Component Admission

```json
{json.dumps(component_admission.get("trusted_components", {}), indent=2, sort_keys=True)}
```

## Component Weights

```json
{json.dumps(model_config.get("component_weights", {}), indent=2, sort_keys=True)}
```

## Probability Calibration

```json
{json.dumps(model_config.get("probability_calibration", {}), indent=2, sort_keys=True)}
```

## Admission Source

```json
{json.dumps(admission_source, indent=2, sort_keys=True)}
```

## Parameter Status

```json
{json.dumps(fit_status, indent=2, sort_keys=True)}
```

## Pollster House Effects

```json
{json.dumps(house_effect_rows[:50], indent=2, sort_keys=True)}
```

## Backtest Summary

```json
{
            json.dumps(
                {
                    "method": backtest_payload.get("method"),
                    "rolling_origin_executed": backtest_payload.get("rolling_origin_executed"),
                    "sample_size_too_small": backtest_payload.get("sample_size_too_small"),
                    "minimum_rows_for_trust": backtest_payload.get("minimum_rows_for_trust"),
                    "metrics": backtest_payload.get("metrics", {}),
                },
                indent=2,
                sort_keys=True,
            )
        }
```

## Source Coverage

Status and retrieval timestamps are preserved in `source_manifest.parquet`; this summary
keeps only stable source identity fields so same-input reruns can reproduce the model
card fingerprint.

```json
{
            json.dumps(
                source_manifest.select(["source_id", "table", "parser_version"]).to_dicts()
                if not source_manifest.is_empty()
                else [],
                indent=2,
                sort_keys=True,
            )
        }
```
"""

    @staticmethod
    def _house_effect_rows(
        house_effects: dict[tuple[str, str | None], Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (_pollster, _option_id), estimate in house_effects.items():
            if hasattr(estimate, "__dict__"):
                payload = dict(estimate.__dict__)
            else:
                payload = {
                    "pollster": getattr(estimate, "pollster", None),
                    "option_id": getattr(estimate, "option_id", None),
                    "effect": getattr(estimate, "effect", None),
                    "raw_effect": getattr(estimate, "raw_effect", None),
                    "prior_effect": getattr(estimate, "prior_effect", None),
                    "shrinkage": getattr(estimate, "shrinkage", None),
                    "poll_count": getattr(estimate, "poll_count", None),
                }
            rows.append(payload)
        return sorted(rows, key=lambda row: (str(row.get("pollster")), str(row.get("option_id"))))
