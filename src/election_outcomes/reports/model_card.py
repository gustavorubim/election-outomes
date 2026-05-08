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
    ) -> str:
        covariance_rows = 0 if residual_covariance is None else residual_covariance.height
        fit_status = {
            "polling": "deterministic aggregator; NumPyro hierarchical model planned",
            "fundamentals": "ridge fit when historical rows meet threshold, otherwise defaults",
            "markets": "configured public-market inversion; calibration artifact planned",
            "public_signals": "experimental unless admission artifact proves value",
            "covariance": (
                "learned from rolling residuals" if covariance_rows else "fixed config fallback"
            ),
        }
        return f"""# Model Card

- Run id: `{run_id}`
- Scenario: `{(scenario or {}).get("name", "default")}`
- Model version: `{model_config.get("model_version")}`
- Admission status: `{component_admission.get("admission_status", "not_available")}`
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

## Parameter Status

```json
{json.dumps(fit_status, indent=2, sort_keys=True)}
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

```json
{
            json.dumps(
                source_manifest.select(
                    ["source_id", "table", "parser_version", "status"]
                ).to_dicts()
                if not source_manifest.is_empty()
                else [],
                indent=2,
                sort_keys=True,
            )
        }
```
"""
