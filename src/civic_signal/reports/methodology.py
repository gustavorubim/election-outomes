from __future__ import annotations

import json
from typing import Any


class MethodologySnapshot:
    def render(
        self,
        run_id: str,
        as_of: str,
        model_config: dict[str, Any],
        source_count: int,
    ) -> str:
        return f"""# Methodology Snapshot

- Run id: `{run_id}`
- As of: `{as_of}`
- Model version: `{model_config.get("model_version")}`
- Simulation count: `{model_config.get("simulation_count")}`
- Source count: `{source_count}`

## Trusted Components

```json
{json.dumps(model_config.get("trusted_components", {}), indent=2, sort_keys=True)}
```

## Component Weights

```json
{json.dumps(model_config.get("component_weights", {}), indent=2, sort_keys=True)}
```

## Probability Calibration

```json
{json.dumps(model_config.get("probability_calibration", {}), indent=2, sort_keys=True)}
```

## Limitations

This run uses the implemented deterministic hybrid engine over the configured source
registry. Public-signal features remain experimental unless admitted by backtest
reward gates. Tier C races are tracked but not assigned trusted probabilities.
Close-margin administrative-risk proxies stay withheld unless explicitly enabled as
experimental output.
"""
