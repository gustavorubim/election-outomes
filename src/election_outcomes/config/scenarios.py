from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from election_outcomes.config.context import ProjectContext


@dataclass(frozen=True)
class Scenario:
    name: str
    payload: dict[str, Any]

    @property
    def default_as_of(self) -> str | None:
        value = self.payload.get("default_as_of")
        return str(value) if value else None

    @property
    def cycle(self) -> int | None:
        value = self.payload.get("cycle")
        return int(value) if value is not None else None

    @property
    def family(self) -> str:
        return str(self.payload.get("family") or self.name)

    @property
    def storage_key(self) -> str:
        return self.family if self.name.endswith("_state") and self.cycle is not None else self.name

    @property
    def control_body(self) -> str | None:
        value = self.payload.get("control_body")
        return str(value) if value else None

    @property
    def holdovers(self) -> dict[str, int]:
        raw = self.payload.get("holdovers")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, int] = {}
        for key, value in raw.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            result[str(key).upper()] = count
        return result

    def filter_catalog(self, catalog: pl.DataFrame, include_cycle: bool = True) -> pl.DataFrame:
        frame = catalog
        for column in ("office_type", "geography_type", "control_body"):
            value = self.payload.get(column)
            if value is not None and column in frame.columns:
                frame = frame.filter(pl.col(column) == value)
        if include_cycle and self.cycle is not None and "cycle" in frame.columns:
            frame = frame.filter(pl.col("cycle") == self.cycle)
        return frame

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, **self.payload}


class ScenarioRegistry:
    def __init__(self, scenarios: dict[str, dict[str, Any]]) -> None:
        self._scenarios = scenarios

    @classmethod
    def from_context(cls, context: ProjectContext) -> ScenarioRegistry:
        payload = context.read_yaml("scenarios.yaml")
        raw = payload.get("scenarios", {})
        if not isinstance(raw, dict):
            raise ValueError("configs/scenarios.yaml must contain a scenarios mapping")
        return cls({str(key): dict(value or {}) for key, value in raw.items()})

    def get(self, name: str | None) -> Scenario | None:
        if name is None:
            return None
        if name not in self._scenarios:
            raise ValueError(f"Unknown scenario {name!r}")
        return Scenario(name=name, payload=dict(self._scenarios[name]))
