from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from civic_signal.config.context import ProjectContext


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
        """Raw holdover seats per declared party (IND kept as a separate key)."""
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

    @property
    def caucus_with(self) -> dict[str, str]:
        """Optional caucus mapping (e.g. IND -> DEM). Defaults to identity."""
        raw = self.payload.get("caucus_with")
        if not isinstance(raw, dict):
            return {}
        return {str(key).upper(): str(value).upper() for key, value in raw.items()}

    @property
    def holdover_caucus_seats(self) -> dict[str, int]:
        """Holdover seats credited to each caucus party.

        IND seats fold into their declared caucus partner (e.g. King and Sanders
        caucus with DEM). Used for control / majority math; the raw `holdovers`
        property still reports per-party counts as declared in the scenario.
        """
        caucus_map = self.caucus_with
        result: dict[str, int] = {}
        for party, seats in self.holdovers.items():
            target = caucus_map.get(party, party)
            result[target] = result.get(target, 0) + seats
        return result

    def filter_catalog(self, catalog: pl.DataFrame, include_cycle: bool = True) -> pl.DataFrame:
        frame = catalog
        for column, aliases in {
            "office_type": ("office_type", "office_types", "offices"),
            "geography_type": ("geography_type", "geography_types"),
            "control_body": ("control_body", "control_bodies"),
        }.items():
            if column in frame.columns:
                frame = self._filter_column(frame, column, aliases)
        if include_cycle and self.cycle is not None and "cycle" in frame.columns:
            frame = frame.filter(pl.col("cycle") == self.cycle)
        return frame

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, **self.payload}

    def _filter_column(
        self, frame: pl.DataFrame, column: str, aliases: tuple[str, ...]
    ) -> pl.DataFrame:
        for alias in aliases:
            if alias not in self.payload:
                continue
            value = self.payload.get(alias)
            if value is None:
                return frame
            if isinstance(value, list):
                values = [str(item) for item in value]
                return frame.filter(pl.col(column).is_in(values))
            return frame.filter(pl.col(column) == str(value))
        return frame


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
