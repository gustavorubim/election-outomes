from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectContext:
    """Resolved project paths and YAML configuration."""

    root: Path
    config_dir: Path
    sources_config: str
    data_dir: Path
    artifacts_dir: Path

    @classmethod
    def create(
        cls,
        root: str | Path | None = None,
        config_dir: str | Path | None = None,
        sources_config: str = "sources.yaml",
        data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> ProjectContext:
        project_root = Path(root or Path.cwd()).resolve()
        resolved_config = Path(config_dir or project_root / "configs").resolve()
        return cls(
            root=project_root,
            config_dir=resolved_config,
            sources_config=sources_config,
            data_dir=Path(data_dir or project_root / "data").resolve(),
            artifacts_dir=Path(artifacts_dir or project_root / "artifacts").resolve(),
        )

    def read_yaml(self, name: str) -> dict[str, Any]:
        path = self.config_dir / name
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        return data

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"
