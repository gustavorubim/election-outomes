from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features.tiering import TierAssessor
from civic_signal.storage.io import write_parquet


@dataclass(frozen=True)
class FeatureBundle:
    races: pl.DataFrame
    options: pl.DataFrame
    polls: pl.DataFrame
    markets: pl.DataFrame
    public_signals: pl.DataFrame
    fundamentals: pl.DataFrame
    results: pl.DataFrame
    backtest_predictions: pl.DataFrame
    race_catalog: pl.DataFrame


class FeatureBuilder:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def run(self) -> FeatureBundle:
        tables = {
            name: self._read_table(name)
            for name in [
                "races",
                "options",
                "polls",
                "market_quotes",
                "public_signals",
                "fundamentals",
                "results",
                "backtest_predictions",
            ]
        }
        assessor = TierAssessor(self.context.read_yaml("tiers.yaml"))
        race_catalog = assessor.assign(
            tables["races"],
            tables["polls"],
            tables["market_quotes"],
            tables["fundamentals"],
            tables["public_signals"],
        )
        write_parquet(race_catalog, self.context.curated_dir / "race_catalog.parquet")
        return FeatureBundle(
            races=tables["races"],
            options=tables["options"],
            polls=tables["polls"],
            markets=tables["market_quotes"],
            public_signals=tables["public_signals"],
            fundamentals=tables["fundamentals"],
            results=tables["results"],
            backtest_predictions=tables["backtest_predictions"],
            race_catalog=race_catalog,
        )

    def _read_table(self, name: str) -> pl.DataFrame:
        path: Path = self.context.curated_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing curated table: {path}")
        return pl.read_parquet(path)
