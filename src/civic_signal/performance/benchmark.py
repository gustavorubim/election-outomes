from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from civic_signal.storage.io import write_json


@dataclass(frozen=True)
class BenchmarkResult:
    payload: dict[str, Any]
    output_path: Path


class PerformanceBenchmark:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def run(
        self,
        bundle: FeatureBundle,
        model_config: dict[str, Any],
        run_id: str,
        draws: int | None = None,
        repeats: int | None = None,
    ) -> BenchmarkResult:
        performance_config = dict(model_config.get("performance", {}))
        draw_count = int(draws or performance_config.get("benchmark_draws", 10000))
        repeat_count = int(repeats or performance_config.get("benchmark_repeats", 3))
        benchmark_config = dict(model_config)
        benchmark_config["simulation_count"] = draw_count

        ensemble = self._ensemble(bundle, benchmark_config)
        engine = SimulationEngine(benchmark_config)
        engine.run(bundle, ensemble)
        elapsed: list[float] = []
        row_counts: list[int] = []
        for _ in range(repeat_count):
            started = time.perf_counter()
            outputs = engine.run(bundle, ensemble)
            elapsed.append(time.perf_counter() - started)
            row_counts.append(outputs.draws.height)
        best_seconds = min(elapsed)
        payload = {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "draws": draw_count,
            "repeats": repeat_count,
            "forecast_draw_rows": int(max(row_counts) if row_counts else 0),
            "best_seconds": best_seconds,
            "rows_per_second": float(max(row_counts) / best_seconds if best_seconds else 0.0),
            "performance": engine.performance_metadata(),
        }
        out_dir = self.context.artifacts_dir / "benchmarks" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = write_json(payload, out_dir / "performance_benchmark.json")
        return BenchmarkResult(payload=payload, output_path=output_path)

    @staticmethod
    def _ensemble(bundle: FeatureBundle, model_config: dict[str, Any]) -> pl.DataFrame:
        estimates = [
            PollingModel(model_config, inference_engine="kalman").run(bundle),
            FundamentalsModel().run(bundle),
            MarketModel(model_config).run(bundle),
            PublicSignalModel(
                trusted=bool(
                    model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(bundle),
        ]
        return EnsembleModel(model_config).run(bundle, estimates)
