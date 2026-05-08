from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.scoring.metrics import score_predictions
from election_outcomes.storage.io import write_json, write_parquet


class BacktestRunner:
    COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "baseline": "baseline_probability",
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
        "ensemble": "ensemble_probability",
    }

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def evaluate(self) -> dict[str, object]:
        path = self.context.curated_dir / "backtest_predictions.parquet"
        frame = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        config = self.context.read_yaml("backtests.yaml")
        minimum_rows = int(config.get("minimum_rows_for_trust", 30))
        metrics = {
            component: score_predictions(frame, column)
            for component, column in self.COMPONENT_COLUMNS.items()
            if column in frame.columns
        }
        baseline_brier = metrics.get("baseline", {}).get("brier")
        ablations = {}
        for component, values in metrics.items():
            if component == "baseline" or baseline_brier is None:
                continue
            ablations[component] = {
                "brier_delta_vs_baseline": values["brier"] - baseline_brier,
                "beats_or_matches_baseline": values["brier"] <= baseline_brier,
            }
        rolling = self._rolling_origin(frame)
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "fixture_scorecard_with_per_cycle_rolling_evaluation",
            "rolling_origin_executed": rolling["executed"],
            "rolling_origin": rolling,
            "minimum_rows_for_trust": minimum_rows,
            "sample_size_too_small": frame.height < minimum_rows,
            "row_count": frame.height,
            "metrics": metrics,
            "ablations": ablations,
        }

    def _rolling_origin(self, frame: pl.DataFrame) -> dict[str, object]:
        if frame.is_empty() or "cycle" not in frame.columns:
            return {
                "executed": False,
                "method": "per_cycle_evaluation_of_fixture_predictions",
                "reason": "no cycle column or empty frame",
                "cycles": [],
                "per_cycle_metrics": {},
            }
        cycles = sorted(int(value) for value in frame["cycle"].unique().to_list())
        if len(cycles) < 2:
            return {
                "executed": False,
                "method": "per_cycle_evaluation_of_fixture_predictions",
                "reason": f"only {len(cycles)} cycle(s) available",
                "cycles": cycles,
                "per_cycle_metrics": {},
            }
        per_cycle: dict[str, dict[str, float]] = {}
        for cycle in cycles:
            cycle_frame = frame.filter(pl.col("cycle") == cycle)
            per_cycle[str(cycle)] = score_predictions(cycle_frame, "ensemble_probability")
        return {
            "executed": True,
            "method": "per_cycle_evaluation_of_fixture_predictions",
            "note": (
                "Per-cycle rescoring of pre-baked predictions. Replace with true rolling-origin "
                "training when component models can refit on prior cycles only."
            ),
            "cycles": cycles,
            "per_cycle_metrics": per_cycle,
        }

    def run(self, run_id: str) -> dict[str, object]:
        payload = self.evaluate()
        out_dir = self.context.artifacts_dir / "backtests" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows = [
            {"component": component, **values} for component, values in payload["metrics"].items()
        ]
        write_parquet(pl.DataFrame(metrics_rows), out_dir / "scorecard.parquet")
        write_json(payload, out_dir / "scorecard.json")
        return payload
