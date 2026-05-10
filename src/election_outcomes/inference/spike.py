from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.scoring import BacktestRunner
from election_outcomes.storage.io import write_json, write_parquet


@dataclass(frozen=True)
class Phase0ComparisonResult:
    run_id: str
    output_dir: Path
    payload: dict[str, Any]


def run_phase0_comparison(
    context: ProjectContext,
    *,
    run_id: str,
    scenario: str = "president_state",
    holdout_cycle: int = 2024,
    bayesian_backend: str | None = None,
) -> Phase0ComparisonResult:
    runner = BacktestRunner(context)
    engines = ["kalman", "bayes"]
    rows = []
    rolling_frames: dict[str, pl.DataFrame] = {}
    scorecards: dict[str, dict[str, Any]] = {}
    for engine in engines:
        artifacts = runner._evaluate(
            scenario=scenario,
            holdout_cycle=holdout_cycle,
            inference_engine=engine,
            bayesian_backend=bayesian_backend if engine == "bayes" else None,
        )
        scorecards[engine] = artifacts.payload
        rolling_frames[engine] = artifacts.rolling_predictions
        metrics = artifacts.payload.get("metrics", {})
        rows.append(
            {
                "engine": engine,
                "row_count": artifacts.payload.get("row_count", 0),
                **{
                    f"ensemble_{key}": value
                    for key, value in dict(metrics.get("ensemble", {})).items()
                },
                **{
                    f"polling_{key}": value
                    for key, value in dict(metrics.get("polling", {})).items()
                },
            }
        )

    comparison = pl.DataFrame(rows)
    payload = _comparison_payload(
        run_id=run_id,
        scenario=scenario,
        holdout_cycle=holdout_cycle,
        comparison=comparison,
    )
    output_dir = context.artifacts_dir / "spikes" / run_id
    write_parquet(comparison, output_dir / "phase0_comparison.parquet")
    for engine, frame in rolling_frames.items():
        write_parquet(frame, output_dir / f"rolling_predictions_{engine}.parquet")
        write_json(scorecards[engine], output_dir / f"scorecard_{engine}.json")
    write_json(payload, output_dir / "comparison.json")
    return Phase0ComparisonResult(run_id=run_id, output_dir=output_dir, payload=payload)


def _comparison_payload(
    *,
    run_id: str,
    scenario: str,
    holdout_cycle: int,
    comparison: pl.DataFrame,
) -> dict[str, Any]:
    rows = {row["engine"]: row for row in comparison.to_dicts()}
    kalman = rows.get("kalman", {})
    bayes = rows.get("bayes", {})
    kalman_log = kalman.get("ensemble_log_score")
    bayes_log = bayes.get("ensemble_log_score")
    log_loss_delta = (
        float(bayes_log) - float(kalman_log)
        if bayes_log is not None and kalman_log is not None
        else None
    )
    return {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario": scenario,
        "holdout_cycle": holdout_cycle,
        "engines": ["kalman", "bayes"],
        "comparison": comparison.to_dicts(),
        "go_no_go": {
            "metric": "ensemble_log_score",
            "bayes_minus_kalman": log_loss_delta,
            "bayes_beats_or_matches_kalman": bool(
                log_loss_delta is not None and log_loss_delta <= 0
            ),
            "status": "pass" if log_loss_delta is not None and log_loss_delta <= 0 else "fail",
        },
    }
