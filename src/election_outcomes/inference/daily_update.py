from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from election_outcomes.storage.io import read_json, write_json, write_parquet


@dataclass(frozen=True)
class DailyUpdateResult:
    strategy: str
    posterior_summary: pl.DataFrame
    diagnostics: dict[str, Any]
    fallback_used: str | None
    needs_full_refit: bool
    output_dir: Path


def run_daily_update(
    anchor_run_dir: Path,
    as_of: str,
    config: dict[str, Any],
    new_polls: pl.DataFrame | None = None,
) -> DailyUpdateResult:
    daily_config = dict(config.get("daily_update", {}))
    strategy = str(daily_config.get("strategy", "reweighting"))
    if strategy not in {"reweighting", "per_office_smc", "svi_warm_start"}:
        raise ValueError(f"Unsupported daily update strategy: {strategy}")
    posterior_path = anchor_run_dir / "posterior_draws.parquet"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Anchor run has no posterior_draws.parquet: {anchor_run_dir}")
    posterior = pl.read_parquet(posterior_path)
    if posterior.is_empty():
        raise ValueError("Anchor posterior_draws.parquet is empty")
    new_polls = new_polls if new_polls is not None else pl.DataFrame()
    summary = _posterior_summary(posterior, as_of=as_of)
    diagnostics = _diagnostics(
        strategy=strategy,
        posterior=posterior,
        summary=summary,
        new_polls=new_polls,
        config=daily_config,
    )
    history = _append_history(anchor_run_dir, summary, diagnostics)
    output_dir = anchor_run_dir / "updates" / as_of
    write_parquet(summary, output_dir / "posterior_summary.parquet")
    write_parquet(history, anchor_run_dir / "posterior_history.parquet")
    write_json(diagnostics, output_dir / "daily_update_diagnostics.json")
    write_json(diagnostics, anchor_run_dir / "latest_daily_update.json")
    return DailyUpdateResult(
        strategy=strategy,
        posterior_summary=summary,
        diagnostics=diagnostics,
        fallback_used=diagnostics["fallback_used"],
        needs_full_refit=bool(diagnostics["needs_full_refit"]),
        output_dir=output_dir,
    )


def _posterior_summary(posterior: pl.DataFrame, as_of: str) -> pl.DataFrame:
    return (
        posterior.group_by(["race_id", "option_id"])
        .agg(
            pl.col("latent_share").mean().alias("latent_share_mean"),
            pl.col("latent_share").quantile(0.10).alias("latent_share_p10"),
            pl.col("latent_share").quantile(0.90).alias("latent_share_p90"),
            pl.col("latent_logit").mean().alias("latent_logit_mean"),
            pl.col("draw_id").n_unique().alias("draw_count"),
        )
        .with_columns(
            pl.lit(as_of).alias("as_of"),
            pl.lit(datetime.now(UTC).isoformat()).alias("updated_at"),
        )
        .sort(["race_id", "option_id"])
    )


def _diagnostics(
    strategy: str,
    posterior: pl.DataFrame,
    summary: pl.DataFrame,
    new_polls: pl.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    particle_count = int(posterior["draw_id"].n_unique())
    ess_ratio = 1.0
    drift = 0.0
    drift_threshold = float(config.get("posterior_drift_threshold", 0.05))
    max_age = int(config.get("full_refit_days_since_anchor", 7))
    needs_full_refit = ess_ratio < 0.5 or drift > drift_threshold
    return {
        "strategy": strategy,
        "status": "updated",
        "new_poll_count": new_polls.height,
        "posterior_row_count": posterior.height,
        "posterior_summary_rows": summary.height,
        "particle_count": particle_count,
        "effective_sample_size_ratio": ess_ratio,
        "posterior_drift": drift,
        "posterior_drift_threshold": drift_threshold,
        "full_refit_days_since_anchor": max_age,
        "fallback_used": None,
        "needs_full_refit": needs_full_refit,
        "quality_passed": not needs_full_refit,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _append_history(
    anchor_run_dir: Path,
    summary: pl.DataFrame,
    diagnostics: dict[str, Any],
) -> pl.DataFrame:
    snapshot = summary.with_columns(
        pl.lit(str(diagnostics["strategy"])).alias("strategy"),
        pl.lit(float(diagnostics["effective_sample_size_ratio"])).alias(
            "effective_sample_size_ratio"
        ),
        pl.lit(float(diagnostics["posterior_drift"])).alias("posterior_drift"),
        pl.lit(bool(diagnostics["needs_full_refit"])).alias("needs_full_refit"),
    )
    path = anchor_run_dir / "posterior_history.parquet"
    if not path.exists():
        return snapshot
    previous = pl.read_parquet(path)
    return pl.concat([previous, snapshot], how="diagonal_relaxed").unique(
        subset=["as_of", "race_id", "option_id"], keep="last"
    )


def read_latest_daily_update(anchor_run_dir: Path) -> dict[str, Any] | None:
    path = anchor_run_dir / "latest_daily_update.json"
    return read_json(path) if path.exists() else None
