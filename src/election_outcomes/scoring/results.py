from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from election_outcomes.storage.io import write_json, write_parquet, write_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResultComparator:
    """Compare a forecast run against actual results."""

    def compare(
        self,
        forecast_run_dir: Path,
        curated_results: pl.DataFrame,
        comparison_id: str,
        cycle: int | None = None,
        office_type: str | None = None,
        race_id: str | None = None,
    ) -> dict[str, Any]:
        race_catalog = pl.read_parquet(forecast_run_dir / "race_catalog.parquet")
        race_forecasts = pl.read_parquet(forecast_run_dir / "race_forecasts.parquet")
        comparison = self._comparison_frame(
            race_catalog=race_catalog,
            race_forecasts=race_forecasts,
            curated_results=curated_results,
            cycle=cycle,
            office_type=office_type,
            race_id=race_id,
        )
        summary = self._summary(
            comparison=comparison,
            comparison_id=comparison_id,
            cycle=cycle,
            office_type=office_type,
            race_id=race_id,
        )
        output_dir = forecast_run_dir / "comparisons" / comparison_id
        output_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(comparison, output_dir / "result_comparison.parquet")
        plot_manifest = self._write_plots(comparison, output_dir)
        summary["plot_manifest"] = plot_manifest
        write_json(summary, output_dir / "result_comparison_summary.json")
        write_text(
            self._html_report(summary=summary, comparison=comparison),
            output_dir / "result_comparison.html",
        )
        write_text(
            self._narrative(summary=summary, comparison=comparison),
            output_dir / "narrative.md",
        )
        return {**summary, "output_dir": str(output_dir)}

    def _comparison_frame(
        self,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        curated_results: pl.DataFrame,
        cycle: int | None,
        office_type: str | None,
        race_id: str | None,
    ) -> pl.DataFrame:
        race_meta = race_catalog.select(
            [
                "race_id",
                "cycle",
                "election_date",
                "geography_type",
                "geography",
                "office_type",
                "race_type",
                "tier",
                "tier_reason",
            ]
        ).unique()
        forecasts = race_forecasts.drop(["tier", "tier_reason"], strict=False).join(
            race_meta, on="race_id", how="left"
        )
        forecasts = self._apply_filters(forecasts, cycle, office_type, race_id)
        actuals = curated_results.select(
            [
                "race_id",
                "option_id",
                pl.col("vote_share").alias("actual_vote_share"),
                pl.col("turnout").alias("actual_turnout"),
                pl.col("winner").alias("actual_winner"),
            ]
        )
        comparison = forecasts.join(actuals, on=["race_id", "option_id"], how="inner")
        if comparison.is_empty():
            return comparison
        comparison = comparison.with_columns(
            pl.col("winner_probability").max().over("race_id").alias("max_winner_probability")
        )
        return comparison.with_columns(
            pl.when(pl.col("winner_probability").is_not_null())
            .then(pl.col("winner_probability") == pl.col("max_winner_probability"))
            .otherwise(False)
            .alias("predicted_winner"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share")).alias("vote_share_error"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share"))
            .abs()
            .alias("absolute_vote_share_error"),
            (pl.col("winner_probability") - pl.col("actual_winner").cast(pl.Float64))
            .pow(2)
            .alias("brier_contribution"),
        ).drop("max_winner_probability")

    @staticmethod
    def _apply_filters(
        frame: pl.DataFrame, cycle: int | None, office_type: str | None, race_id: str | None
    ) -> pl.DataFrame:
        filtered = frame
        if cycle is not None:
            filtered = filtered.filter(pl.col("cycle") == cycle)
        if office_type is not None:
            filtered = filtered.filter(pl.col("office_type") == office_type)
        if race_id is not None:
            filtered = filtered.filter(pl.col("race_id") == race_id)
        return filtered

    def _summary(
        self,
        comparison: pl.DataFrame,
        comparison_id: str,
        cycle: int | None,
        office_type: str | None,
        race_id: str | None,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "comparison_id": comparison_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "filters": {
                "cycle": cycle,
                "office_type": office_type,
                "race_id": race_id,
            },
            "row_count": comparison.height,
            "race_count": comparison["race_id"].n_unique()
            if "race_id" in comparison.columns
            else 0,
        }
        if comparison.is_empty():
            return {
                **base,
                "winner_accuracy": None,
                "mean_absolute_vote_share_error": None,
                "brier_score": None,
                "upset_count": 0,
            }
        actual_winners = comparison.filter(pl.col("actual_winner")).select(
            ["race_id", pl.col("option_id").alias("actual_winner_option_id")]
        )
        predicted_winners = comparison.filter(pl.col("predicted_winner")).select(
            ["race_id", pl.col("option_id").alias("predicted_winner_option_id")]
        )
        winner_eval = predicted_winners.join(actual_winners, on="race_id", how="inner")
        winner_accuracy = (
            winner_eval.select(
                (pl.col("predicted_winner_option_id") == pl.col("actual_winner_option_id")).mean()
            ).item()
            if not winner_eval.is_empty()
            else None
        )
        actual_winner_rows = comparison.filter(pl.col("actual_winner"))
        upset_count = actual_winner_rows.filter(pl.col("winner_probability") < 0.5).height
        return {
            **base,
            "winner_accuracy": None if winner_accuracy is None else float(winner_accuracy),
            "mean_absolute_vote_share_error": self._mean_or_none(
                comparison, "absolute_vote_share_error"
            ),
            "brier_score": self._mean_or_none(comparison, "brier_contribution"),
            "upset_count": upset_count,
        }

    @staticmethod
    def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
        if frame.is_empty() or column not in frame.columns:
            return None
        value = frame.select(pl.col(column).mean()).item()
        return None if value is None else float(value)

    def _write_plots(
        self, comparison: pl.DataFrame, output_dir: Path
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {"comparison": []}
        self._add_plot(
            manifest,
            self._vote_share_plot(comparison, plot_dir),
            "Forecast vote share versus actual vote share",
        )
        self._add_plot(
            manifest,
            self._winner_probability_plot(comparison, plot_dir),
            "Winner probability versus actual outcome",
        )
        return manifest

    @staticmethod
    def _add_plot(manifest: dict[str, list[dict[str, str]]], path: Path | None, title: str) -> None:
        if path is None:
            return
        manifest["comparison"].append({"title": title, "path": f"plots/{path.name}"})

    def _vote_share_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = comparison.filter(pl.col("vote_share_mean").is_not_null())
        if frame.is_empty():
            return None
        actual = frame["actual_vote_share"].to_numpy()
        predicted = frame["vote_share_mean"].to_numpy()
        labels = frame["option_id"].to_list()
        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        ax.scatter(actual, predicted, color="#4c78a8", s=70)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        for x_value, y_value, label in zip(actual, predicted, labels, strict=True):
            ax.annotate(str(label).split("-")[-1], (x_value, y_value), fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Actual vote share")
        ax.set_ylabel("Forecast mean vote share")
        ax.set_title("Forecast vs Actual Vote Share")
        return self._save(fig, plot_dir / "vote_share_forecast_vs_actual.png")

    def _winner_probability_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = comparison.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        x_values = frame["winner_probability"].to_numpy()
        y_values = frame["actual_winner"].cast(pl.Int8).to_numpy()
        colors = np.where(y_values == 1, "#59a14f", "#e15759")
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.scatter(x_values, y_values, color=colors, s=80)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("Forecast winner probability")
        ax.set_ylabel("Actual winner")
        ax.set_yticks([0, 1])
        ax.set_title("Winner Probability vs Actual Outcome")
        return self._save(fig, plot_dir / "winner_probability_vs_actual.png")

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _html_report(summary: dict[str, Any], comparison: pl.DataFrame) -> str:
        rows = ""
        if not comparison.is_empty():
            for row in comparison.select(
                [
                    "race_id",
                    "option_id",
                    "winner_probability",
                    "vote_share_mean",
                    "actual_vote_share",
                    "actual_winner",
                    "predicted_winner",
                    "absolute_vote_share_error",
                ]
            ).iter_rows(named=True):
                rows += (
                    "<tr>"
                    + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values())
                    + "</tr>"
                )
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Result Comparison</title></head>
<body>
<h1>Result Comparison: {html.escape(str(summary["comparison_id"]))}</h1>
<h2>Summary</h2>
<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
<h2>Plots</h2>
<figure><img src="plots/vote_share_forecast_vs_actual.png" width="800"
alt="Forecast vote share versus actual vote share"></figure>
<figure><img src="plots/winner_probability_vs_actual.png" width="800"
alt="Winner probability versus actual outcome"></figure>
<h2>Rows</h2>
<table>
<thead><tr><th>Race</th><th>Option</th><th>Win Prob</th><th>Forecast Share</th>
<th>Actual Share</th><th>Actual Winner</th><th>Predicted Winner</th><th>Abs Error</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>
"""

    @staticmethod
    def _narrative(summary: dict[str, Any], comparison: pl.DataFrame) -> str:
        if comparison.is_empty():
            return (
                "# Forecast Comparison Narrative\n\n"
                "No matching forecast and result rows were found.\n"
            )
        largest_error = (
            comparison.sort("absolute_vote_share_error", descending=True)
            .select(["race_id", "option_id", "absolute_vote_share_error"])
            .row(0, named=True)
        )
        misses = comparison.filter(pl.col("actual_winner") & ~pl.col("predicted_winner"))
        miss_text = (
            "No winner misses among matched races."
            if misses.is_empty()
            else "Missed winners: "
            + ", ".join(
                f"{row['race_id']} ({row['option_id']})" for row in misses.iter_rows(named=True)
            )
            + "."
        )
        return f"""# Forecast Comparison Narrative

- Compared races: `{summary["race_count"]}`
- Matched rows: `{summary["row_count"]}`
- Winner accuracy: `{summary["winner_accuracy"]}`
- Mean absolute vote-share error: `{summary["mean_absolute_vote_share_error"]}`
- Brier score: `{summary["brier_score"]}`
- Upset count: `{summary["upset_count"]}`

{miss_text}

Largest vote-share error: `{largest_error["race_id"]}` / `{largest_error["option_id"]}`.

Absolute error: `{largest_error["absolute_vote_share_error"]}`.
"""
