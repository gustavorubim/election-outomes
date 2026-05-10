from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from election_outcomes.reports._style import (
    NEUTRAL,
    SIZE_PANEL,
    apply_rcparams,
    party_color,
    report_css,
    style_axis,
)
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
        insight_artifacts = self._write_insight_tables(comparison, output_dir)
        plot_manifest = self._write_plots(comparison, output_dir)
        summary["insight_artifacts"] = insight_artifacts
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
                "seats",
                "control_body",
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
        for column in ("vote_share_mean", "winner_probability"):
            if column not in comparison.columns:
                comparison = comparison.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        comparison = comparison.with_columns(
            pl.col("vote_share_mean").cast(pl.Float64),
            pl.col("winner_probability").cast(pl.Float64),
        )
        comparison = comparison.with_columns(
            (pl.col("vote_share_mean") - pl.col("actual_vote_share")).alias("vote_share_error"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share"))
            .abs()
            .alias("absolute_vote_share_error"),
            (pl.col("winner_probability") - pl.col("actual_winner").cast(pl.Float64))
            .pow(2)
            .alias("brier_contribution"),
        )
        race_outcomes = self._race_outcome_frame(comparison)
        outcome_columns = [
            column
            for column in race_outcomes.columns
            if column
            in {
                "race_id",
                "predicted_winner_option_id",
                "predicted_winner_name",
                "predicted_winner_party",
                "predicted_winner_probability",
                "actual_winner_option_id",
                "actual_winner_name",
                "actual_winner_party",
                "actual_winner_probability",
                "race_winner_correct",
            }
        ]
        comparison = comparison.join(
            race_outcomes.select(outcome_columns), on="race_id", how="left"
        )
        return comparison.with_columns(
            (pl.col("option_id") == pl.col("predicted_winner_option_id")).alias("predicted_winner")
        )

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
                "state_accuracy": None,
                "state_accuracy_n": 0,
                "ec_winner_accuracy": None,
                "electoral_college": self._empty_electoral_college_summary(),
                "mean_absolute_vote_share_error": None,
                "brier_score": None,
                "upset_count": 0,
                "actual_winner_probabilities": [],
                "largest_misses": [],
                "race_outcomes": [],
            }
        race_outcomes = self._race_outcome_frame(comparison)
        winner_accuracy = self._mean_bool_or_none(race_outcomes, "race_winner_correct")
        presidential_states = race_outcomes.filter(
            (pl.col("office_type") == "president") & (pl.col("geography_type") == "state")
        )
        state_accuracy = self._mean_bool_or_none(presidential_states, "race_winner_correct")
        electoral_college = self._electoral_college_summary(presidential_states)
        actual_winner_rows = comparison.filter(pl.col("actual_winner"))
        upset_count = actual_winner_rows.filter(pl.col("winner_probability") < 0.5).height
        return {
            **base,
            "winner_accuracy": None if winner_accuracy is None else float(winner_accuracy),
            "state_accuracy": None if state_accuracy is None else float(state_accuracy),
            "state_accuracy_n": presidential_states.height,
            "ec_winner_accuracy": electoral_college["winner_accuracy"],
            "electoral_college": electoral_college,
            "mean_absolute_vote_share_error": self._mean_or_none(
                comparison, "absolute_vote_share_error"
            ),
            "brier_score": self._mean_or_none(comparison, "brier_contribution"),
            "upset_count": upset_count,
            "actual_winner_probabilities": self._actual_winner_probabilities(race_outcomes),
            "largest_misses": self._largest_misses(comparison),
            "race_outcomes": self._json_records(race_outcomes),
        }

    def _write_insight_tables(self, comparison: pl.DataFrame, output_dir: Path) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        race_outcomes = self._race_outcome_frame(comparison)
        if not race_outcomes.is_empty():
            write_parquet(race_outcomes, output_dir / "race_outcomes.parquet")
            artifacts["race_outcomes"] = "race_outcomes.parquet"
        largest_misses = self._largest_miss_frame(comparison, limit=25)
        if not largest_misses.is_empty():
            write_parquet(largest_misses, output_dir / "largest_misses.parquet")
            artifacts["largest_misses"] = "largest_misses.parquet"
        return artifacts

    @staticmethod
    def _race_outcome_frame(comparison: pl.DataFrame) -> pl.DataFrame:
        schema = {
            "race_id": pl.Utf8,
            "cycle": pl.Int64,
            "geography_type": pl.Utf8,
            "geography": pl.Utf8,
            "office_type": pl.Utf8,
            "race_type": pl.Utf8,
            "seats": pl.Int64,
            "control_body": pl.Utf8,
            "predicted_winner_option_id": pl.Utf8,
            "predicted_winner_name": pl.Utf8,
            "predicted_winner_party": pl.Utf8,
            "predicted_winner_probability": pl.Float64,
            "actual_winner_option_id": pl.Utf8,
            "actual_winner_name": pl.Utf8,
            "actual_winner_party": pl.Utf8,
            "actual_winner_probability": pl.Float64,
            "race_winner_correct": pl.Boolean,
        }
        if comparison.is_empty() or "race_id" not in comparison.columns:
            return pl.DataFrame(schema=schema)

        name_expr = pl.col("name") if "name" in comparison.columns else pl.lit(None)
        party_expr = pl.col("party") if "party" in comparison.columns else pl.lit(None)
        base_columns = [
            column
            for column in [
                "race_id",
                "cycle",
                "geography_type",
                "geography",
                "office_type",
                "race_type",
                "seats",
                "control_body",
            ]
            if column in comparison.columns
        ]
        base = (
            comparison.sort(["race_id", "option_id"])
            .group_by("race_id", maintain_order=True)
            .head(1)
            .select(base_columns)
        )
        forecast_candidates = comparison.filter(pl.col("winner_probability").is_not_null())
        if forecast_candidates.is_empty():
            predicted = base.select("race_id").with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("predicted_winner_option_id"),
                pl.lit(None, dtype=pl.Utf8).alias("predicted_winner_name"),
                pl.lit(None, dtype=pl.Utf8).alias("predicted_winner_party"),
                pl.lit(None, dtype=pl.Float64).alias("predicted_winner_probability"),
            )
        else:
            sorted_forecast = forecast_candidates.sort(
                ["race_id", "winner_probability", "option_id"],
                descending=[False, True, False],
            )
            predicted = (
                sorted_forecast.group_by("race_id", maintain_order=True)
                .head(1)
                .select(
                    [
                        "race_id",
                        pl.col("option_id").alias("predicted_winner_option_id"),
                        name_expr.alias("predicted_winner_name"),
                        party_expr.alias("predicted_winner_party"),
                        pl.col("winner_probability").alias("predicted_winner_probability"),
                    ]
                )
            )
        actual = (
            comparison.filter(pl.col("actual_winner"))
            .sort(["race_id", "option_id"])
            .group_by("race_id", maintain_order=True)
            .head(1)
            .select(
                [
                    "race_id",
                    pl.col("option_id").alias("actual_winner_option_id"),
                    name_expr.alias("actual_winner_name"),
                    party_expr.alias("actual_winner_party"),
                    pl.col("winner_probability").alias("actual_winner_probability"),
                ]
            )
        )
        outcome = (
            base.join(predicted, on="race_id", how="left")
            .join(actual, on="race_id", how="left")
            .with_columns(
                pl.when(
                    pl.col("predicted_winner_option_id").is_not_null()
                    & pl.col("actual_winner_option_id").is_not_null()
                )
                .then(pl.col("predicted_winner_option_id") == pl.col("actual_winner_option_id"))
                .otherwise(None)
                .alias("race_winner_correct")
            )
        )
        for column, dtype in schema.items():
            if column not in outcome.columns:
                outcome = outcome.with_columns(pl.lit(None, dtype=dtype).alias(column))
        return outcome.select(list(schema))

    @staticmethod
    def _largest_miss_frame(comparison: pl.DataFrame, limit: int = 10) -> pl.DataFrame:
        columns = [
            "race_id",
            "geography",
            "office_type",
            "option_id",
            "name",
            "party",
            "winner_probability",
            "actual_winner_probability",
            "vote_share_mean",
            "actual_vote_share",
            "vote_share_error",
            "absolute_vote_share_error",
            "actual_winner",
            "predicted_winner",
            "race_winner_correct",
        ]
        present = [column for column in columns if column in comparison.columns]
        if comparison.is_empty() or "absolute_vote_share_error" not in comparison.columns:
            return pl.DataFrame()
        frame = comparison.filter(pl.col("absolute_vote_share_error").is_not_null())
        if frame.is_empty():
            return pl.DataFrame()
        return frame.sort("absolute_vote_share_error", descending=True).head(limit).select(present)

    @classmethod
    def _largest_misses(cls, comparison: pl.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
        return cls._json_records(cls._largest_miss_frame(comparison, limit=limit))

    @classmethod
    def _actual_winner_probabilities(cls, race_outcomes: pl.DataFrame) -> list[dict[str, Any]]:
        columns = [
            "race_id",
            "geography",
            "office_type",
            "seats",
            "actual_winner_option_id",
            "actual_winner_name",
            "actual_winner_party",
            "actual_winner_probability",
            "race_winner_correct",
        ]
        present = [column for column in columns if column in race_outcomes.columns]
        if race_outcomes.is_empty():
            return []
        return cls._json_records(race_outcomes.select(present).sort("race_id"))

    @classmethod
    def _electoral_college_summary(cls, race_outcomes: pl.DataFrame) -> dict[str, Any]:
        if race_outcomes.is_empty() or "seats" not in race_outcomes.columns:
            return cls._empty_electoral_college_summary()
        frame = race_outcomes.filter(
            (pl.col("seats").is_not_null())
            & (pl.col("seats") > 0)
            & pl.col("actual_winner_party").is_not_null()
            & pl.col("predicted_winner_party").is_not_null()
        )
        if frame.is_empty():
            return cls._empty_electoral_college_summary()
        modeled_votes = int(frame["seats"].sum())
        full_ec = modeled_votes >= 270
        threshold = 270.0 if full_ec else modeled_votes / 2.0
        predicted_counts = cls._party_vote_counts(frame, "predicted_winner_party")
        actual_counts = cls._party_vote_counts(frame, "actual_winner_party")
        predicted_winner = cls._party_with_most_votes(predicted_counts, threshold, full_ec)
        actual_winner = cls._party_with_most_votes(actual_counts, threshold, full_ec)
        winner_correct = (
            None
            if predicted_winner is None or actual_winner is None
            else predicted_winner == actual_winner
        )
        return {
            "available": True,
            "scope": "full_electoral_college" if full_ec else "modeled_state_slice",
            "modeled_electoral_votes": modeled_votes,
            "state_count": frame.height,
            "threshold": threshold,
            "winner_accuracy": None if winner_correct is None else float(winner_correct),
            "winner_correct": winner_correct,
            "predicted_winner_party": predicted_winner,
            "actual_winner_party": actual_winner,
            "predicted_party_electoral_votes": predicted_counts,
            "actual_party_electoral_votes": actual_counts,
        }

    @staticmethod
    def _empty_electoral_college_summary() -> dict[str, Any]:
        return {
            "available": False,
            "scope": "not_applicable",
            "modeled_electoral_votes": 0,
            "state_count": 0,
            "threshold": None,
            "winner_accuracy": None,
            "winner_correct": None,
            "predicted_winner_party": None,
            "actual_winner_party": None,
            "predicted_party_electoral_votes": [],
            "actual_party_electoral_votes": [],
        }

    @classmethod
    def _party_vote_counts(cls, frame: pl.DataFrame, party_column: str) -> list[dict[str, Any]]:
        counts = (
            frame.group_by(party_column)
            .agg(pl.col("seats").sum().alias("electoral_votes"))
            .rename({party_column: "party"})
            .sort("electoral_votes", descending=True)
        )
        return cls._json_records(counts)

    @staticmethod
    def _party_with_most_votes(
        counts: list[dict[str, Any]], threshold: float, require_threshold: bool
    ) -> str | None:
        if not counts:
            return None
        top = counts[0]
        votes = float(top.get("electoral_votes") or 0)
        if len(counts) > 1 and votes == float(counts[1].get("electoral_votes") or 0):
            return None
        if require_threshold and votes < threshold:
            return None
        return str(top.get("party"))

    @staticmethod
    def _mean_bool_or_none(frame: pl.DataFrame, column: str) -> float | None:
        if frame.is_empty() or column not in frame.columns:
            return None
        value = frame.select(pl.col(column).cast(pl.Float64).mean()).item()
        return None if value is None else float(value)

    @classmethod
    def _json_records(cls, frame: pl.DataFrame) -> list[dict[str, Any]]:
        return [
            {key: cls._json_value(value) for key, value in row.items()} for row in frame.to_dicts()
        ]

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
        if frame.is_empty() or column not in frame.columns:
            return None
        value = frame.select(pl.col(column).mean()).item()
        return None if value is None else float(value)

    def _write_plots(
        self, comparison: pl.DataFrame, output_dir: Path
    ) -> dict[str, list[dict[str, str]]]:
        apply_rcparams()
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
        self._add_plot(
            manifest,
            self._actual_winner_probability_plot(comparison, plot_dir),
            "Actual-winner probability histogram",
        )
        self._add_plot(
            manifest,
            self._actual_winner_swarm_plot(comparison, plot_dir),
            "Actual-winner probability swarm",
        )
        self._add_plot(
            manifest,
            self._largest_misses_plot(comparison, plot_dir),
            "Largest vote-share misses",
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
        colors = [
            party_color(row.get("party"), default="#4c78a8") for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        ax.scatter(
            actual,
            predicted,
            color=colors,
            s=34,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL["muted"], linewidth=1)
        label_frame = frame.sort("absolute_vote_share_error", descending=True).head(8)
        for row in label_frame.iter_rows(named=True):
            ax.annotate(
                str(row.get("geography") or row.get("race_id") or row.get("option_id")),
                (float(row["actual_vote_share"]), float(row["vote_share_mean"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=NEUTRAL["muted"],
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Actual vote share")
        ax.set_ylabel("Forecast mean vote share")
        ax.set_title("Forecast vs Actual Vote Share")
        style_axis(ax)
        return self._save(fig, plot_dir / "vote_share_forecast_vs_actual.png")

    def _winner_probability_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = comparison.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        x_values = frame["winner_probability"].to_numpy()
        y_values = frame["actual_winner"].cast(pl.Int8).to_numpy()
        jitter = np.linspace(-0.055, 0.055, len(y_values)) if len(y_values) else np.array([])
        if len(jitter):
            jitter = np.roll(jitter, len(jitter) // 3)
        colors = np.where(y_values == 1, NEUTRAL["win"], NEUTRAL["loss"])
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        ax.scatter(
            x_values,
            y_values + jitter,
            color=colors,
            s=34,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.axvline(0.5, color=NEUTRAL["muted"], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("Forecast winner probability")
        ax.set_yticks([0, 1], ["Lost", "Won"])
        ax.set_title("Winner Probability vs Actual Outcome")
        style_axis(ax)
        return self._save(fig, plot_dir / "winner_probability_vs_actual.png")

    def _actual_winner_probability_plot(
        self, comparison: pl.DataFrame, plot_dir: Path
    ) -> Path | None:
        race_outcomes = self._race_outcome_frame(comparison)
        if race_outcomes.is_empty():
            return None
        frame = race_outcomes.filter(pl.col("actual_winner_probability").is_not_null()).sort(
            "actual_winner_probability"
        )
        if frame.is_empty():
            return None
        correct = frame.filter(pl.col("race_winner_correct"))[
            "actual_winner_probability"
        ].to_numpy()
        missed = frame.filter(~pl.col("race_winner_correct"))[
            "actual_winner_probability"
        ].to_numpy()
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        ax.hist(
            [correct, missed],
            bins=np.linspace(0, 1, 11),
            stacked=True,
            color=[NEUTRAL["win"], NEUTRAL["loss"]],
            label=["Called winner", "Missed winner"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(0.5, color=NEUTRAL["muted"], linestyle="--", linewidth=1)
        callout_lines = [
            f"{row.get('geography') or row['race_id']}: "
            f"{float(row['actual_winner_probability']):.0%}"
            for row in frame.head(8).iter_rows(named=True)
        ]
        if callout_lines:
            ax.text(
                0.03,
                0.95,
                "Lowest-confidence winners\n" + "\n".join(callout_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=NEUTRAL["ink"],
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#ffffff",
                    "edgecolor": NEUTRAL["rule"],
                    "alpha": 0.92,
                },
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Forecast probability assigned to actual winner")
        ax.set_ylabel("Race count")
        ax.set_title("Actual-Winner Probability Distribution")
        ax.legend(loc="upper right")
        style_axis(ax)
        return self._save(fig, plot_dir / "actual_winner_probabilities.png")

    def _actual_winner_swarm_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        race_outcomes = self._race_outcome_frame(comparison)
        if race_outcomes.is_empty():
            return None
        frame = race_outcomes.filter(pl.col("actual_winner_probability").is_not_null()).sort(
            "actual_winner_probability"
        )
        if frame.is_empty():
            return None
        values = frame["actual_winner_probability"].to_numpy()
        y_values = np.zeros_like(values, dtype=float)
        if values.size:
            y_values = ((np.arange(values.size) % 17) - 8) / 120.0
        colors = [
            NEUTRAL["win"] if bool(row["race_winner_correct"]) else NEUTRAL["loss"]
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        ax.scatter(
            values,
            y_values,
            color=colors,
            s=26,
            alpha=0.74,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.axvline(0.5, color=NEUTRAL["muted"], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 0.12)
        ax.set_yticks([])
        ax.set_xlabel("Forecast probability assigned to actual winner")
        ax.set_title("Actual-Winner Probability Swarm")
        style_axis(ax, grid_axis="x")
        return self._save(fig, plot_dir / "actual_winner_probability_swarm.png")

    def _largest_misses_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = self._largest_miss_frame(comparison, limit=12)
        if frame.is_empty():
            return None
        frame = frame.sort("absolute_vote_share_error")
        labels = [
            f"{row.get('name') or row['option_id']}\n{row['race_id']}"
            for row in frame.iter_rows(named=True)
        ]
        values = frame["absolute_vote_share_error"].to_list()
        colors = [
            "#e15759" if bool(row.get("actual_winner")) else "#4c78a8"
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(float(value) + 0.002, idx, f"{float(value):.1%}", va="center", fontsize=9)
        ax.set_xlabel("Absolute vote-share error")
        ax.set_title("Largest Vote-Share Misses")
        style_axis(ax, grid_axis="x")
        return self._save(fig, plot_dir / "largest_vote_share_misses.png")

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _html_report(summary: dict[str, Any], comparison: pl.DataFrame) -> str:
        row_columns = [
            "race_id",
            "option_id",
            "winner_probability",
            "actual_winner_probability",
            "vote_share_mean",
            "actual_vote_share",
            "actual_winner",
            "predicted_winner",
            "race_winner_correct",
            "absolute_vote_share_error",
        ]
        present = [column for column in row_columns if column in comparison.columns]
        option_rows = ""
        if not comparison.is_empty():
            for row in comparison.select(present).iter_rows(named=True):
                option_rows += (
                    "<tr>"
                    + "".join(
                        f"<td>{html.escape(ResultComparator._format_cell(value))}</td>"
                        for value in row.values()
                    )
                    + "</tr>"
                )
        header_cells = "".join(
            f"<th>{html.escape(column.replace('_', ' ').title())}</th>" for column in present
        )
        race_outcomes = ResultComparator._race_outcome_frame(comparison)
        race_rows = ""
        if not race_outcomes.is_empty():
            for row in race_outcomes.sort(["office_type", "geography", "race_id"]).iter_rows(
                named=True
            ):
                race_rows += ResultComparator._race_outcome_row(row)
        miss_rows = "".join(
            ResultComparator._largest_miss_row(row) for row in summary.get("largest_misses", [])
        )
        filters = summary.get("filters", {})
        filter_parts = [
            f"{key.replace('_', ' ')}: {value}"
            for key, value in filters.items()
            if value is not None
        ]
        filter_text = " | ".join(filter_parts) if filter_parts else "all configured races"
        metrics = "\n".join(
            [
                ResultComparator._metric_card(
                    "Races",
                    ResultComparator._format_number(summary.get("race_count"), digits=0),
                    f"{ResultComparator._format_number(summary.get('row_count'), digits=0)} rows",
                ),
                ResultComparator._metric_card(
                    "Winner accuracy",
                    ResultComparator._format_pct(summary.get("winner_accuracy")),
                    "race-level called winner",
                ),
                ResultComparator._metric_card(
                    "Mean abs. error",
                    ResultComparator._format_pct(summary.get("mean_absolute_vote_share_error")),
                    "option vote-share error",
                ),
                ResultComparator._metric_card(
                    "Brier score",
                    ResultComparator._format_number(summary.get("brier_score"), digits=3),
                    "winner-probability loss",
                ),
                ResultComparator._metric_card(
                    "Upsets",
                    ResultComparator._format_number(summary.get("upset_count"), digits=0),
                    "actual winners below 50%",
                ),
            ]
        )
        plot_figures = ResultComparator._plot_figures(summary)
        summary_json = html.escape(json.dumps(summary, indent=2, sort_keys=True))
        option_table = (
            f"""
<div class="table-shell audit-table">
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>{option_rows}</tbody>
</table>
</div>
"""
            if header_cells
            else '<p class="muted-copy">No option-level rows were matched.</p>'
        )
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Result Comparison</title>
<style>
{report_css()}
{ResultComparator._comparison_css()}
</style>
</head>
<body>
<main class="container result-comparison">
<header class="hero comparison-hero">
<p class="eyebrow">Forecast vs actual</p>
<h1>{html.escape(str(summary["comparison_id"]))}</h1>
<p class="subtitle">
Compared {html.escape(ResultComparator._format_number(summary.get("race_count"), digits=0))}
races across {html.escape(filter_text)} using the forecast snapshot and curated actuals.
</p>
</header>

<section class="result-band metric-band" aria-label="Comparison summary">
<div class="metric-grid">
{metrics}
</div>
</section>

<section class="result-band">
<div class="band-header">
<div>
<p class="eyebrow">Model behavior</p>
<h2>Calibration And Miss Patterns</h2>
</div>
<p class="band-note">
Distribution, swarm, and miss panels show calibration and error concentration.
</p>
</div>
<div class="result-plot-grid">
{plot_figures}
</div>
</section>

<section class="result-band">
<div class="band-header">
<div>
<p class="eyebrow">Race detail</p>
<h2>Race-By-Race Outcomes</h2>
</div>
<p class="band-note">Sorted by office, geography, and race id.</p>
</div>
<div class="table-shell race-table">
<table>
<thead>
<tr>
<th>Race</th>
<th>Predicted winner</th>
<th>Actual winner</th>
<th>Result</th>
</tr>
</thead>
<tbody>{race_rows}</tbody>
</table>
</div>
</section>

<section class="result-band">
<div class="band-header">
<div>
<p class="eyebrow">Where to inspect first</p>
<h2>Largest Vote-Share Misses</h2>
</div>
</div>
<div class="table-shell compact-table">
<table>
<thead>
<tr>
<th>Race</th>
<th>Option</th>
<th>Party</th>
<th>Forecast</th>
<th>Actual</th>
<th>Error</th>
</tr>
</thead>
<tbody>{miss_rows}</tbody>
</table>
</div>
</section>

<section class="result-band audit-band">
<details>
<summary>Summary JSON</summary>
<pre>{summary_json}</pre>
</details>
<details>
<summary>Option-level comparison rows</summary>
{option_table}
</details>
</section>
</main>
</body>
</html>
"""

    @staticmethod
    def _plot_figures(summary: dict[str, Any]) -> str:
        figures = []
        for entry in summary.get("plot_manifest", {}).get("comparison", []):
            title = str(entry["title"])
            path = str(entry["path"])
            css_class = "plot-card"
            if "Largest" in title:
                css_class += " plot-card-wide"
            figures.append(
                f'<figure class="{css_class}">'
                f'<img src="{html.escape(path)}" alt="{html.escape(title)}" loading="lazy">'
                f"<figcaption>{html.escape(title)}</figcaption>"
                "</figure>"
            )
        return "\n".join(figures)

    @staticmethod
    def _metric_card(label: str, value: str, detail: str) -> str:
        return (
            '<article class="metric-card">'
            f'<span class="label">{html.escape(label)}</span>'
            f'<strong class="value">{html.escape(value)}</strong>'
            f'<span class="detail">{html.escape(detail)}</span>'
            "</article>"
        )

    @staticmethod
    def _race_outcome_row(row: dict[str, Any]) -> str:
        geography = row.get("geography") or row.get("race_id")
        race_meta = " | ".join(
            str(value)
            for value in [row.get("office_type"), row.get("geography_type"), row.get("seats")]
            if value not in {None, ""}
        )
        correct = bool(row.get("race_winner_correct"))
        result_class = "win" if correct else "loss"
        result_text = "Correct" if correct else "Miss"
        return (
            "<tr>"
            "<td>"
            f"<strong>{html.escape(str(geography))}</strong>"
            f'<span class="cell-note">{html.escape(str(row.get("race_id") or ""))}</span>'
            f'<span class="cell-note">{html.escape(race_meta)}</span>'
            "</td>"
            f"<td>{ResultComparator._winner_label(row, 'predicted')}</td>"
            f"<td>{ResultComparator._winner_label(row, 'actual')}</td>"
            f'<td><span class="pill {result_class}">{result_text}</span></td>'
            "</tr>"
        )

    @staticmethod
    def _largest_miss_row(row: dict[str, Any]) -> str:
        race = row.get("geography") or row.get("race_id")
        option = row.get("name") or row.get("option_id")
        return (
            "<tr>"
            f"<td>{html.escape(str(race))}</td>"
            f"<td>{html.escape(str(option))}</td>"
            f"<td>{ResultComparator._party_pill(row.get('party'))}</td>"
            f"<td>{html.escape(ResultComparator._format_pct(row.get('vote_share_mean')))}</td>"
            f"<td>{html.escape(ResultComparator._format_pct(row.get('actual_vote_share')))}</td>"
            "<td>"
            f"{html.escape(ResultComparator._format_pct(row.get('absolute_vote_share_error')))}"
            "</td>"
            "</tr>"
        )

    @staticmethod
    def _winner_label(row: dict[str, Any], prefix: str) -> str:
        name = row.get(f"{prefix}_winner_name") or row.get(f"{prefix}_winner_option_id") or "n/a"
        party = row.get(f"{prefix}_winner_party")
        probability = ResultComparator._format_pct(row.get(f"{prefix}_winner_probability"))
        return (
            '<div class="winner-cell">'
            f"<strong>{html.escape(str(name))}</strong>"
            f"{ResultComparator._party_pill(party)}"
            f'<span class="cell-note">{html.escape(probability)} probability</span>'
            "</div>"
        )

    @staticmethod
    def _party_pill(party: Any) -> str:
        if party is None:
            return '<span class="pill neutral">n/a</span>'
        party_text = str(party)
        party_class = party_text.lower() if party_text.upper() in {"DEM", "REP"} else "neutral"
        return f'<span class="pill {party_class}">{html.escape(party_text)}</span>'

    @staticmethod
    def _format_pct(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(numeric):
            return "n/a"
        return f"{numeric:.1%}"

    @staticmethod
    def _format_number(value: Any, *, digits: int = 1) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(numeric):
            return "n/a"
        if digits == 0:
            return f"{numeric:,.0f}"
        return f"{numeric:,.{digits}f}"

    @staticmethod
    def _comparison_css() -> str:
        return """
.result-comparison { max-width: 1180px; }
.comparison-hero { margin-bottom: 20px; }
.result-band { margin: 24px 0 34px; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 15px 16px;
  min-width: 0;
}
.metric-card .label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 700;
}
.metric-card .value {
  display: block;
  margin-top: 4px;
  font-size: 27px;
  line-height: 1.05;
  letter-spacing: 0;
}
.metric-card .detail {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 12px;
}
.band-header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: end;
  margin-bottom: 12px;
}
.band-header h2 { margin: 0; }
.band-note {
  color: var(--muted);
  font-size: 13px;
  max-width: 330px;
  margin: 0;
  text-align: right;
}
.result-plot-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.plot-card {
  margin: 0;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 12px 12px 14px;
}
.plot-card-wide { grid-column: 1 / -1; }
.plot-card img {
  width: 100%;
  max-height: 430px;
  object-fit: contain;
  display: block;
  background: #fff;
  border-radius: 4px;
}
.plot-card-wide img { max-height: 480px; }
.plot-card figcaption {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
}
.table-shell {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  overflow: auto;
}
.race-table { max-height: 520px; }
.compact-table { max-height: 360px; }
.audit-table { max-height: 440px; margin-top: 12px; }
.table-shell thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--card);
}
.winner-cell {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
}
.winner-cell strong { min-width: 120px; }
.cell-note {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.muted-copy { color: var(--muted); }
.audit-band details {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.audit-band summary {
  cursor: pointer;
  font-weight: 700;
}
.audit-band pre {
  margin: 12px 0 0;
  overflow-x: auto;
  background: #f0eee6;
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 12px;
  font-size: 12px;
}
@media (max-width: 980px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .result-plot-grid { grid-template-columns: 1fr; }
  .plot-card-wide { grid-column: auto; }
}
@media (max-width: 640px) {
  .metric-grid { grid-template-columns: 1fr; }
  .band-header { display: block; }
  .band-note { text-align: left; margin-top: 6px; }
}
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
        actual_probability_rows = ResultComparator._markdown_records(
            summary["actual_winner_probabilities"],
            [
                "race_id",
                "actual_winner_party",
                "actual_winner_probability",
                "race_winner_correct",
            ],
        )
        largest_miss_rows = ResultComparator._markdown_records(
            summary["largest_misses"],
            [
                "race_id",
                "option_id",
                "absolute_vote_share_error",
                "actual_winner",
                "predicted_winner",
            ],
        )
        state_accuracy = summary["state_accuracy"]
        state_count = summary["state_accuracy_n"]
        ec_accuracy = summary["ec_winner_accuracy"]
        ec_scope = summary["electoral_college"]["scope"]
        return f"""# Forecast Comparison Narrative

- Compared races: `{summary["race_count"]}`
- Matched rows: `{summary["row_count"]}`
- Winner accuracy: `{summary["winner_accuracy"]}`
- Presidential state accuracy: `{state_accuracy}` over `{state_count}` state races
- Electoral College winner accuracy: `{ec_accuracy}` ({ec_scope})
- Mean absolute vote-share error: `{summary["mean_absolute_vote_share_error"]}`
- Brier score: `{summary["brier_score"]}`
- Upset count: `{summary["upset_count"]}`

{miss_text}

Largest vote-share error: `{largest_error["race_id"]}` / `{largest_error["option_id"]}`.

Absolute error: `{largest_error["absolute_vote_share_error"]}`.

Actual-winner probabilities:

{actual_probability_rows}

Largest misses:

{largest_miss_rows}
"""

    @staticmethod
    def _format_cell(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _markdown_records(records: list[dict[str, Any]], columns: list[str]) -> str:
        if not records:
            return "- n/a"
        lines = []
        for record in records:
            parts = [f"{column}={record.get(column)}" for column in columns]
            lines.append("- " + "; ".join(parts))
        return "\n".join(lines)
