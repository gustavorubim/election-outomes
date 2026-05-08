from __future__ import annotations

# ruff: noqa: I001

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PARTY_COLORS = {
    "DEM": "#2f78bc",
    "REP": "#c44e52",
    "YES": "#3a8f5d",
    "NO": "#777777",
}
PAPER_COLOR = "#f6f4ef"
AXIS_COLOR = "#ffffff"
GRID_COLOR = "#ded9cf"
TEXT_COLOR = "#242424"
MUTED_COLOR = "#6b6b6b"
GOLD_COLOR = "#c87922"


class PlotGenerator:
    """Static plot generation for calibration and projection diagnostics."""

    def render_all(
        self,
        artifact_dir: Path,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        control_forecasts: pl.DataFrame,
        ecosystem_forecasts: pl.DataFrame,
        backtest_predictions: pl.DataFrame,
        backtest_payload: dict[str, Any],
        methodology_benchmark: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = artifact_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {
            "calibration": [],
            "projection": [],
            "benchmark": [],
        }
        self._add(
            manifest,
            "calibration",
            self._calibration_curve(plot_dir, backtest_predictions),
            "Calibration curve",
        )
        self._add(
            manifest,
            "calibration",
            self._brier_by_component(plot_dir, backtest_payload),
            "Brier score by model component",
        )
        self._add(
            manifest,
            "calibration",
            self._interval_coverage(plot_dir, backtest_payload),
            "Historical interval coverage",
        )
        self._add(
            manifest,
            "projection",
            self._race_probability_bars(plot_dir, race_forecasts),
            "Winner probabilities by race",
        )
        self._add(
            manifest,
            "projection",
            self._vote_share_intervals(plot_dir, race_forecasts),
            "Projected vote-share intervals",
        )
        self._add(
            manifest,
            "projection",
            self._control_projection(plot_dir, control_forecasts),
            "Seat/control projections",
        )
        self._add(
            manifest,
            "projection",
            self._turnout_and_recount(plot_dir, ecosystem_forecasts),
            "Turnout and recount-risk projections",
        )
        self._add(
            manifest,
            "projection",
            self._tier_coverage(plot_dir, race_catalog),
            "Forecast coverage by tier",
        )
        self._add(
            manifest,
            "projection",
            self._electoral_college_distribution(plot_dir, race_catalog, forecast_draws),
            "Modeled Electoral College distribution",
        )
        self._add(
            manifest,
            "projection",
            self._electoral_college_swarm(plot_dir, race_catalog, forecast_draws),
            "Top-line Electoral College simulation swarm",
        )
        self._add(
            manifest,
            "benchmark",
            self._methodology_benchmark(plot_dir, methodology_benchmark or {}),
            "Silver/FiveThirtyEight methodology benchmark",
        )
        return manifest

    @staticmethod
    def write_manifest(manifest: dict[str, list[dict[str, str]]], artifact_dir: Path) -> None:
        path = artifact_dir / "plot_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _add(
        manifest: dict[str, list[dict[str, str]]],
        category: str,
        path: Path | None,
        title: str,
    ) -> None:
        if path is None:
            return
        manifest[category].append({"title": title, "path": f"plots/{path.name}"})

    def _calibration_curve(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        if frame.is_empty() or "ensemble_probability" not in frame.columns:
            return None
        df = frame.select(["ensemble_probability", "actual_winner"]).drop_nulls()
        if df.is_empty():
            return None
        probability = df["ensemble_probability"].cast(pl.Float64).to_numpy()
        actual = df["actual_winner"].cast(pl.Float64).to_numpy()
        bins = np.linspace(0, 1, 6)
        xs: list[float] = []
        ys: list[float] = []
        for lower, upper in pairwise(bins):
            mask = (probability >= lower) & (
                probability < upper if upper < 1 else probability <= upper
            )
            if np.any(mask):
                xs.append(float(np.mean(probability[mask])))
                ys.append(float(np.mean(actual[mask])))
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
        ax.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.scatter(xs, ys, color=PARTY_COLORS["DEM"], s=72, zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x + 0.025, y, f"{y:.0%}", va="center", fontsize=9, color=TEXT_COLOR)
        ax.text(
            0.02,
            0.95,
            f"n={len(probability)} held-out rows",
            transform=ax.transAxes,
            color=MUTED_COLOR,
            fontsize=10,
            va="top",
        )
        ax.set_xlabel("Mean forecast probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Calibration Curve", loc="left", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax)
        return self._save(fig, plot_dir / "calibration_curve.png")

    def _brier_by_component(self, plot_dir: Path, backtest_payload: dict[str, Any]) -> Path | None:
        metrics = backtest_payload.get("metrics", {})
        rows = [
            (component, values["brier"])
            for component, values in metrics.items()
            if isinstance(values, dict) and "brier" in values
        ]
        if not rows:
            return None
        rows = sorted(rows, key=lambda item: item[1], reverse=True)
        labels, values = zip(*rows, strict=True)
        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
        colors = [PARTY_COLORS["DEM"] if label == "ensemble" else "#8d8d8d" for label in labels]
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=9)
        ax.set_ylabel("Brier score")
        ax.set_xlabel("Lower is better")
        ax.set_title("Backtest Brier Score by Component", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "brier_by_component.png")

    def _interval_coverage(self, plot_dir: Path, backtest_payload: dict[str, Any]) -> Path | None:
        coverage = (
            backtest_payload.get("metrics", {}).get("ensemble", {}).get("interval_90_coverage")
        )
        if coverage is None:
            return None
        observed = float(coverage)
        fig, ax = plt.subplots(figsize=(6.4, 5.1), dpi=150)
        ax.bar(
            ["Nominal 90%", "Observed"],
            [0.9, observed],
            color=["#999999", PARTY_COLORS["DEM"] if observed >= 0.85 else GOLD_COLOR],
        )
        ax.axhline(0.9, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.text(1, min(1.0, observed + 0.04), f"{observed:.0%}", ha="center", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title("Historical Interval Coverage", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax)
        return self._save(fig, plot_dir / "interval_coverage.png")

    def _race_probability_bars(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        frame = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        frame = frame.sort("winner_probability", descending=True)
        labels = [self._option_label(row) for row in frame.iter_rows(named=True)]
        values = frame["winner_probability"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]), "#f28e2b") for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, len(labels) * 0.55)), dpi=150)
        ax.barh(labels, values, color=colors)
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        for idx, value in enumerate(values):
            x = min(0.98, float(value) + 0.025)
            ax.text(x, idx, f"{float(value):.1%}", va="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Winner probability")
        ax.set_title("Race-Level Winner Probabilities", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "race_probability_bars.png")

    def _vote_share_intervals(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        frame = race_forecasts.filter(pl.col("vote_share_mean").is_not_null())
        required = {"vote_share_mean", "vote_share_p05", "vote_share_p95"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        sorted_frame = frame.sort("vote_share_mean", descending=True)
        labels = [self._option_label(row) for row in sorted_frame.iter_rows(named=True)]
        mean = np.array(sorted_frame["vote_share_mean"].to_list())
        low = np.array(sorted_frame["vote_share_p05"].to_list())
        high = np.array(sorted_frame["vote_share_p95"].to_list())
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, len(labels) * 0.55)), dpi=150)
        for idx, row in enumerate(sorted_frame.iter_rows(named=True)):
            color = PARTY_COLORS.get(str(row.get("party")), "#2f4b7c")
            ax.errorbar(
                float(mean[idx]),
                labels[idx],
                xerr=[[float(mean[idx] - low[idx])], [float(high[idx] - mean[idx])]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.4,
                capsize=4,
                markersize=6,
            )
            ax.text(
                min(0.98, float(high[idx]) + 0.012),
                labels[idx],
                f"{float(mean[idx]):.1%}",
                va="center",
                fontsize=10,
                color=TEXT_COLOR,
            )
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        xmin = max(0, min(float(low.min()) - 0.04, 0.48))
        xmax = min(1, max(float(high.max()) + 0.07, 0.52))
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Projected vote share with 90% interval")
        ax.set_title("Vote-Share Projection Intervals", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "vote_share_intervals.png")

    def _control_projection(self, plot_dir: Path, control_forecasts: pl.DataFrame) -> Path | None:
        if control_forecasts.is_empty():
            return None
        labels = [
            f"{row['control_body']} | {row['party']}"
            for row in control_forecasts.iter_rows(named=True)
        ]
        values = control_forecasts["seat_count_mean"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]), "#e15759")
            for row in control_forecasts.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(8.2, max(4.2, len(labels) * 0.6)), dpi=150)
        ax.barh(labels, values, color=colors)
        for idx, row in enumerate(control_forecasts.iter_rows(named=True)):
            value = float(row["seat_count_mean"])
            ax.text(value + 0.15, idx, f"{value:.1f}", va="center", fontsize=10)
        threshold = (
            int(control_forecasts["control_threshold"].max())
            if "control_threshold" in control_forecasts.columns
            else None
        )
        modeled = (
            int(control_forecasts["modeled_seats"].max())
            if "modeled_seats" in control_forecasts.columns
            else None
        )
        if threshold is not None and modeled is not None:
            if threshold <= max(float(max(values)), float(modeled)) * 1.1:
                ax.axvline(threshold, color=TEXT_COLOR, linewidth=1.2)
            else:
                ax.text(
                    0.02,
                    0.94,
                    f"National threshold {threshold} is outside this {modeled}-seat slice.",
                    transform=ax.transAxes,
                    color=GOLD_COLOR,
                    fontsize=10,
                    va="top",
                )
        ax.set_xlabel("Mean projected seats/wins in modeled races")
        ax.set_title("Control Projection", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "control_projection.png")

    def _turnout_and_recount(
        self, plot_dir: Path, ecosystem_forecasts: pl.DataFrame
    ) -> Path | None:
        if ecosystem_forecasts.is_empty():
            return None
        labels = ecosystem_forecasts["race_id"].to_list()
        recount = ecosystem_forecasts["recount_probability"].to_list()
        frame = ecosystem_forecasts.sort("recount_probability", descending=True)
        labels = frame["race_id"].to_list()
        recount = frame["recount_probability"].to_list()
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, len(labels) * 0.55)), dpi=150)
        ax.barh(labels, recount, color="#76b7b2")
        for idx, value in enumerate(recount):
            ax.text(min(0.98, float(value) + 0.02), idx, f"{float(value):.1%}", va="center")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recount probability")
        ax.set_title("Recount Risk by Race", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.text(
            0,
            1.04,
            "This is a close-margin proxy unless a calibrated recount model is available.",
            transform=ax.transAxes,
            color=MUTED_COLOR,
            fontsize=10,
        )
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "turnout_recount_risk.png")

    def _tier_coverage(self, plot_dir: Path, race_catalog: pl.DataFrame) -> Path | None:
        if race_catalog.is_empty():
            return None
        counts = race_catalog.group_by("tier").agg(pl.len().alias("count")).sort("tier")
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=150)
        bars = ax.bar(counts["tier"].to_list(), counts["count"].to_list(), color="#b07aa1")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel("Race count")
        ax.set_title("Forecast Coverage by Tier", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "tier_coverage.png")

    def _electoral_college_distribution(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        parties = seat_counts["party"].unique().sort().to_list()
        modeled_seats = self._modeled_presidential_seats(race_catalog)
        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
        bins: int | np.ndarray = 20
        if modeled_seats <= 80:
            bins = np.arange(-0.5, modeled_seats + 1.5, 1)
        for party in parties:
            values = seat_counts.filter(pl.col("party") == party)["electoral_votes"].to_numpy()
            ax.hist(
                values,
                bins=bins,
                alpha=0.58,
                label=str(party),
                color=PARTY_COLORS.get(str(party), "#999999"),
            )
        threshold = 270 if modeled_seats >= 270 else modeled_seats / 2
        threshold_label = "270 EV threshold" if modeled_seats >= 270 else "modeled-slice majority"
        ax.axvline(threshold, color=TEXT_COLOR, linestyle="--", linewidth=1.2)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.92,
            threshold_label,
            rotation=90,
            va="top",
            ha="right",
            color=TEXT_COLOR,
            fontsize=9,
        )
        if modeled_seats < 270:
            ax.text(
                0.02,
                0.94,
                f"Only {modeled_seats} electoral votes are modeled in this scenario.",
                transform=ax.transAxes,
                color=GOLD_COLOR,
                fontsize=10,
                va="top",
            )
            ax.set_xlim(-0.5, modeled_seats + 0.5)
        ax.set_xlabel("Modeled electoral votes")
        ax.set_ylabel("Draw count")
        ax.set_title("Electoral College Distribution", loc="left", fontweight="bold")
        ax.legend()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "electoral_college_distribution.png")

    def _electoral_college_swarm(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        frame = seat_counts.sort(["party", "draw_id"])
        if frame.height > 100:
            frame = frame.group_by("party", maintain_order=True).head(50)
        modeled_seats = self._modeled_presidential_seats(race_catalog)
        fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=150)
        parties = frame["party"].unique().sort().to_list()
        for party_index, party in enumerate(parties):
            values = frame.filter(pl.col("party") == party)["electoral_votes"].to_numpy()
            rng = np.random.default_rng(20260508 + len(values))
            y = rng.normal(loc=party_index * 0.25, scale=0.045, size=len(values))
            ax.scatter(
                values,
                y,
                s=46,
                alpha=0.72,
                color=PARTY_COLORS.get(str(party), "#999999"),
                edgecolor="white",
                linewidth=0.5,
                label=str(party),
            )
        threshold = 270 if modeled_seats >= 270 else modeled_seats / 2
        ax.axvline(threshold, color=TEXT_COLOR, linewidth=1.2)
        if modeled_seats < 270:
            ax.text(
                0.02,
                0.91,
                "Representative draw swarm for the modeled state slice only.",
                transform=ax.transAxes,
                color=GOLD_COLOR,
                fontsize=10,
                va="top",
            )
            ax.set_xlim(-0.5, modeled_seats + 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Electoral votes in representative simulation draws")
        ax.set_title("Top-Line Simulation Swarm", loc="left", fontweight="bold")
        ax.legend(loc="upper right", frameon=False)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "topline_electoral_swarm.png")

    def _methodology_benchmark(
        self, plot_dir: Path, methodology_benchmark: dict[str, Any]
    ) -> Path | None:
        rows = methodology_benchmark.get("rows", [])
        if not rows:
            return None
        labels = [str(row["dimension"]) for row in rows]
        values = [float(row["score"]) for row in rows]
        fig, ax = plt.subplots(figsize=(9.6, max(4.5, len(labels) * 0.52)), dpi=150)
        colors = [PARTY_COLORS["DEM"] if value >= 0.95 else GOLD_COLOR for value in values]
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(min(0.98, value + 0.025), idx, f"{value:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Methodology parity score")
        ax.set_title(
            "Silver/FiveThirtyEight Methodology Benchmark",
            loc="left",
            fontweight="bold",
        )
        self._style_axis(ax)
        return self._save(fig, plot_dir / "silver_methodology_benchmark.png")

    @staticmethod
    def _option_label(row: dict[str, Any]) -> str:
        party = str(row.get("party") or "")
        name = str(row.get("name") or row.get("option_id"))
        race_id = str(row.get("race_id") or "")
        return f"{name} ({party})\n{race_id}" if party else f"{name}\n{race_id}"

    @staticmethod
    def _modeled_presidential_seats(race_catalog: pl.DataFrame) -> int:
        if race_catalog.is_empty() or "seats" not in race_catalog.columns:
            return 0
        president = race_catalog.filter(
            (pl.col("office_type") == "president") & (pl.col("control_body") == "president")
        )
        return 0 if president.is_empty() else int(president["seats"].sum())

    @staticmethod
    def _presidential_seat_counts(
        race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> pl.DataFrame:
        if race_catalog.is_empty() or forecast_draws.is_empty():
            return pl.DataFrame()
        president = race_catalog.filter(
            (pl.col("office_type") == "president") & (pl.col("control_body") == "president")
        )
        if president.is_empty():
            return pl.DataFrame()
        winner_counts = (
            forecast_draws.join(
                president.select(["race_id", "seats"]),
                on="race_id",
                how="inner",
            )
            .filter(pl.col("winner"))
            .group_by(["draw_id", "party"])
            .agg(pl.col("seats").sum().alias("electoral_votes"))
        )
        draws = forecast_draws.select("draw_id").unique()
        parties = forecast_draws.select("party").unique()
        return (
            draws.join(parties, how="cross")
            .join(winner_counts, on=["draw_id", "party"], how="left")
            .with_columns(pl.col("electoral_votes").fill_null(0))
            .sort(["party", "draw_id"])
        )

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor(AXIS_COLOR)
        ax.figure.set_facecolor(PAPER_COLOR)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_fontsize(14)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
