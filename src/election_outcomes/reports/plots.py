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
        poll_trajectory: pl.DataFrame | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = artifact_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {
            "distribution": [],
            "calibration": [],
            "projection": [],
            "trajectory": [],
            "drivers": [],
            "stability": [],
            "model_quality": [],
            "benchmark": [],
        }
        self._add(
            manifest,
            "distribution",
            self._seat_count_histogram(
                plot_dir, race_catalog, forecast_draws, control_forecasts
            ),
            "Seat-count distribution per party",
        )
        self._add(
            manifest,
            "distribution",
            self._race_vote_share_kde_grid(plot_dir, race_forecasts, forecast_draws),
            "Vote-share KDEs for the closest races",
        )
        self._add(
            manifest,
            "drivers",
            self._tipping_point_bars(plot_dir, control_forecasts),
            "Tipping-point race ranking",
        )
        self._add(
            manifest,
            "drivers",
            self._driver_waterfall(plot_dir, race_forecasts),
            "Component drivers for closest races",
        )
        self._add(
            manifest,
            "calibration",
            self._reliability_diagram_with_band(plot_dir, backtest_predictions),
            "Reliability diagram with 95% Wilson band",
        )
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
            "trajectory",
            self._polling_state_trajectory(
                plot_dir, poll_trajectory if poll_trajectory is not None else pl.DataFrame()
            ),
            "Kalman polling trajectories with uncertainty",
        )
        self._add(
            manifest,
            "trajectory",
            self._polling_trajectory(plot_dir, backtest_predictions),
            "Polling trajectory from rolling-origin cuts",
        )
        self._add(
            manifest,
            "stability",
            self._simulation_probability_convergence(plot_dir, forecast_draws),
            "Simulation probability convergence",
        )
        self._add(
            manifest,
            "model_quality",
            self._electoral_college_chain_trace(plot_dir, race_catalog, forecast_draws),
            "Posterior simulation chain traces",
        )
        self._add(
            manifest,
            "model_quality",
            self._kalman_posterior_uncertainty(
                plot_dir, poll_trajectory if poll_trajectory is not None else pl.DataFrame()
            ),
            "Kalman posterior uncertainty",
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

    def _electoral_college_chain_trace(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        parties = seat_counts["party"].unique().sort().to_list()
        if not parties:
            return None
        chain_count = 4
        fig, axes = plt.subplots(
            len(parties),
            1,
            figsize=(9.2, max(3.6, 2.6 * len(parties))),
            dpi=150,
            sharex=True,
        )
        axes_array = np.atleast_1d(axes)
        for axis, party in zip(axes_array, parties, strict=False):
            party_frame = (
                seat_counts.filter(pl.col("party") == party)
                .with_columns((pl.col("draw_id") % chain_count).alias("chain"))
                .sort("draw_id")
            )
            color = PARTY_COLORS.get(str(party), "#999999")
            for chain in range(chain_count):
                values = party_frame.filter(pl.col("chain") == chain)["electoral_votes"].to_numpy()
                if values.size == 0:
                    continue
                axis.plot(
                    np.arange(values.size),
                    values,
                    linewidth=1.1,
                    alpha=0.78,
                    color=color,
                    label=f"chain {chain + 1}",
                )
            mean_value = float(party_frame["electoral_votes"].mean())
            axis.axhline(mean_value, color=TEXT_COLOR, linewidth=1.0, linestyle="--")
            axis.text(
                0.01,
                0.88,
                f"{party} mean={mean_value:.1f} EV",
                transform=axis.transAxes,
                color=TEXT_COLOR,
                fontsize=9,
                va="top",
            )
            axis.set_ylabel("EV")
            self._style_axis(axis)
        axes_array[0].set_title(
            "Posterior Simulation Chain Traces",
            loc="left",
            fontweight="bold",
        )
        axes_array[-1].set_xlabel("Draw index within split chain")
        axes_array[0].text(
            0.01,
            1.04,
            "MCMC-style split chains from posterior simulations, not a separate MCMC sampler.",
            transform=axes_array[0].transAxes,
            color=MUTED_COLOR,
            fontsize=9,
            va="bottom",
        )
        axes_array[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
        return self._save(fig, plot_dir / "electoral_college_chain_traces.png")

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

    def _polling_trajectory(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {"race_id", "option_id", "polls_probability"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        time_column = (
            "as_of_offset_days"
            if "as_of_offset_days" in frame.columns
            else "as_of"
            if "as_of" in frame.columns
            else None
        )
        if time_column is None:
            return None
        plot_frame = frame.filter(pl.col("polls_probability").is_not_null())
        if "actual_winner" in plot_frame.columns:
            plot_frame = plot_frame.filter(pl.col("actual_winner"))
        if plot_frame.is_empty():
            return None
        sort_columns = ["race_id", time_column]
        plot_frame = plot_frame.sort(sort_columns)
        series_keys = (
            plot_frame.select(["race_id", "option_id"])
            .unique(maintain_order=True)
            .head(8)
            .to_dicts()
        )
        fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
        for index, key in enumerate(series_keys):
            series = plot_frame.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            ).sort(time_column)
            if series.height < 1:
                continue
            x_values = series[time_column].to_list()
            y_values = series["polls_probability"].to_list()
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            color = list(PARTY_COLORS.values())[index % len(PARTY_COLORS)]
            ax.plot(x_values, y_values, marker="o", linewidth=2.0, color=color, label=label)
        ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Polling component win probability")
        if time_column == "as_of_offset_days":
            ax.set_xlabel("Days before election")
            ax.invert_xaxis()
        else:
            ax.set_xlabel("As-of date")
        ax.set_title("Polling Probability Trajectory", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=8)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "polling_probability_trajectory.png")

    def _polling_state_trajectory(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {
            "race_id",
            "option_id",
            "trajectory_date",
            "latent_vote_share",
            "latent_sigma",
        }
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        final_rows = (
            frame.sort("trajectory_date")
            .group_by(["race_id", "option_id"], maintain_order=True)
            .tail(1)
            .with_columns((pl.col("latent_vote_share") - 0.5).abs().alias("_distance"))
            .sort("_distance")
            .head(8)
        )
        if final_rows.is_empty():
            return None
        selected = frame.join(
            final_rows.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        ).sort(["race_id", "option_id", "trajectory_date"])
        fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
        colors = list(PARTY_COLORS.values())
        for index, key in enumerate(
            final_rows.select(["race_id", "option_id"]).iter_rows(named=True)
        ):
            series = selected.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            )
            if series.is_empty():
                continue
            x_values = series["trajectory_date"].to_list()
            mean = np.array(series["latent_vote_share"].to_list(), dtype=float)
            sigma = np.array(series["latent_sigma"].to_list(), dtype=float)
            lower = np.clip(mean - 1.645 * sigma, 0.0, 1.0)
            upper = np.clip(mean + 1.645 * sigma, 0.0, 1.0)
            color = colors[index % len(colors)]
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            ax.plot(x_values, mean, linewidth=2.0, color=color, label=label)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.12, linewidth=0)
            if {"poll_count", "mean_observed_share"}.issubset(set(series.columns)):
                poll_rows = series.filter(
                    (pl.col("poll_count") > 0) & pl.col("mean_observed_share").is_not_null()
                )
                if not poll_rows.is_empty():
                    ax.scatter(
                        poll_rows["trajectory_date"].to_list(),
                        poll_rows["mean_observed_share"].to_list(),
                        color=color,
                        edgecolors=AXIS_COLOR,
                        linewidths=0.7,
                        s=32,
                        zorder=4,
                    )
        ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_ylim(0.35, 0.65)
        ax.set_xlabel("Date")
        ax.set_ylabel("Latent vote share")
        ax.set_title("Kalman Polling Trajectories", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=7)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "polling_kalman_trajectories.png")

    def _kalman_posterior_uncertainty(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {"race_id", "option_id", "trajectory_date", "latent_sigma"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        final_rows = (
            frame.sort("trajectory_date")
            .group_by(["race_id", "option_id"], maintain_order=True)
            .tail(1)
            .sort("latent_sigma", descending=True)
            .head(8)
        )
        if final_rows.is_empty():
            return None
        selected = frame.join(
            final_rows.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        ).sort(["race_id", "option_id", "trajectory_date"])
        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
        colors = list(PARTY_COLORS.values())
        for index, key in enumerate(
            final_rows.select(["race_id", "option_id"]).iter_rows(named=True)
        ):
            series = selected.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            )
            if series.is_empty():
                continue
            color = colors[index % len(colors)]
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            ax.plot(
                series["trajectory_date"].to_list(),
                series["latent_sigma"].to_list(),
                linewidth=1.9,
                color=color,
                label=label,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Posterior SD")
        ax.set_title("Kalman Posterior Uncertainty", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=7)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "kalman_posterior_uncertainty.png")

    def _simulation_probability_convergence(
        self, plot_dir: Path, forecast_draws: pl.DataFrame
    ) -> Path | None:
        required = {"draw_id", "race_id", "option_id", "winner"}
        if forecast_draws.is_empty() or not required.issubset(set(forecast_draws.columns)):
            return None
        probabilities = (
            forecast_draws.group_by(["race_id", "option_id"])
            .agg(pl.col("winner").mean().alias("final_probability"))
            .filter(pl.col("final_probability").is_not_null())
            .sort("final_probability", descending=True)
            .head(8)
        )
        if probabilities.is_empty():
            return None
        selected = forecast_draws.join(
            probabilities.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
        for index, row in enumerate(probabilities.iter_rows(named=True)):
            series = selected.filter(
                (pl.col("race_id") == row["race_id"]) & (pl.col("option_id") == row["option_id"])
            ).sort("draw_id")
            wins = series["winner"].cast(pl.Int8).to_numpy()
            if wins.size == 0:
                continue
            cumulative = np.cumsum(wins) / np.arange(1, wins.size + 1)
            draw_ids = np.arange(1, wins.size + 1)
            label = f"{row['race_id']} / {str(row['option_id']).split('-')[-1]}"
            color = list(PARTY_COLORS.values())[index % len(PARTY_COLORS)]
            ax.plot(draw_ids, cumulative, linewidth=1.8, color=color, label=label)
            ax.axhline(float(row["final_probability"]), color=color, alpha=0.22, linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Simulation draws consumed")
        ax.set_ylabel("Cumulative winner probability")
        ax.set_title("Simulation Probability Convergence", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=8)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "simulation_probability_convergence.png")

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

    # -----------------------------------------------------------------------
    # Distribution-aware additions (Phase 2 of the dashboard redesign).
    # -----------------------------------------------------------------------

    def _seat_count_histogram(
        self,
        plot_dir: Path,
        race_catalog: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        control_forecasts: pl.DataFrame,
    ) -> Path | None:
        """Histogram of total seats per party across draws with majority threshold."""
        if (
            race_catalog.is_empty()
            or forecast_draws.is_empty()
            or control_forecasts.is_empty()
        ):
            return None
        catalog = race_catalog.select(["race_id", "control_body", "seats"])
        joined = forecast_draws.join(catalog, on="race_id", how="inner").filter(
            pl.col("control_body").is_not_null()
        )
        if joined.is_empty():
            return None
        bodies = control_forecasts.select(["control_body", "control_threshold"]).unique()
        if bodies.is_empty():
            return None
        body = str(bodies.row(0, named=True)["control_body"])
        threshold = int(bodies.row(0, named=True)["control_threshold"])
        body_draws = joined.filter(pl.col("control_body") == body).filter(pl.col("winner"))
        if body_draws.is_empty():
            return None
        per_draw = (
            body_draws.group_by(["draw_id", "party"])
            .agg(pl.col("seats").sum().alias("seat_count"))
            .sort("draw_id")
        )
        holdovers = {
            str(row["party"]).upper(): int(row.get("holdover_seats") or 0)
            for row in control_forecasts.iter_rows(named=True)
            if str(row["control_body"]) == body
        }
        parties_present = sorted(per_draw["party"].unique().to_list())
        if not parties_present:
            return None
        fig, ax = plt.subplots(figsize=(9.6, 4.0), dpi=150)
        all_max = 0
        for party in parties_present:
            slice_ = per_draw.filter(pl.col("party") == party)["seat_count"].to_numpy()
            slice_ = slice_ + holdovers.get(party.upper(), 0)
            if slice_.size == 0:
                continue
            bins = max(15, int(slice_.max() - slice_.min()) + 1)
            color = PARTY_COLORS.get(str(party).upper(), ACCENT_FALLBACK)
            ax.hist(
                slice_,
                bins=bins,
                alpha=0.55,
                color=color,
                edgecolor=color,
                linewidth=1.0,
                label=str(party).upper(),
            )
            all_max = max(all_max, int(slice_.max()))
        ax.axvline(threshold, color=MUTED_COLOR, linestyle="--", linewidth=1.2)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.95,
            f"  threshold = {threshold}",
            color=MUTED_COLOR,
            fontsize=9,
            va="top",
        )
        ax.set_xlabel(f"Total {body} seats per draw")
        ax.set_ylabel("Simulation draws")
        ax.set_title(
            f"{body.title()} Seat-Count Distribution",
            loc="left",
            fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=9)
        self._style_axis(ax)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        return self._save(fig, plot_dir / "seat_count_histogram.png")

    def _race_vote_share_kde_grid(
        self,
        plot_dir: Path,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        max_panels: int = 12,
    ) -> Path | None:
        """Small multiples of vote-share KDEs for the most competitive races."""
        if race_forecasts.is_empty() or forecast_draws.is_empty():
            return None
        candidates = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if candidates.is_empty():
            return None
        ranked = (
            candidates.with_columns(
                (pl.col("winner_probability") - 0.5).abs().alias("_dist")
            )
            .group_by("race_id", maintain_order=True)
            .agg(pl.col("_dist").min().alias("_dist"))
            .sort("_dist")
            .head(max_panels)
        )
        race_ids = ranked["race_id"].to_list()
        if not race_ids:
            return None
        n = len(race_ids)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4.5, rows * 3.0), dpi=150, squeeze=False
        )
        for index, race_id in enumerate(race_ids):
            ax = axes[index // cols][index % cols]
            self._render_race_kde_panel(ax, race_id, race_forecasts, forecast_draws)
        for index in range(n, rows * cols):
            axes[index // cols][index % cols].axis("off")
        fig.suptitle(
            "Vote-Share Distribution — Most Competitive Races",
            x=0.02,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        return self._save(fig, plot_dir / "race_vote_share_kde.png")

    def _render_race_kde_panel(
        self,
        ax: plt.Axes,
        race_id: str,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
    ) -> None:
        race_draws = forecast_draws.filter(pl.col("race_id") == race_id)
        race_meta = race_forecasts.filter(pl.col("race_id") == race_id)
        if race_draws.is_empty() or race_meta.is_empty():
            ax.text(0.5, 0.5, "no draws", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        for option in race_meta.iter_rows(named=True):
            samples = (
                race_draws.filter(pl.col("option_id") == option["option_id"])["vote_share"]
                .to_numpy()
                .astype(float)
            )
            if samples.size < 5:
                continue
            party = str(option.get("party") or "").upper()
            color = PARTY_COLORS.get(party, ACCENT_FALLBACK)
            xs = np.linspace(max(0.0, samples.min() - 0.05), min(1.0, samples.max() + 0.05), 200)
            density = self._gaussian_kde(samples, xs)
            ax.fill_between(xs, density, alpha=0.18, color=color)
            ax.plot(xs, density, color=color, linewidth=1.6, label=party)
            ax.axvline(
                float(option.get("vote_share_mean") or np.mean(samples)),
                color=color,
                linestyle="-",
                linewidth=0.9,
                alpha=0.7,
            )
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=0.8)
        ax.set_xlim(0.25, 0.75)
        ax.set_yticks([])
        ax.set_title(race_id, fontsize=11, color=TEXT_COLOR, loc="left")
        ax.set_xlabel("Vote share", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)

    def _tipping_point_bars(
        self,
        plot_dir: Path,
        control_forecasts: pl.DataFrame,
    ) -> Path | None:
        """Horizontal bars of pivotal-flip rates for the top tipping-point races."""
        if control_forecasts.is_empty() or "pivotal_rates" not in control_forecasts.columns:
            return None
        rows: list[dict[str, object]] = []
        for record in control_forecasts.iter_rows(named=True):
            raw = record.get("pivotal_rates")
            if not raw:
                continue
            try:
                payload = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            for item in payload:
                rows.append(
                    {
                        "race_id": str(item.get("race_id")),
                        "party": str(record.get("party") or "").upper(),
                        "pivotal_rate": float(item.get("pivotal_rate") or 0.0),
                    }
                )
        if not rows:
            return None
        frame = (
            pl.DataFrame(rows)
            .sort("pivotal_rate", descending=True)
            .head(10)
            .reverse()
        )
        labels = [
            f"{row['race_id']} ({row['party']})" for row in frame.iter_rows(named=True)
        ]
        values = frame["pivotal_rate"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]).upper(), ACCENT_FALLBACK)
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(9.6, max(3.6, len(labels) * 0.42)), dpi=150)
        ax.barh(labels, values, color=colors)
        ax.set_xlabel("Pivotal-flip rate (P this race decides control)")
        ax.set_title("Tipping-Point Races", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        for index, value in enumerate(values):
            ax.text(value + 0.005, index, f"{value:.1%}", va="center", fontsize=9)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "tipping_point_bars.png")

    def _driver_waterfall(
        self,
        plot_dir: Path,
        race_forecasts: pl.DataFrame,
        max_panels: int = 6,
    ) -> Path | None:
        """Per-race component contribution waterfalls for the closest races."""
        if race_forecasts.is_empty() or "component_contributions" not in race_forecasts.columns:
            return None
        candidates = race_forecasts.filter(
            pl.col("winner_probability").is_not_null()
            & pl.col("component_contributions").is_not_null()
        )
        if candidates.is_empty():
            return None
        candidates = candidates.with_columns(
            (pl.col("winner_probability") - 0.5).abs().alias("_dist")
        ).sort("_dist")
        chosen: list[dict[str, object]] = []
        seen_races: set[str] = set()
        for row in candidates.iter_rows(named=True):
            race_id = str(row["race_id"])
            if race_id in seen_races:
                continue
            seen_races.add(race_id)
            chosen.append(row)
            if len(chosen) >= max_panels:
                break
        if not chosen:
            return None
        cols = min(3, len(chosen))
        rows_n = int(np.ceil(len(chosen) / cols))
        fig, axes = plt.subplots(
            rows_n, cols, figsize=(cols * 4.5, rows_n * 3.2), dpi=150, squeeze=False
        )
        for index, row in enumerate(chosen):
            ax = axes[index // cols][index % cols]
            self._render_driver_waterfall(ax, row)
        for index in range(len(chosen), rows_n * cols):
            axes[index // cols][index % cols].axis("off")
        fig.suptitle(
            "Component Drivers — Closest Races",
            x=0.02,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        return self._save(fig, plot_dir / "driver_waterfall.png")

    def _render_driver_waterfall(self, ax: plt.Axes, row: dict[str, object]) -> None:
        try:
            payload = json.loads(str(row.get("component_contributions") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict) or not payload:
            ax.text(0.5, 0.5, "no decomposition", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        components = []
        contributions = []
        for component, info in payload.items():
            if not isinstance(info, dict):
                continue
            try:
                contribution = float(info.get("weighted_marginal_win_probability", 0.0)) - float(
                    info.get("weight", 0.0)
                ) * 0.5
            except (TypeError, ValueError):
                continue
            components.append(component)
            contributions.append(contribution)
        if not components:
            ax.text(0.5, 0.5, "empty contributions", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        order = sorted(zip(components, contributions, strict=True), key=lambda item: item[1])
        components_sorted, contributions_sorted = zip(*order, strict=True)
        colors = [
            PARTY_COLORS["DEM"] if value > 0 else PARTY_COLORS["REP"]
            for value in contributions_sorted
        ]
        ax.barh(list(components_sorted), list(contributions_sorted), color=colors)
        ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
        ax.set_title(
            f"{row.get('race_id')} → {str(row.get('option_id') or '').split('-')[-1]}",
            fontsize=11,
            color=TEXT_COLOR,
            loc="left",
        )
        ax.set_xlabel("Contribution to D-leaning win probability", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)

    def _reliability_diagram_with_band(
        self, plot_dir: Path, frame: pl.DataFrame
    ) -> Path | None:
        """Calibration curve with Wilson confidence band and forecast histogram."""
        if frame.is_empty() or "ensemble_probability" not in frame.columns:
            return None
        df = frame.select(["ensemble_probability", "actual_winner"]).drop_nulls()
        if df.is_empty():
            return None
        probability = df["ensemble_probability"].cast(pl.Float64).to_numpy()
        actual = df["actual_winner"].cast(pl.Float64).to_numpy()
        bins = np.linspace(0, 1, 11)
        xs: list[float] = []
        ys: list[float] = []
        ns: list[int] = []
        for lower, upper in pairwise(bins):
            mask = (probability >= lower) & (
                probability < upper if upper < 1 else probability <= upper
            )
            if not np.any(mask):
                continue
            xs.append(float(np.mean(probability[mask])))
            ys.append(float(np.mean(actual[mask])))
            ns.append(int(mask.sum()))
        if not xs:
            return None
        z = 1.96
        lower_band = []
        upper_band = []
        for y, n in zip(ys, ns, strict=True):
            if n == 0:
                lower_band.append(y)
                upper_band.append(y)
                continue
            denom = 1 + z * z / n
            centre = (y + z * z / (2 * n)) / denom
            half = (z / denom) * np.sqrt(y * (1 - y) / n + z * z / (4 * n * n))
            lower_band.append(max(0.0, centre - half))
            upper_band.append(min(1.0, centre + half))
        fig, (ax_main, ax_hist) = plt.subplots(
            2, 1, figsize=(7.5, 6.0), dpi=150, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_main.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax_main.fill_between(xs, lower_band, upper_band, alpha=0.18, color=PARTY_COLORS["DEM"])
        ax_main.scatter(xs, ys, s=72, color=PARTY_COLORS["DEM"], zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax_main.text(x + 0.02, y, f"{y:.0%}", va="center", fontsize=9, color=TEXT_COLOR)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.set_xlabel("Mean forecast probability")
        ax_main.set_ylabel("Observed win rate")
        ax_main.set_title(
            "Reliability Diagram (95% Wilson band)",
            loc="left",
            fontweight="bold",
        )
        ax_main.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax_main.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax_main)
        ax_hist.hist(probability, bins=bins, color=PARTY_COLORS["DEM"], alpha=0.55)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylabel("n", fontsize=9)
        ax_hist.set_xlabel("Forecast probability distribution")
        ax_hist.xaxis.set_major_formatter(PercentFormatter(1.0))
        for spine in ("top", "right", "left"):
            ax_hist.spines[spine].set_visible(False)
        ax_hist.spines["bottom"].set_color(GRID_COLOR)
        return self._save(fig, plot_dir / "reliability_diagram.png")

    @staticmethod
    def _gaussian_kde(samples: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Plain Gaussian KDE without depending on scipy."""
        if samples.size == 0:
            return np.zeros_like(xs)
        std = float(np.std(samples))
        if std == 0:
            std = 1e-3
        bw = std * (4 / (3 * samples.size)) ** (1 / 5)
        bw = max(bw, 1e-3)
        diffs = (xs[:, None] - samples[None, :]) / bw
        kernel = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
        return kernel.sum(axis=1) / (samples.size * bw)


ACCENT_FALLBACK = "#547c70"
