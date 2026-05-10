from __future__ import annotations

import html
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

from election_outcomes.storage.io import write_json, write_parquet, write_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CycleEvaluationReport:
    """Aggregate same-date presidential cycle evaluations into one dashboard."""

    def render(
        self,
        rows: list[dict[str, Any]],
        output_dir: Path,
        run_id: str,
        as_of_mm_dd: str,
    ) -> dict[str, Any]:
        frame = pl.DataFrame(rows) if rows else self._empty_frame()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(frame, output_dir / "cycle_summary.parquet")
        plot_manifest = self._write_plots(frame, output_dir)
        payload = {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "as_of_mm_dd": as_of_mm_dd,
            "cycle_count": frame.height,
            "aggregate": self._aggregate(frame),
            "plot_manifest": plot_manifest,
            "cycles": frame.to_dicts(),
        }
        write_json(payload, output_dir / "cycle_summary.json")
        write_text(self._html_report(payload, frame), output_dir / "cycle_eval.html")
        write_text(self._narrative(payload, frame), output_dir / "narrative.md")
        return {**payload, "output_dir": str(output_dir)}

    @staticmethod
    def _empty_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "cycle": pl.Int64,
                "as_of": pl.Utf8,
                "forecast_run_id": pl.Utf8,
                "control_body": pl.Utf8,
                "majority_threshold": pl.Int64,
                "forecast_ec_winner_party": pl.Utf8,
                "actual_ec_winner_party": pl.Utf8,
                "state_topline_ec_winner_party": pl.Utf8,
                "state_topline_ec_winner_accuracy": pl.Float64,
                "forecast_ec_win_probability": pl.Float64,
                "forecast_ec_p10": pl.Float64,
                "forecast_ec_p50": pl.Float64,
                "forecast_ec_p90": pl.Float64,
                "dem_seat_count_mean": pl.Float64,
                "rep_seat_count_mean": pl.Float64,
                "dem_majority_probability": pl.Float64,
                "rep_majority_probability": pl.Float64,
                "ec_winner_accuracy": pl.Float64,
                "state_accuracy": pl.Float64,
                "state_accuracy_n": pl.Int64,
                "brier_score": pl.Float64,
                "mean_absolute_vote_share_error": pl.Float64,
                "upset_count": pl.Int64,
                "missed_states": pl.Utf8,
                "race_count": pl.Int64,
                "diagnostics_path": pl.Utf8,
                "comparison_path": pl.Utf8,
            }
        )

    def _write_plots(self, frame: pl.DataFrame, output_dir: Path) -> list[dict[str, str]]:
        if frame.is_empty():
            return []
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        chamber = self._chamber_label(frame)
        outcome_label = self._outcome_label(frame)
        race_label = self._race_unit_label(frame)
        plots: list[dict[str, str]] = []
        for path, title in [
            (
                self._ec_probability_plot(frame, plot_dir, chamber, outcome_label),
                f"{chamber} {outcome_label.lower()} probability by cycle",
            ),
            (
                self._accuracy_plot(frame, plot_dir, chamber, race_label),
                f"{race_label} accuracy and Brier score by cycle",
            ),
            (
                self._error_plot(frame, plot_dir, chamber),
                "Vote-share error and upset count by cycle",
            ),
        ]:
            if path is not None:
                plots.append({"path": str(path.relative_to(output_dir)), "title": title})
        return plots

    @staticmethod
    def _ec_probability_plot(
        frame: pl.DataFrame, plot_dir: Path, chamber: str, outcome_label: str
    ) -> Path | None:
        required = {"cycle", "forecast_ec_winner_party", "forecast_ec_win_probability"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        colors = [
            "#2b6cb0" if party == "DEM" else "#c43b3b" if party == "REP" else "#8a8f98"
            for party in rows["forecast_ec_winner_party"].to_list()
        ]
        probabilities = CycleEvaluationReport._numeric_values(rows, "forecast_ec_win_probability")
        ax.bar(
            [str(cycle) for cycle in rows["cycle"].to_list()],
            probabilities,
            color=colors,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title(
            f"{chamber} {outcome_label} Probability By Cycle",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        path = plot_dir / "ec_winner_probability_by_cycle.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _accuracy_plot(
        frame: pl.DataFrame, plot_dir: Path, chamber: str, race_label: str
    ) -> Path | None:
        required = {"cycle", "state_accuracy", "brier_score"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        x_labels = [str(cycle) for cycle in rows["cycle"].to_list()]
        x = range(len(x_labels))
        accuracy = CycleEvaluationReport._numeric_values(rows, "state_accuracy")
        brier = CycleEvaluationReport._numeric_values(rows, "brier_score")
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        ax.plot(
            x,
            accuracy,
            marker="o",
            color="#245b8f",
            label=f"{race_label} accuracy",
        )
        ax.plot(x, brier, marker="s", color="#9c6f19", label="Brier score")
        ax.set_xticks(list(x), x_labels)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{chamber} Accuracy And Calibration By Cycle",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        path = plot_dir / "accuracy_brier_by_cycle.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _error_plot(frame: pl.DataFrame, plot_dir: Path, chamber: str) -> Path | None:
        required = {"cycle", "mean_absolute_vote_share_error", "upset_count"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        x_labels = [str(cycle) for cycle in rows["cycle"].to_list()]
        x = list(range(len(x_labels)))
        errors = CycleEvaluationReport._numeric_values(rows, "mean_absolute_vote_share_error")
        upsets = CycleEvaluationReport._numeric_values(rows, "upset_count")
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        ax.bar(x, errors, color="#547c70")
        ax.set_ylabel("Vote-share MAE")
        ax.set_xticks(x, x_labels)
        ax2 = ax.twinx()
        ax2.plot(x, upsets, color="#c43b3b", marker="o", label="Upsets")
        ax2.set_ylabel("Upset count")
        ax.set_title(
            f"{chamber} Error And Upsets By Cycle",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.25)
        for axis in (ax, ax2):
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
        path = plot_dir / "error_upsets_by_cycle.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _aggregate(frame: pl.DataFrame) -> dict[str, Any]:
        if frame.is_empty():
            return {
                "mean_state_accuracy": None,
                "mean_brier_score": None,
                "mean_vote_share_mae": None,
                "ec_winner_accuracy": None,
                "majority_winner_accuracy": None,
                "total_upsets": 0,
            }

        def safe_mean(column: str) -> float | None:
            if column not in frame.columns:
                return None
            non_null = frame[column].drop_nulls()
            if non_null.len() == 0:
                return None
            return float(non_null.mean())

        return {
            "mean_state_accuracy": safe_mean("state_accuracy"),
            "mean_brier_score": safe_mean("brier_score"),
            "mean_vote_share_mae": safe_mean("mean_absolute_vote_share_error"),
            "ec_winner_accuracy": safe_mean("ec_winner_accuracy"),
            "majority_winner_accuracy": safe_mean("ec_winner_accuracy"),
            "total_upsets": int(frame["upset_count"].fill_null(0).sum())
            if "upset_count" in frame.columns
            else 0,
        }

    def _html_report(self, payload: dict[str, Any], frame: pl.DataFrame) -> str:
        aggregate = payload["aggregate"]
        chamber_label = self._chamber_label(frame)
        threshold_label = self._threshold_label(frame)
        narrative = self._narrative_summary(frame, aggregate, chamber_label, threshold_label)
        kpi_html = self._kpi_html(payload, aggregate, chamber_label, threshold_label)
        plot_html = "".join(
            "<figure>"
            f"<img src='{html.escape(plot['path'])}' alt='{html.escape(plot['title'])}'>"
            f"<figcaption>{html.escape(plot['title'])}</figcaption></figure>"
            for plot in payload["plot_manifest"]
        )
        row_html = "".join(self._row_html(row) for row in frame.sort("cycle").to_dicts())
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cycle Evaluation {html.escape(str(payload["run_id"]))}</title>
  <style>{self._css()}</style>
</head>
<body>
<div class="container">
  <header class="hero">
    <p class="eyebrow">Historical benchmark</p>
    <h1>{html.escape(chamber_label)} Cycle Evaluation</h1>
    <p class="subtitle">
      Same-date forecasts as of {html.escape(str(payload["as_of_mm_dd"]))}, compared
      against actual outcomes. Majority is reached at
      <strong>{html.escape(str(threshold_label))}</strong> seats/votes.
    </p>
  </header>

  <div class="narrative">{html.escape(narrative)}</div>

  <section class="section">
    <h2>Headline numbers</h2>
    <div class="kpi-row">{kpi_html}</div>
  </section>

  <section class="section">
    <h2>Distribution across cycles</h2>
    <p class="subtitle" style="margin-bottom:14px;">
      Probability of the headline outcome, race-level accuracy, and error metrics
      plotted side by side so you can see how the forecast performed in each cycle.
    </p>
    <div class="plot-grid">{plot_html}</div>
  </section>

  <section class="section">
    <h2>Cycle results — majority story</h2>
    <table>
      <thead>
        <tr>
          <th>Cycle</th>
          <th>Forecast majority</th>
          <th>DEM seats (mean)</th>
          <th>REP seats (mean)</th>
          <th>D maj prob</th>
          <th>R maj prob</th>
          <th>Seat p10/p50/p90</th>
          <th>Race acc</th>
          <th>Brier</th>
          <th>Missed</th>
          <th>Links</th>
        </tr>
      </thead>
      <tbody>{row_html}</tbody>
    </table>
  </section>
</div>
</body>
</html>
"""

    @classmethod
    def _row_html(cls, row: dict[str, Any]) -> str:
        diagnostics = html.escape(str(row["diagnostics_path"]))
        comparison = html.escape(str(row["comparison_path"]))
        forecast_party = html.escape(str(row.get("forecast_ec_winner_party") or "?"))
        return (
            "<tr>"
            f"<td>{row['cycle']}</td>"
            f"<td>{forecast_party} "
            f"({cls._pct(row['forecast_ec_win_probability'])})</td>"
            f"<td>{cls._num(row.get('dem_seat_count_mean'), 1)}</td>"
            f"<td>{cls._num(row.get('rep_seat_count_mean'), 1)}</td>"
            f"<td>{cls._pct(row.get('dem_majority_probability'))}</td>"
            f"<td>{cls._pct(row.get('rep_majority_probability'))}</td>"
            f"<td>{cls._num(row['forecast_ec_p10'], 0)} / {cls._num(row['forecast_ec_p50'], 0)} / "
            f"{cls._num(row['forecast_ec_p90'], 0)}</td>"
            f"<td>{cls._pct(row['state_accuracy'])}</td>"
            f"<td>{cls._num(row['brier_score'], 4)}</td>"
            f"<td>{html.escape(str(row['missed_states'] or '-'))}</td>"
            f"<td><a href='{diagnostics}'>diagnostics</a> | "
            f"<a href='{comparison}'>comparison</a></td>"
            "</tr>"
        )

    def _narrative(self, payload: dict[str, Any], frame: pl.DataFrame) -> str:
        aggregate = payload["aggregate"]
        chamber_label = self._chamber_label(frame)
        threshold_label = self._threshold_label(frame)
        lines = [
            f"# {chamber_label} Cycle Evaluation Narrative",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- Same-date cut: `{payload['as_of_mm_dd']}`",
            f"- Cycles evaluated: `{payload['cycle_count']}`",
            f"- Majority threshold: `{threshold_label}`",
            f"- Majority winner accuracy: `{self._pct(aggregate.get('majority_winner_accuracy'))}`",
            f"- Mean state/seat accuracy: `{self._pct(aggregate['mean_state_accuracy'])}`",
            f"- Mean Brier score: `{self._num(aggregate['mean_brier_score'], 4)}`",
            f"- Mean vote-share MAE: `{self._pct(aggregate['mean_vote_share_mae'])}`",
            f"- Total actual-winner upsets: `{aggregate['total_upsets']}`",
            "",
            "## Cycles",
            "",
        ]
        for row in frame.sort("cycle").to_dicts():
            d_seats = self._num(row.get("dem_seat_count_mean"), 1)
            r_seats = self._num(row.get("rep_seat_count_mean"), 1)
            d_prob = self._pct(row.get("dem_majority_probability"))
            r_prob = self._pct(row.get("rep_majority_probability"))
            lines.append(
                f"- **{row['cycle']}**: forecast majority "
                f"{row.get('forecast_ec_winner_party') or '?'} "
                f"({self._pct(row['forecast_ec_win_probability'])}). "
                f"DEM mean seats {d_seats} (maj {d_prob}); "
                f"REP mean seats {r_seats} (maj {r_prob}); "
                f"race accuracy {self._pct(row['state_accuracy'])}; "
                f"missed: {row['missed_states'] or '-'}."
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _chamber_label(frame: pl.DataFrame) -> str:
        if frame.is_empty() or "control_body" not in frame.columns:
            return "Cycle"
        bodies = [value for value in frame["control_body"].drop_nulls().unique().to_list() if value]
        if not bodies:
            return "Cycle"
        body = str(bodies[0])
        return {
            "president": "Electoral College (Presidential)",
            "senate": "U.S. Senate",
            "house": "U.S. House",
        }.get(body, body.title())

    @staticmethod
    def _outcome_label(frame: pl.DataFrame) -> str:
        """Probability label for the chamber's headline metric."""
        if frame.is_empty() or "control_body" not in frame.columns:
            return "Winner"
        bodies = [value for value in frame["control_body"].drop_nulls().unique().to_list() if value]
        if not bodies:
            return "Winner"
        body = str(bodies[0])
        return {
            "president": "EC Winner",
            "senate": "Majority",
            "house": "Majority",
        }.get(body, "Winner")

    @staticmethod
    def _race_unit_label(frame: pl.DataFrame) -> str:
        """Per-race accuracy label that matches the scenario's geography unit."""
        if frame.is_empty() or "control_body" not in frame.columns:
            return "Race"
        bodies = [value for value in frame["control_body"].drop_nulls().unique().to_list() if value]
        if not bodies:
            return "Race"
        body = str(bodies[0])
        return {
            "president": "State",
            "senate": "Senate seat",
            "house": "House district",
        }.get(body, "Race")

    @staticmethod
    def _threshold_label(frame: pl.DataFrame) -> str:
        if frame.is_empty() or "majority_threshold" not in frame.columns:
            return "n/a"
        thresholds = [
            int(value)
            for value in frame["majority_threshold"].drop_nulls().unique().to_list()
            if value is not None
        ]
        if not thresholds:
            return "n/a"
        if len(thresholds) == 1:
            return str(thresholds[0])
        return ", ".join(str(value) for value in sorted(thresholds))

    @staticmethod
    def _pct(value: Any) -> str:
        numeric = CycleEvaluationReport._coerce_numeric(value)
        if numeric is None:
            return "n/a"
        return f"{numeric * 100:.1f}%"

    @staticmethod
    def _num(value: Any, digits: int) -> str:
        numeric = CycleEvaluationReport._coerce_numeric(value)
        if numeric is None:
            return "n/a"
        return f"{numeric:.{digits}f}"

    @staticmethod
    def _numeric_values(frame: pl.DataFrame, column: str) -> list[float]:
        return [
            numeric
            if (numeric := CycleEvaluationReport._coerce_numeric(value)) is not None
            else math.nan
            for value in frame[column].to_list()
        ]

    @staticmethod
    def _coerce_numeric(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _css() -> str:
        from election_outcomes.reports._style import report_css

        return report_css()

    @staticmethod
    def _kpi_html(
        payload: dict[str, Any],
        aggregate: dict[str, Any],
        chamber_label: str,
        threshold_label: str,
    ) -> str:
        items: list[tuple[str, str, str]] = [
            ("Cycles evaluated", str(payload["cycle_count"]), "rolling-origin holdouts"),
            (
                "Majority threshold",
                str(threshold_label),
                f"{chamber_label} control gate",
            ),
            (
                "Majority winner accuracy",
                CycleEvaluationReport._pct(aggregate.get("majority_winner_accuracy")),
                "share of cycles called correctly",
            ),
            (
                "Mean race accuracy",
                CycleEvaluationReport._pct(aggregate.get("mean_state_accuracy")),
                "average per-race winner accuracy",
            ),
            (
                "Mean Brier score",
                CycleEvaluationReport._num(aggregate.get("mean_brier_score"), 4),
                "lower is better",
            ),
            (
                "Total upsets",
                str(aggregate.get("total_upsets") or 0),
                "races forecast lost but won",
            ),
        ]
        return "".join(
            "<div class='kpi'>"
            f"<span class='label'>{html.escape(label)}</span>"
            f"<strong class='value'>{html.escape(str(value))}</strong>"
            f"<div class='detail'>{html.escape(detail)}</div>"
            "</div>"
            for label, value, detail in items
        )

    @staticmethod
    def _narrative_summary(
        frame: pl.DataFrame,
        aggregate: dict[str, Any],
        chamber_label: str,
        threshold_label: str,
    ) -> str:
        if frame.is_empty():
            return ""
        cycles = sorted(int(c) for c in frame["cycle"].drop_nulls().unique().to_list())
        first, last = cycles[0], cycles[-1]
        majority_acc = aggregate.get("majority_winner_accuracy")
        race_acc = aggregate.get("mean_state_accuracy")
        brier = aggregate.get("mean_brier_score")
        parts = [
            f"{chamber_label} forecast across {len(cycles)} cycles "
            f"({first}-{last}); majority threshold {threshold_label}."
        ]
        if majority_acc is not None:
            parts.append(f"Majority winner correct on {float(majority_acc) * 100:.0f}% of cycles.")
        if race_acc is not None:
            parts.append(f"Per-race accuracy averaged {float(race_acc) * 100:.1f}%.")
        if brier is not None:
            parts.append(f"Mean Brier {float(brier):.4f} (lower is better).")
        return " ".join(parts)
