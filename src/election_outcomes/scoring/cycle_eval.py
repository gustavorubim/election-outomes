from __future__ import annotations

import html
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
        plots: list[dict[str, str]] = []
        for path, title in [
            (
                self._ec_probability_plot(frame, plot_dir),
                "Electoral College winner probability by cycle",
            ),
            (self._accuracy_plot(frame, plot_dir), "State accuracy and Brier score by cycle"),
            (self._error_plot(frame, plot_dir), "Vote-share error and upset count by cycle"),
        ]:
            if path is not None:
                plots.append({"path": str(path.relative_to(output_dir)), "title": title})
        return plots

    @staticmethod
    def _ec_probability_plot(frame: pl.DataFrame, plot_dir: Path) -> Path | None:
        required = {"cycle", "forecast_ec_winner_party", "forecast_ec_win_probability"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        colors = [
            "#2b6cb0" if party == "DEM" else "#c43b3b"
            for party in rows["forecast_ec_winner_party"].to_list()
        ]
        ax.bar(
            [str(cycle) for cycle in rows["cycle"].to_list()],
            rows["forecast_ec_win_probability"].to_list(),
            color=colors,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("EC Winner Probability By Cycle", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        path = plot_dir / "ec_winner_probability_by_cycle.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _accuracy_plot(frame: pl.DataFrame, plot_dir: Path) -> Path | None:
        required = {"cycle", "state_accuracy", "brier_score"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        x_labels = [str(cycle) for cycle in rows["cycle"].to_list()]
        x = range(len(x_labels))
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        ax.plot(
            x,
            rows["state_accuracy"].to_list(),
            marker="o",
            color="#245b8f",
            label="State accuracy",
        )
        ax.plot(x, rows["brier_score"].to_list(), marker="s", color="#9c6f19", label="Brier score")
        ax.set_xticks(list(x), x_labels)
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy And Calibration By Cycle", loc="left", fontweight="bold")
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
    def _error_plot(frame: pl.DataFrame, plot_dir: Path) -> Path | None:
        required = {"cycle", "mean_absolute_vote_share_error", "upset_count"}
        if not required.issubset(set(frame.columns)):
            return None
        rows = frame.sort("cycle")
        x_labels = [str(cycle) for cycle in rows["cycle"].to_list()]
        x = list(range(len(x_labels)))
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
        ax.bar(x, rows["mean_absolute_vote_share_error"].to_list(), color="#547c70")
        ax.set_ylabel("Vote-share MAE")
        ax.set_xticks(x, x_labels)
        ax2 = ax.twinx()
        ax2.plot(x, rows["upset_count"].to_list(), color="#c43b3b", marker="o", label="Upsets")
        ax2.set_ylabel("Upset count")
        ax.set_title("Error And Upsets By Cycle", loc="left", fontweight="bold")
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
            if "upset_count" in frame.columns else 0,
        }

    def _html_report(self, payload: dict[str, Any], frame: pl.DataFrame) -> str:
        aggregate = payload["aggregate"]
        chamber_label = self._chamber_label(frame)
        threshold_label = self._threshold_label(frame)
        cards = [
            ("Cycles evaluated", payload["cycle_count"]),
            (f"{chamber_label} majority threshold", threshold_label),
            ("Majority winner accuracy", self._pct(aggregate.get("majority_winner_accuracy"))),
            ("Mean state/seat accuracy", self._pct(aggregate["mean_state_accuracy"])),
            ("Mean Brier", self._num(aggregate["mean_brier_score"], 4)),
            ("Total upsets", aggregate["total_upsets"]),
        ]
        card_html = "".join(
            "<div class='card'>"
            f"<span>{html.escape(str(label))}</span><strong>{value}</strong></div>"
            for label, value in cards
        )
        plot_html = "".join(
            "<figure class='plot-card'>"
            f"<img src='{html.escape(plot['path'])}' alt='{html.escape(plot['title'])}'>"
            f"<figcaption>{html.escape(plot['title'])}</figcaption></figure>"
            for plot in payload["plot_manifest"]
        )
        row_html = "".join(self._row_html(row) for row in frame.sort("cycle").to_dicts())
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cycle Evaluation {html.escape(str(payload["run_id"]))}</title>
  <style>{self._css()}</style>
</head>
<body>
  <header>
    <p class="eyebrow">Historical benchmark</p>
    <h1>{html.escape(chamber_label)} Cycle Evaluation</h1>
    <p class="subtitle">
      Same-date forecasts as of {html.escape(str(payload["as_of_mm_dd"]))}, compared
      against actual outcomes. Majority is reached at
      <strong>{html.escape(str(threshold_label))}</strong> seats/votes.
    </p>
  </header>
  <section class="cards">{card_html}</section>
  <section class="plots">{plot_html}</section>
  <section class="panel">
    <h2>Cycle Results — Majority Story</h2>
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
        bodies = [
            value
            for value in frame["control_body"].drop_nulls().unique().to_list()
            if value
        ]
        if not bodies:
            return "Cycle"
        body = str(bodies[0])
        return {
            "president": "Electoral College (Presidential)",
            "senate": "U.S. Senate",
            "house": "U.S. House",
        }.get(body, body.title())

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
        if value is None:
            return "n/a"
        return f"{float(value) * 100:.1f}%"

    @staticmethod
    def _num(value: Any, digits: int) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    @staticmethod
    def _css() -> str:
        return """
:root { --ink: #202124; --muted: #656a70; --rule: #d8dde3; --bg: #f6f4ef; --card: #ffffff; }
body {
  margin: 0;
  padding: 36px;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header { max-width: 1040px; margin: 0 auto 24px; }
h1 { margin: 0; font-size: 44px; letter-spacing: 0; }
h2 { margin: 0 0 14px; }
.eyebrow {
  margin: 0 0 8px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-size: 12px;
  font-weight: 700;
}
.subtitle { color: var(--muted); max-width: 760px; }
.cards, .plots, .panel { max-width: 1040px; margin: 0 auto 20px; }
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.card, .panel, .plot-card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  box-shadow: 0 6px 20px rgba(42, 37, 25, .05);
}
.card { padding: 16px; }
.card span { color: var(--muted); display: block; }
.card strong { font-size: 28px; }
.plots { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.plot-card { margin: 0; padding: 12px; }
.plot-card img { width: 100%; height: 220px; object-fit: contain; display: block; }
.plot-card figcaption { color: var(--muted); font-size: 13px; margin-top: 8px; }
.panel { padding: 18px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td {
  text-align: left;
  border-bottom: 1px solid var(--rule);
  padding: 9px 7px;
  vertical-align: top;
}
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
a { color: #245b8f; }
@media (max-width: 860px) {
  body { padding: 20px; }
  .cards, .plots { grid-template-columns: 1fr; }
  h1 { font-size: 36px; }
}
"""
