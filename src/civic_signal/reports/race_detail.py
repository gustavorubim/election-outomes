"""Per-race detail pages: KDE + polling trajectory + drivers + provenance.

Each Tier A race gets a standalone HTML file under
`artifacts/<run>/races/<race_id>.html` with charts that focus on a single
contest. The diagnostics dashboard links to these pages from its forecast
table so readers can drill into any race.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from civic_signal.reports._style import (
    NEUTRAL,
    PARTY,
    SIZE_PANEL,
    apply_rcparams,
    party_color,
    report_css,
    style_axis,
)
from civic_signal.storage.io import write_text


class RaceDetailRenderer:
    """Generate one HTML page per Tier A race plus a small-multiples chart set."""

    def render_all(
        self,
        artifact_dir: Path,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        poll_trajectory: pl.DataFrame | None = None,
        max_races: int = 24,
    ) -> dict[str, str]:
        """Return a mapping of race_id -> relative html path for each generated page."""
        if race_forecasts.is_empty() or race_catalog.is_empty():
            return {}
        apply_rcparams()
        targets = self._pick_races(race_forecasts, max_races)
        if not targets:
            return {}
        race_dir = artifact_dir / "races"
        race_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = race_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        catalog_meta = {row["race_id"]: row for row in race_catalog.iter_rows(named=True)}
        results: dict[str, str] = {}
        for race_id in targets:
            page_path = race_dir / f"{race_id}.html"
            kde_path = self._render_kde(plot_dir, race_id, race_forecasts, forecast_draws)
            trajectory_path = self._render_trajectory(plot_dir, race_id, poll_trajectory)
            waterfall_path = self._render_waterfall(plot_dir, race_id, race_forecasts)
            html_doc = self._render_page(
                race_id=race_id,
                race_meta=catalog_meta.get(race_id, {}),
                race_forecasts=race_forecasts,
                kde_path=kde_path,
                trajectory_path=trajectory_path,
                waterfall_path=waterfall_path,
            )
            write_text(html_doc, page_path)
            results[race_id] = f"races/{race_id}.html"
        return results

    @staticmethod
    def _pick_races(race_forecasts: pl.DataFrame, max_races: int) -> list[str]:
        candidates = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if candidates.is_empty():
            return []
        ranked = (
            candidates.with_columns((pl.col("winner_probability") - 0.5).abs().alias("_dist"))
            .group_by("race_id", maintain_order=True)
            .agg(pl.col("_dist").min().alias("_dist"))
            .sort("_dist")
            .head(max_races)
        )
        return ranked["race_id"].to_list()

    def _render_kde(
        self,
        plot_dir: Path,
        race_id: str,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
    ) -> Path | None:
        race_meta = race_forecasts.filter(pl.col("race_id") == race_id)
        race_draws = forecast_draws.filter(pl.col("race_id") == race_id)
        if race_meta.is_empty() or race_draws.is_empty():
            return None
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        for option in race_meta.iter_rows(named=True):
            samples = (
                race_draws.filter(pl.col("option_id") == option["option_id"])["vote_share"]
                .to_numpy()
                .astype(float)
            )
            if samples.size < 5:
                continue
            color = party_color(option.get("party"))
            xs = np.linspace(
                max(0.0, float(samples.min()) - 0.05),
                min(1.0, float(samples.max()) + 0.05),
                240,
            )
            density = self._gaussian_kde(samples, xs)
            ax.fill_between(xs, density, alpha=0.20, color=color)
            ax.plot(xs, density, color=color, linewidth=1.8, label=str(option.get("party") or ""))
            mean_val = float(option.get("vote_share_mean") or np.mean(samples))
            ax.axvline(mean_val, color=color, linewidth=1.0, alpha=0.7)
        ax.axvline(0.5, color=NEUTRAL["muted"], linestyle="--", linewidth=0.9)
        ax.set_xlim(0.25, 0.75)
        ax.set_xlabel("Vote share")
        ax.set_yticks([])
        ax.set_title("Vote-share distribution", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax, grid_axis="x")
        path = plot_dir / f"{race_id}_kde.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def _render_trajectory(
        self,
        plot_dir: Path,
        race_id: str,
        poll_trajectory: pl.DataFrame | None,
    ) -> Path | None:
        if poll_trajectory is None or poll_trajectory.is_empty():
            return None
        required = {
            "race_id",
            "option_id",
            "trajectory_date",
            "latent_vote_share",
            "latent_sigma",
        }
        if not required.issubset(set(poll_trajectory.columns)):
            return None
        slice_ = poll_trajectory.filter(pl.col("race_id") == race_id)
        if slice_.is_empty():
            return None
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        for option_id in slice_["option_id"].unique().to_list():
            series = slice_.filter(pl.col("option_id") == option_id).sort("trajectory_date")
            if series.is_empty():
                continue
            party = str(option_id).split("-")[-1].upper()
            if party in {"D", "DEM"}:
                lookup = "DEM"
            elif party in {"R", "REP"}:
                lookup = "REP"
            else:
                lookup = party
            color = party_color(lookup)
            x_values = series["trajectory_date"].to_list()
            mean = np.array(series["latent_vote_share"].to_list(), dtype=float)
            sigma = np.array(series["latent_sigma"].to_list(), dtype=float)
            lower = np.clip(mean - 1.645 * sigma, 0.0, 1.0)
            upper = np.clip(mean + 1.645 * sigma, 0.0, 1.0)
            ax.plot(x_values, mean, color=color, linewidth=1.8, label=party)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.axhline(0.5, color=NEUTRAL["muted"], linestyle="--", linewidth=0.8)
        ax.set_ylim(0.30, 0.70)
        ax.set_ylabel("Latent vote share")
        ax.set_title("Polling trajectory (Kalman)", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax)
        fig.autofmt_xdate()
        path = plot_dir / f"{race_id}_trajectory.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def _render_waterfall(
        self,
        plot_dir: Path,
        race_id: str,
        race_forecasts: pl.DataFrame,
    ) -> Path | None:
        slice_ = race_forecasts.filter(
            (pl.col("race_id") == race_id) & pl.col("component_contributions").is_not_null()
        )
        if slice_.is_empty():
            return None
        primary = slice_.sort("winner_probability", descending=True).row(0, named=True)
        try:
            payload = json.loads(str(primary.get("component_contributions") or "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not payload:
            return None
        components = []
        contributions = []
        for component, info in payload.items():
            if not isinstance(info, dict):
                continue
            try:
                contribution = (
                    float(info.get("weighted_marginal_win_probability", 0.0))
                    - float(info.get("weight", 0.0)) * 0.5
                )
            except (TypeError, ValueError):
                continue
            components.append(component)
            contributions.append(contribution)
        if not components:
            return None
        order = sorted(zip(components, contributions, strict=True), key=lambda item: item[1])
        components_sorted, contributions_sorted = zip(*order, strict=True)
        colors = [PARTY["DEM"] if value > 0 else PARTY["REP"] for value in contributions_sorted]
        fig, ax = plt.subplots(figsize=SIZE_PANEL)
        ax.barh(list(components_sorted), list(contributions_sorted), color=colors)
        ax.axvline(0, color=NEUTRAL["muted"], linewidth=0.8)
        ax.set_xlabel("Contribution to D-leaning win probability")
        ax.set_title("Component drivers (top option)", loc="left", fontweight="bold")
        style_axis(ax, grid_axis="x")
        path = plot_dir / f"{race_id}_waterfall.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _gaussian_kde(samples: np.ndarray, xs: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.zeros_like(xs)
        std = float(np.std(samples))
        if std == 0:
            std = 1e-3
        bw = max(std * (4 / (3 * samples.size)) ** (1 / 5), 1e-3)
        diffs = (xs[:, None] - samples[None, :]) / bw
        kernel = np.exp(-0.5 * diffs * diffs) / math.sqrt(2 * math.pi)
        return kernel.sum(axis=1) / (samples.size * bw)

    def _render_page(
        self,
        race_id: str,
        race_meta: dict[str, Any],
        race_forecasts: pl.DataFrame,
        kde_path: Path | None,
        trajectory_path: Path | None,
        waterfall_path: Path | None,
    ) -> str:
        race_rows = race_forecasts.filter(pl.col("race_id") == race_id).sort("option_id")
        rows_html = "".join(self._option_row(row) for row in race_rows.iter_rows(named=True))
        plot_html = self._plot_grid(kde_path, trajectory_path, waterfall_path)
        meta_pills = self._meta_pills(race_meta)
        narrative = self._race_narrative(race_meta, race_rows)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Race detail {html.escape(race_id)}</title>
  <style>{report_css()}</style>
</head>
<body>
<div class="container">
  <header class="hero">
    <p class="eyebrow">Race detail</p>
    <h1>{html.escape(race_id)}</h1>
    <p class="subtitle">{meta_pills}</p>
  </header>

  <div class="narrative">{html.escape(narrative)}</div>

  <section class="section">
    <h2>Forecast options</h2>
    <table>
      <thead>
        <tr>
          <th>Option</th><th>Party</th><th>Win prob</th>
          <th>Vote share (mean)</th><th>90% interval</th><th>Top drivers</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>Distribution and drivers</h2>
    <div class="plot-grid">{plot_html}</div>
  </section>

  <section class="section">
    <h2>Provenance</h2>
    <p class="subtitle">
      Race tier: <strong>{html.escape(str(race_meta.get("tier") or "n/a"))}</strong>.
      Office type: {html.escape(str(race_meta.get("office_type") or "n/a"))}.
      Geography: {html.escape(str(race_meta.get("geography") or "n/a"))}.
      Cycle: {html.escape(str(race_meta.get("cycle") or "n/a"))}.
    </p>
  </section>
</div>
</body>
</html>
"""

    @staticmethod
    def _option_row(row: dict[str, Any]) -> str:
        prob = row.get("winner_probability")
        share_mean = row.get("vote_share_mean")
        p05 = row.get("vote_share_p05")
        p95 = row.get("vote_share_p95")
        drivers = str(row.get("top_drivers") or "")
        party = str(row.get("party") or "?")
        pill_class = (
            "dem" if party.upper() == "DEM" else "rep" if party.upper() == "REP" else "neutral"
        )
        prob_cell = "-" if prob is None else f"{float(prob) * 100:.1f}%"
        share_cell = "-" if share_mean is None else f"{float(share_mean) * 100:.1f}%"
        if p05 is None or p95 is None:
            interval_cell = "-"
        else:
            interval_cell = f"{float(p05) * 100:.1f}-{float(p95) * 100:.1f}%"
        return (
            "<tr>"
            f"<td>{html.escape(str(row.get('name') or row.get('option_id') or ''))}</td>"
            f"<td><span class='pill {pill_class}'>{html.escape(party)}</span></td>"
            f"<td>{prob_cell}</td>"
            f"<td>{share_cell}</td>"
            f"<td>{interval_cell}</td>"
            f"<td>{html.escape(drivers)}</td>"
            "</tr>"
        )

    @staticmethod
    def _plot_grid(*paths: Path | None) -> str:
        figs: list[str] = []
        labels = ["Vote-share KDE", "Polling trajectory", "Component drivers"]
        for path, label in zip(paths, labels, strict=False):
            if path is None:
                continue
            figs.append(
                "<figure>"
                f"<img src='plots/{html.escape(path.name)}' alt='{html.escape(label)}'>"
                f"<figcaption>{html.escape(label)}</figcaption>"
                "</figure>"
            )
        return "".join(figs) or "<p class='subtitle'>No plots emitted.</p>"

    @staticmethod
    def _meta_pills(race_meta: dict[str, Any]) -> str:
        bits: list[str] = []
        if race_meta.get("office_type"):
            bits.append(html.escape(str(race_meta["office_type"]).title()))
        if race_meta.get("geography"):
            bits.append(html.escape(str(race_meta["geography"])))
        if race_meta.get("cycle") is not None:
            bits.append(f"Cycle {race_meta['cycle']}")
        if race_meta.get("tier"):
            bits.append(f"Tier {race_meta['tier']}")
        return " · ".join(bits)

    @staticmethod
    def _race_narrative(race_meta: dict[str, Any], race_rows: pl.DataFrame) -> str:
        if race_rows.is_empty():
            return "No forecast emitted for this race."
        winner = (
            race_rows.filter(pl.col("winner_probability").is_not_null())
            .sort("winner_probability", descending=True)
            .row(0, named=True)
            if not race_rows.filter(pl.col("winner_probability").is_not_null()).is_empty()
            else None
        )
        if winner is None:
            return "Race tier withholds a probability call."
        prob = float(winner["winner_probability"])
        margin_lo = winner.get("vote_share_p05")
        margin_hi = winner.get("vote_share_p95")
        interval = (
            ""
            if margin_lo is None or margin_hi is None
            else f" (90% interval {float(margin_lo) * 100:.1f}-{float(margin_hi) * 100:.1f}%)"
        )
        return (
            f"Top option {winner.get('name') or winner.get('option_id')} "
            f"({winner.get('party')}) holds {prob * 100:.1f}% win probability"
            f"{interval}. Race tier: {race_meta.get('tier') or 'n/a'}."
        )
