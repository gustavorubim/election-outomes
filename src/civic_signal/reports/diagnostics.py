from __future__ import annotations

import html
import json
from typing import Any

import polars as pl


class DiagnosticsReport:
    def render(
        self,
        run_id: str,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        source_manifest: pl.DataFrame,
        backtest_payload: dict[str, Any],
        reward_card: dict[str, Any] | None = None,
        plot_manifest: dict[str, list[dict[str, str]]] | None = None,
        methodology_benchmark: dict[str, Any] | None = None,
        control_forecasts: pl.DataFrame | None = None,
        ecosystem_forecasts: pl.DataFrame | None = None,
        posterior_diagnostics: dict[str, Any] | None = None,
        fundamentals_prior: pl.DataFrame | None = None,
    ) -> str:
        rewards = (reward_card or {}).get("rewards", {})
        methodology_benchmark = methodology_benchmark or {}
        top = self._topline(race_forecasts, control_forecasts)
        css = self._css()
        narrative = self._narrative_blurb(race_forecasts, control_forecasts, backtest_payload)
        margin_label = "Threshold margin" if top.get("control_body") else "Projected margin"
        margin_detail = (
            "vs majority threshold" if top.get("control_body") else "mean two-party margin"
        )
        probability_label = "Control probability" if top.get("control_body") else "Top probability"
        overview_filenames = (
            ["electoral_college_distribution.png", "topline_electoral_swarm.png"]
            if top.get("control_body") == "president"
            else ["seat_count_histogram.png", "control_projection.png"]
        )
        overview_plot_cards, overview_plot_paths = self._priority_plot_cards(
            plot_manifest or {}, overview_filenames
        )
        distribution_plots = self._plot_sections(
            plot_manifest or {},
            categories=["distribution"],
            exclude_paths=overview_plot_paths,
        )
        driver_plots = self._plot_sections(
            plot_manifest or {},
            categories=["drivers"],
        )
        metric_cards = "\n".join(
            [
                self._metric_card(margin_label, top["margin"], margin_detail),
                self._metric_card("Forecast rows", race_forecasts.height, "candidate/option rows"),
                self._metric_card("Sources", source_manifest.height, "manifested inputs"),
                self._metric_card(
                    "Backtest rows",
                    backtest_payload.get("row_count", 0),
                    "rolling-origin sample",
                ),
            ]
        )
        projection_plots = self._plot_sections(
            plot_manifest or {},
            categories=["projection"],
            exclude_paths=overview_plot_paths,
        )
        model_quality_plots = self._plot_sections(
            plot_manifest or {},
            categories=["model_quality", "calibration", "trajectory", "stability", "benchmark"],
        )
        posterior_plots = self._plot_sections(plot_manifest or {}, categories=["posterior"])
        fundamentals_prior_plots = self._plot_sections(
            plot_manifest or {}, categories=["fundamentals_prior"]
        )
        fundamentals_prior_table = self._fundamentals_prior_summary(
            fundamentals_prior if fundamentals_prior is not None else pl.DataFrame()
        )
        posterior_section = (
            f"""
  <section id="posterior_diagnostics" class="panel plot-panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Bayesian fit</p>
        <h2>Posterior Diagnostics</h2>
      </div>
    </div>
    {self._posterior_diagnostics_summary(posterior_diagnostics or {})}
    {posterior_plots}
  </section>
"""
            if posterior_diagnostics
            and str(posterior_diagnostics.get("engine", "kalman")) != "kalman"
            else ""
        )
        fundamentals_prior_section = (
            f"""
  <section id="fundamentals_prior" class="panel plot-panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Bayesian prior</p>
        <h2>Fundamentals Prior</h2>
      </div>
    </div>
    {fundamentals_prior_table}
    {fundamentals_prior_plots}
  </section>
"""
            if fundamentals_prior is not None and not fundamentals_prior.is_empty()
            else ""
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forecast Diagnostics {html.escape(run_id)}</title>
  <style>{css}</style>
</head>
<body>
<main class="page diagnostics-dashboard">
  <header class="hero diagnostics-hero">
    <div>
      <p class="eyebrow">Election forecast diagnostics</p>
      <h1>{html.escape(top["headline"])}</h1>
      <p class="lede">{html.escape(top["lede"])}</p>
    </div>
    <div class="hero-score">
      <span class="score-label">{html.escape(probability_label)}</span>
      <span class="score-value">{self._pct(top["probability"])}</span>
      <span class="score-subtitle">{html.escape(top["winner_name"])}</span>
    </div>
  </header>

  <section class="kpi-strip" aria-label="Forecast summary">
        {metric_cards}
  </section>

  {f'<div class="narrative">{html.escape(narrative)}</div>' if narrative else ""}

  <section class="overview-layout">
    <div class="overview-main">
      <div class="section-head">
        <div>
          <p class="eyebrow">Executive overview</p>
          <h2>Distribution And Probability View</h2>
        </div>
      </div>
      <div class="overview-plot-grid">
        {overview_plot_cards or "<p class='muted'>No overview plots generated.</p>"}
      </div>
    </div>
    <aside class="overview-side">
      <div class="section-head compact-head">
        <div>
          <p class="eyebrow">Control readout</p>
          <h2>Scenario Scope</h2>
        </div>
      </div>
      <p class="scope-summary">
        {html.escape(str(race_catalog.height))} active races,
        {html.escape(str(source_manifest.height))} manifested inputs, and
        {html.escape(str(backtest_payload.get("row_count", 0)))}
        rolling-origin rows.
      </p>
      {self._control_table(control_forecasts)}
      <div class="section-head compact-head">
        <div>
          <p class="eyebrow">Tight races</p>
          <h2>Closest Contests</h2>
        </div>
      </div>
      {self._closest_race_list(race_forecasts)}
    </aside>
  </section>

  {posterior_section}

  {fundamentals_prior_section}

  <section class="panel plot-panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Outcome distribution</p>
        <h2>Where The Forecast Lives</h2>
      </div>
    </div>
    <p class="muted">
      Histogram of total seats per party across simulation draws and KDEs of vote
      share for the closest races. Distribution charts replace headline-mean bars
      so the uncertainty around the call is visible.
    </p>
    {distribution_plots or "<p class='muted'>No distribution plots emitted.</p>"}
  </section>

  <section class="panel plot-panel topline-plots">
    <div class="section-head">
      <div>
        <p class="eyebrow">Top-line forecast</p>
        <h2>Projection Views</h2>
      </div>
    </div>
    {projection_plots}
  </section>

  <section class="panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Current forecast</p>
        <h2>Race Probabilities And Vote Share</h2>
      </div>
    </div>
    <div class="table-shell forecast-shell">
      {self._forecast_table(race_forecasts)}
    </div>
  </section>

  <section class="panel plot-panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Drivers</p>
        <h2>Tipping Points And Component Decomposition</h2>
      </div>
    </div>
    <p class="muted">
      Tipping-point bars rank races by the probability that they decide the chamber
      majority. Waterfalls show how each component (polling, fundamentals, markets)
      moved the win-probability for the most competitive races.
    </p>
    {driver_plots or "<p class='muted'>No driver plots emitted.</p>"}
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Why it moved</p>
          <h2>Model Drivers</h2>
        </div>
      </div>
      {self._driver_cards(race_forecasts)}
    </div>
    <div class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Model health</p>
          <h2>Trust Gates</h2>
        </div>
      </div>
      {self._reward_grid(rewards)}
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Backtest</p>
          <h2>Scorecard Snapshot</h2>
        </div>
      </div>
      {self._backtest_summary(backtest_payload)}
    </div>
    <div class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Methodology</p>
          <h2>Silver/FiveThirtyEight Benchmark</h2>
        </div>
      </div>
      {self._benchmark_summary(methodology_benchmark)}
    </div>
  </section>

  <section class="panel plot-panel">
    <div class="section-head">
      <div>
        <p class="eyebrow">Model quality</p>
        <h2>Model Quality</h2>
      </div>
    </div>
    <p class="muted model-quality-note">
      Chain traces are MCMC-style split posterior simulation draws. The polling fit is a
      deterministic Kalman/state-space posterior, not a full MCMC sampler.
    </p>
    {model_quality_plots}
  </section>

  <section id="source_audit" class="panel compact">
    <div class="section-head">
      <div>
        <p class="eyebrow">Audit trail</p>
        <h2>Run Metadata</h2>
      </div>
    </div>
    {self._audit_summary(race_catalog, source_manifest, control_forecasts, ecosystem_forecasts)}
  </section>
</main>
</body>
</html>
"""

    @staticmethod
    def _topline(
        race_forecasts: pl.DataFrame, control_forecasts: pl.DataFrame | None = None
    ) -> dict[str, Any]:
        if control_forecasts is not None and not control_forecasts.is_empty():
            top_control = control_forecasts.sort("control_probability", descending=True).row(
                0, named=True
            )
            party = str(top_control["party"])
            probability = float(top_control["control_probability"])
            mean_count = float(top_control["seat_count_mean"])
            p10 = top_control.get("seat_count_p10")
            p90 = top_control.get("seat_count_p90")
            threshold = int(top_control.get("control_threshold") or 0)
            body = str(top_control.get("control_body") or "control")
            modeled = int(top_control.get("modeled_seats") or 0)
            holdovers = int(top_control.get("holdover_seats") or 0)
            party_name = {"DEM": "Democrats", "REP": "Republicans"}.get(party.upper(), party)
            president = control_forecasts.filter(pl.col("control_body") == "president")
            if not president.is_empty():
                top_control = president.sort("control_probability", descending=True).row(
                    0, named=True
                )
                party = str(top_control["party"])
                probability = float(top_control["control_probability"])
                mean_count = float(top_control["seat_count_mean"])
                threshold = int(top_control.get("control_threshold") or 270)
                party_name = {"DEM": "Democrats", "REP": "Republicans"}.get(party.upper(), party)
                return {
                    "headline": f"{party_name} favored in the Electoral College",
                    "lede": (
                        f"Mean simulated Electoral College count is {mean_count:.1f}; "
                        f"control threshold is {threshold}. Distribution plots below show "
                        "the simulated paths and uncertainty."
                    ),
                    "winner_name": f"{party} EC win",
                    "probability": probability,
                    "margin": f"{mean_count - threshold:+.1f} EV vs threshold",
                    "control_body": "president",
                    "party": party,
                    "mean_count": mean_count,
                    "threshold": threshold,
                    "p10": top_control.get("seat_count_p10"),
                    "p90": top_control.get("seat_count_p90"),
                }
            body_title = body.title()
            interval = ""
            if p10 is not None and p90 is not None:
                interval = f" Central 80% interval: {float(p10):.0f}-{float(p90):.0f} seats."
            modeled_note = (
                f" Includes {holdovers} holdovers and {modeled} modeled seats."
                if holdovers
                else f" Covers {modeled} modeled seats."
            )
            return {
                "headline": f"{party_name} favored for {body_title} control",
                "lede": (
                    f"{party} control probability is {probability * 100:.1f}% with "
                    f"{mean_count:.1f} projected seats against a {threshold}-seat threshold."
                    f"{interval}{modeled_note}"
                ),
                "winner_name": f"{party} control",
                "probability": probability,
                "margin": f"{mean_count - threshold:+.1f} seats vs threshold",
                "control_body": body,
                "party": party,
                "mean_count": mean_count,
                "threshold": threshold,
                "p10": p10,
                "p90": p90,
                "modeled_seats": modeled,
                "holdover_seats": holdovers,
            }
        frame = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return {
                "headline": "No trusted probability available",
                "lede": "All tracked races are below the probability-publication threshold.",
                "winner_name": "withheld",
                "probability": None,
                "margin": "n/a",
                "control_body": None,
            }
        top = frame.sort("winner_probability", descending=True).row(0, named=True)
        race = frame.filter(pl.col("race_id") == top["race_id"]).sort(
            "vote_share_mean", descending=True
        )
        margin = "n/a"
        if race.height >= 2 and "vote_share_mean" in race.columns:
            values = race["vote_share_mean"].to_list()
            margin = f"{(float(values[0]) - float(values[1])) * 100:+.1f} pts"
        winner_name = str(top.get("name") or top.get("option_id"))
        probability = float(top["winner_probability"])
        return {
            "headline": f"{winner_name} leads in {top['race_id']}",
            "lede": (
                "This page summarizes the current probabilistic forecast, uncertainty, "
                "model drivers, backtest gates, and audit artifacts."
            ),
            "winner_name": winner_name,
            "probability": probability,
            "margin": margin,
            "control_body": None,
        }

    @classmethod
    def _metric_card(cls, label: str, value: object, detail: str) -> str:
        if isinstance(value, float):
            value_text = cls._pct(value)
        else:
            value_text = html.escape(str(value))
        return (
            '<div class="metric-card">'
            f'<span class="metric-label">{html.escape(label)}</span>'
            f"<strong>{value_text}</strong>"
            f"<span>{html.escape(detail)}</span>"
            "</div>"
        )

    @classmethod
    def _insight_strip(
        cls,
        race_catalog: pl.DataFrame,
        control_forecasts: pl.DataFrame | None,
        ecosystem_forecasts: pl.DataFrame | None,
        backtest_payload: dict[str, Any],
    ) -> str:
        cards = []
        modeled_ev = 0
        presidential = pl.DataFrame()
        if not race_catalog.is_empty() and {"office_type", "seats"}.issubset(race_catalog.columns):
            presidential = race_catalog.filter(pl.col("office_type") == "president")
            if not presidential.is_empty():
                modeled_ev = int(presidential["seats"].sum())
        scope_value = f"{race_catalog.height} races"
        scope_note = "Report covers the active races selected for this run."
        if modeled_ev:
            scope_value = f"{modeled_ev} modeled EV"
            scope_note = (
                "Scenario covers a partial presidential state slice, not all 538 electoral votes."
                if modeled_ev < 270
                else "Scenario covers a national Electoral College frame."
            )
        cards.append(cls._insight_card("Scenario Scope", scope_value, scope_note))
        if control_forecasts is not None and not control_forecasts.is_empty():
            top_control = control_forecasts.sort("control_probability", descending=True).row(
                0, named=True
            )
            party = str(top_control["party"])
            mean_count = float(top_control["seat_count_mean"])
            threshold = int(top_control.get("control_threshold") or 0)
            holdovers = int(top_control.get("holdover_seats") or 0)
            modeled_seats = int(top_control.get("modeled_seats") or 0)
            control_value = f"{party} {cls._pct(top_control['control_probability'])}"
            control_detail = (
                f"Mean seats: {mean_count:.1f}; threshold: {threshold or 'n/a'}; "
                f"modeled: {modeled_seats}; holdovers: {holdovers}."
            )
            cards.append(
                cls._insight_card(
                    "Control Readout",
                    control_value,
                    control_detail,
                )
            )
        if (
            ecosystem_forecasts is not None
            and not ecosystem_forecasts.is_empty()
            and "recount_probability" in ecosystem_forecasts.columns
        ):
            risk_frame = ecosystem_forecasts.filter(pl.col("recount_probability").is_not_null())
            if not risk_frame.is_empty():
                top_risk = risk_frame.sort("recount_probability", descending=True).row(
                    0, named=True
                )
                cards.append(
                    cls._insight_card(
                        "Closest-Race Risk",
                        cls._pct(top_risk.get("recount_probability")),
                        f"Highest recount-risk race: {top_risk.get('race_id')}.",
                    )
                )
        trust_note = (
            "Trust gates remain experimental because the rolling-origin sample is below threshold."
            if backtest_payload.get("sample_size_too_small")
            else "Rolling-origin sample meets the configured trust threshold."
        )
        cards.append(
            cls._insight_card(
                "Backtest Trust",
                str(backtest_payload.get("row_count", 0)),
                trust_note,
            )
        )
        return "".join(cards)

    @staticmethod
    def _insight_card(label: str, value: str, detail: str) -> str:
        return (
            '<div class="insight-card">'
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<p>{html.escape(detail)}</p>"
            "</div>"
        )

    @staticmethod
    def _priority_plot_cards(
        plot_manifest: dict[str, list[dict[str, str]]], filenames: list[str]
    ) -> tuple[str, list[str]]:
        cards = []
        used_paths = []
        for filename in filenames:
            for entries in plot_manifest.values():
                match = next(
                    (entry for entry in entries if str(entry.get("path", "")).endswith(filename)),
                    None,
                )
                if match is None:
                    continue
                title = html.escape(str(match.get("title") or filename))
                path = html.escape(str(match["path"]))
                used_paths.append(str(match["path"]))
                cards.append(
                    '<figure class="plot-card overview-plot-card">'
                    f'<img src="{path}" alt="{title}" decoding="async">'
                    f"<figcaption>{title}</figcaption></figure>"
                )
                break
        return "\n".join(cards), used_paths

    @classmethod
    def _control_table(cls, control_forecasts: pl.DataFrame | None) -> str:
        if control_forecasts is None or control_forecasts.is_empty():
            return '<p class="muted">No chamber/control forecast generated.</p>'
        rows = []
        for row in control_forecasts.sort("control_probability", descending=True).iter_rows(
            named=True
        ):
            party = str(row.get("party") or "n/a")
            rows.append(
                "<tr>"
                f'<td><span class="party-token {party.lower()}">{html.escape(party)}</span></td>'
                f"<td>{cls._pct(row.get('control_probability'))}</td>"
                f"<td>{float(row.get('seat_count_mean') or 0):.1f}</td>"
                f"<td>{float(row.get('seat_count_p10') or 0):.0f}-"
                f"{float(row.get('seat_count_p90') or 0):.0f}</td>"
                "</tr>"
            )
        return (
            '<table class="compact-table control-table"><thead><tr><th>Party</th>'
            "<th>Control</th><th>Mean</th><th>80% range</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    @classmethod
    def _closest_race_list(cls, race_forecasts: pl.DataFrame) -> str:
        if race_forecasts.is_empty() or "winner_probability" not in race_forecasts.columns:
            return '<p class="muted">No race forecast rows available.</p>'
        frame = (
            race_forecasts.filter(pl.col("winner_probability").is_not_null())
            .with_columns((pl.col("winner_probability") - 0.5).abs().alias("_dist"))
            .sort("_dist")
            .head(5)
        )
        if frame.is_empty():
            return '<p class="muted">No competitive races available.</p>'
        items = []
        for row in frame.iter_rows(named=True):
            race_id = str(row.get("race_id") or "")
            name = str(row.get("name") or row.get("option_id") or "n/a")
            party = str(row.get("party") or "")
            probability = cls._pct(row.get("winner_probability"))
            link = (
                f'<a href="races/{html.escape(race_id)}.html">{html.escape(race_id)}</a>'
                if race_id
                else "n/a"
            )
            items.append(
                "<li>"
                f"<strong>{html.escape(name)}</strong>"
                f"<span>{link} · {html.escape(party)} · {probability}</span>"
                "</li>"
            )
        return f'<ol class="closest-list">{"".join(items)}</ol>'

    @classmethod
    def _forecast_table(cls, race_forecasts: pl.DataFrame) -> str:
        frame = race_forecasts.sort(["race_id", "winner_probability"], descending=[False, True])
        if frame.is_empty():
            return '<p class="muted">No forecast rows were generated.</p>'
        rows = []
        for row in frame.iter_rows(named=True):
            probability = row.get("winner_probability")
            share = row.get("vote_share_mean")
            low = row.get("vote_share_p05")
            high = row.get("vote_share_p95")
            bar_width = 0 if probability is None else max(0, min(100, float(probability) * 100))
            party = html.escape(str(row.get("party") or ""))
            race_id = str(row.get("race_id") or "")
            escaped_id = html.escape(race_id)
            race_link = (
                f"<a class='race-link' href='races/{escaped_id}.html'>{escaped_id}</a>"
                if race_id
                else "&mdash;"
            )
            rows.append(
                "<tr>"
                f"<td><strong>{html.escape(str(row.get('name') or row.get('option_id')))}</strong>"
                f"<span>{race_link} · {party}</span></td>"
                f"<td>{cls._pct(probability)}</td>"
                f"<td>{cls._pct(share)}</td>"
                f"<td>{cls._pct(low)} to {cls._pct(high)}</td>"
                f'<td><div class="prob-bar"><i style="width:{bar_width:.1f}%"></i></div></td>'
                "</tr>"
            )
        return (
            '<table class="forecast-table"><thead><tr><th>Candidate</th><th>Win prob</th>'
            "<th>Mean share</th><th>90% interval</th><th></th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    @staticmethod
    def _driver_cards(race_forecasts: pl.DataFrame) -> str:
        if race_forecasts.is_empty():
            return '<p class="muted">No driver rows available.</p>'
        frame = race_forecasts
        if "winner_probability" in frame.columns:
            frame = (
                frame.filter(pl.col("winner_probability").is_not_null())
                .with_columns((pl.col("winner_probability") - 0.5).abs().alias("_dist"))
                .sort("_dist")
                .head(12)
            )
        else:
            frame = frame.head(12)
        cards = []
        for row in frame.sort(["race_id", "option_id"]).iter_rows(named=True):
            contributions = row.get("component_contributions") or "{}"
            try:
                parsed = json.loads(str(contributions))
            except json.JSONDecodeError:
                parsed = {}
            parts = []
            for component, payload in parsed.items():
                share = float(payload.get("vote_share", 0.0))
                weight = float(payload.get("weight", 0.0))
                parts.append(
                    f"<li><b>{html.escape(component)}</b>: share {share * 100:.1f}%, "
                    f"weight {weight:.2f}</li>"
                )
            if not parts:
                parts.append("<li>No admitted component contribution surfaced.</li>")
            cards.append(
                '<div class="driver-card">'
                f"<h3>{html.escape(str(row.get('name') or row.get('option_id')))}</h3>"
                f"<p>{html.escape(str(row.get('top_drivers') or 'No drivers'))}</p>"
                f"<ul>{''.join(parts)}</ul>"
                "</div>"
            )
        return (
            '<p class="muted compact-note">Showing driver detail for the most competitive '
            f'{len(cards)} forecast rows.</p><div class="driver-grid">{"".join(cards)}</div>'
        )

    @staticmethod
    def _reward_grid(rewards: dict[str, Any]) -> str:
        if not rewards:
            return '<p class="muted">No reward card generated.</p>'
        items = []
        for key, value in rewards.items():
            passed = value.get("passed")
            status = "neutral" if passed is None else "pass" if passed else "fail"
            label = "external" if passed is None else "pass" if passed else "needs work"
            items.append(
                f'<div class="reward {status}"><strong>{html.escape(key)}</strong>'
                f"<span>{html.escape(label)}</span></div>"
            )
        return f'<div class="reward-grid">{"".join(items)}</div>'

    @staticmethod
    def _posterior_diagnostics_summary(diagnostics: dict[str, Any]) -> str:
        if not diagnostics:
            return '<p class="muted">No posterior diagnostics were emitted.</p>'
        fields = [
            ("Engine", diagnostics.get("engine")),
            ("Parameterization", diagnostics.get("parameterization", "n/a")),
            ("Draws", diagnostics.get("draw_count")),
            ("Race-options", diagnostics.get("race_option_count")),
            ("Polls", diagnostics.get("poll_count")),
            ("Divergences", diagnostics.get("divergences")),
            ("R-hat max", diagnostics.get("r_hat_max")),
            ("ESS min", diagnostics.get("ess_min")),
            ("Fallback", diagnostics.get("fallback_used")),
        ]
        rows = [
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape('n/a' if value is None else str(value))}</td>"
            "</tr>"
            for label, value in fields
        ]
        return (
            '<table class="compact-table"><thead><tr><th>Metric</th><th>Value</th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    @staticmethod
    def _fundamentals_prior_summary(fundamentals_prior: pl.DataFrame) -> str:
        if fundamentals_prior.is_empty():
            return '<p class="muted">No fundamentals prior artifact was emitted.</p>'
        method_counts = (
            fundamentals_prior.group_by("prior_method").agg(pl.len().alias("count")).to_dicts()
            if "prior_method" in fundamentals_prior.columns
            else []
        )
        payload = {
            "rows": fundamentals_prior.height,
            "methods": method_counts,
            "mean_sd_logit": float(fundamentals_prior["sd_logit"].mean())
            if "sd_logit" in fundamentals_prior.columns
            else None,
        }
        return f"<pre>{html.escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>"

    @staticmethod
    def _backtest_summary(payload: dict[str, Any]) -> str:
        metrics = payload.get("metrics", {})
        rows = []
        for component, values in metrics.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(component))}</td>"
                f"<td>{float(values.get('brier', 0.0)):.3f}</td>"
                f"<td>{float(values.get('log_score', 0.0)):.3f}</td>"
                f"<td>{float(values.get('expected_calibration_error', 0.0)):.3f}</td>"
                "</tr>"
            )
        warning = ""
        if payload.get("sample_size_too_small"):
            warning = (
                '<p class="callout">Backtest sample is below the configured trust threshold; '
                "calibration and component admission remain experimental.</p>"
            )
        return (
            f"{warning}<table><thead><tr><th>Component</th><th>Brier</th><th>Log</th>"
            f"<th>ECE</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    @staticmethod
    def _benchmark_summary(payload: dict[str, Any]) -> str:
        if not payload:
            return '<p class="muted">No methodology benchmark generated.</p>'
        rows = []
        for row in payload.get("rows", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('dimension')))}</td>"
                f"<td>{html.escape(str(row.get('tier', 'n/a')))}</td>"
                f"<td>{float(row.get('score', 0.0)):.2f}</td>"
                f"<td>{html.escape(str(row.get('current')))}</td>"
                "</tr>"
            )
        return (
            f'<p class="benchmark-score">{float(payload.get("summary_score", 0.0)):.2f}</p>'
            f"<p>{html.escape(str(payload.get('status')))}</p>"
            '<p><a href="silver_benchmark.html">Open source-backed benchmark details</a></p>'
            f"<table><thead><tr><th>Dimension</th><th>Tier</th><th>Score</th>"
            f"<th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    @staticmethod
    def _plot_sections(
        plot_manifest: dict[str, list[dict[str, str]]],
        categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> str:
        sections = []
        category_filter = set(categories or [])
        excluded = set(exclude_categories or [])
        excluded_paths = set(exclude_paths or [])
        for category, entries in plot_manifest.items():
            if category_filter and category not in category_filter:
                continue
            if category in excluded:
                continue
            figures = []
            for entry in entries:
                title = html.escape(entry["title"])
                path = html.escape(entry["path"])
                if entry["path"] in excluded_paths:
                    continue
                figures.append(
                    '<figure class="plot-card">'
                    f'<img src="{path}" alt="{title}" loading="lazy" decoding="async">'
                    f"<figcaption>{title}</figcaption></figure>"
                )
            if figures:
                category_label = category.replace("_", " ").title()
                sections.append(
                    f'<div class="plot-section"><h3>{html.escape(category_label)}</h3>'
                    f'<div class="plot-grid">{"".join(figures)}</div></div>'
                )
        return "\n".join(sections) if sections else '<p class="muted">No plots generated.</p>'

    @staticmethod
    def _single_plot_cards(
        plot_manifest: dict[str, list[dict[str, str]]], filenames: list[str]
    ) -> str:
        cards = [
            DiagnosticsReport._single_plot_card(plot_manifest, filename) for filename in filenames
        ]
        return f'<div class="summary-plot-grid">{"".join(cards)}</div>'

    @staticmethod
    def _single_plot_card(plot_manifest: dict[str, list[dict[str, str]]], filename: str) -> str:
        for entries in plot_manifest.values():
            for entry in entries:
                if str(entry.get("path", "")).endswith(filename):
                    title = html.escape(str(entry.get("title") or filename))
                    path = html.escape(str(entry["path"]))
                    return (
                        '<figure class="plot-card summary-plot-card">'
                        f'<img src="{path}" alt="{title}" loading="lazy" decoding="async">'
                        f"<figcaption>{title}</figcaption></figure>"
                    )
        return f'<p class="muted">Top-line plot was not generated: {html.escape(filename)}.</p>'

    @staticmethod
    def _audit_summary(
        race_catalog: pl.DataFrame,
        source_manifest: pl.DataFrame,
        control_forecasts: pl.DataFrame | None,
        ecosystem_forecasts: pl.DataFrame | None,
    ) -> str:
        tier_counts = (
            race_catalog.group_by("tier").agg(pl.len().alias("count")).to_dicts()
            if not race_catalog.is_empty()
            else []
        )
        control_rows = 0 if control_forecasts is None else control_forecasts.height
        ecosystem_rows = 0 if ecosystem_forecasts is None else ecosystem_forecasts.height
        payload = {
            "tier_counts": tier_counts,
            "source_rows": source_manifest.height,
            "control_rows": control_rows,
            "ecosystem_rows": ecosystem_rows,
        }
        return f"<pre>{html.escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>"

    @staticmethod
    def _pct(value: object) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return html.escape(str(value))

    @staticmethod
    def _narrative_blurb(
        race_forecasts: pl.DataFrame,
        control_forecasts: pl.DataFrame | None,
        backtest_payload: dict[str, Any],
    ) -> str:
        """One-paragraph summary derived from existing run artifacts."""
        lines: list[str] = []
        if control_forecasts is not None and not control_forecasts.is_empty():
            top = control_forecasts.sort("control_probability", descending=True).row(0, named=True)
            body = str(top.get("control_body") or "control")
            party = str(top.get("party") or "?")
            prob = top.get("control_probability")
            mean_seats = top.get("seat_count_mean")
            threshold = top.get("control_threshold")
            if prob is not None and mean_seats is not None and threshold is not None:
                lines.append(
                    f"{party} forecast majority of the {body} with "
                    f"{float(prob) * 100:.1f}% probability "
                    f"(mean {float(mean_seats):.1f} of {threshold} seats)."
                )
        if not race_forecasts.is_empty() and "winner_probability" in race_forecasts.columns:
            close = race_forecasts.filter(
                pl.col("winner_probability").is_not_null()
                & (pl.col("winner_probability") > 0.0)
                & (pl.col("winner_probability") < 1.0)
            ).with_columns((pl.col("winner_probability") - 0.5).abs().alias("_dist"))
            if not close.is_empty():
                tightest = close.sort("_dist").row(0, named=True)
                lines.append(
                    f"Closest race: {tightest['race_id']} "
                    f"({float(tightest['winner_probability']) * 100:.1f}% top probability)."
                )
        rolling = backtest_payload.get("rolling_origin") or {}
        cycles = rolling.get("cycles") or []
        rows = backtest_payload.get("row_count")
        if cycles and rows:
            lines.append(
                f"Rolling-origin trained on {rows} rows across cycles "
                f"{', '.join(str(c) for c in cycles)}."
            )
        return " ".join(lines)

    @staticmethod
    def _css() -> str:
        return """
:root {
  color-scheme: light;
  --bg: #f6f4ef;
  --paper: #ffffff;
  --ink: #242424;
  --muted: #6b6b6b;
  --rule: #ded9cf;
  --panel: #fbfaf7;
  --blue: #1f77b4;
  --red: #d62728;
  --gold: #c87922;
  --green: #3a8f5d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
.page { max-width: 1180px; margin: 0 auto; padding: 34px 24px 64px; }
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 230px;
  gap: 18px;
  align-items: stretch;
  margin-bottom: 18px;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  margin: 0 0 8px;
}
h1 { font-size: 42px; line-height: 1.08; margin: 0; letter-spacing: 0; max-width: 860px; }
h2 { font-size: 24px; margin: 0; }
h3 { margin: 0 0 8px; }
.lede { color: var(--muted); max-width: 840px; font-size: 16px; margin-bottom: 0; }
.hero-score, .metric-card, .panel, .overview-main, .overview-side {
  background: var(--paper);
  border: 1px solid var(--rule);
  box-shadow: 0 1px 0 rgba(0,0,0,.04);
  border-radius: 8px;
}
.hero-score { padding: 20px; display: flex; flex-direction: column; justify-content: center; }
.score-label, .metric-label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  font-weight: 700;
}
.score-value { font-size: 44px; font-weight: 800; line-height: 1; letter-spacing: 0; }
.score-subtitle { color: var(--muted); }
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.metric-card { padding: 15px 16px; min-width: 0; }
.metric-card strong { display: block; font-size: 26px; margin: 5px 0; letter-spacing: 0; }
.metric-card span:last-child { color: var(--muted); font-size: 13px; }
.overview-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, .75fr);
  gap: 18px;
  align-items: start;
  margin: 18px 0;
}
.overview-main, .overview-side { padding: 18px; }
.overview-plot-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.overview-plot-card img { max-height: 360px; object-fit: contain; }
.insight-strip {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-bottom: 14px;
}
.insight-card {
  background: var(--panel);
  border: 1px solid var(--rule);
  color: var(--ink);
  padding: 12px;
  border-radius: 8px;
}
.insight-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  font-weight: 700;
}
.insight-card strong { display: block; font-size: 22px; margin: 6px 0; }
.insight-card p { margin: 0; color: var(--muted); font-size: 13px; }
.panel { padding: 20px; margin-bottom: 18px; }
.compact { margin-bottom: 0; }
.two-col { display: grid; grid-template-columns: 1.15fr .85fr; gap: 18px; }
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 16px;
  margin-bottom: 16px;
}
.compact-head { margin-top: 16px; margin-bottom: 10px; }
.compact-head:first-child { margin-top: 0; }
.scope-summary {
  color: var(--muted);
  margin: 0 0 14px;
  font-size: 14px;
}
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th {
  text-align: left;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--rule);
}
td { border-bottom: 1px solid #eee9df; padding: 10px 8px; vertical-align: top; }
td span { display: block; color: var(--muted); font-size: 12px; }
.table-shell {
  border: 1px solid var(--rule);
  border-radius: 8px;
  overflow: auto;
  background: var(--paper);
}
.forecast-shell { max-height: 560px; }
.forecast-shell thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--paper);
}
.compact-table { font-size: 13px; }
.control-table td:nth-child(2),
.forecast-table td:nth-child(2),
.forecast-table td:nth-child(3) { font-size: 20px; font-weight: 700; }
.party-token {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
}
.party-token.dem { color: var(--blue); background: rgba(31, 119, 180, .12); }
.party-token.rep { color: var(--red); background: rgba(214, 39, 40, .12); }
.prob-bar {
  width: 150px;
  height: 10px;
  background: #eee9df;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 6px;
}
.prob-bar i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--blue), #69a9d8);
}
.closest-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 9px;
}
.closest-list li {
  border-bottom: 1px solid #eee9df;
  padding-bottom: 9px;
}
.closest-list li:last-child { border-bottom: none; padding-bottom: 0; }
.closest-list strong { display: block; font-size: 13px; }
.closest-list span { color: var(--muted); font-size: 12px; }
.driver-grid, .reward-grid { display: grid; gap: 10px; }
.driver-grid { max-height: 520px; overflow: auto; padding-right: 4px; }
.driver-card {
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 14px;
  background: var(--panel);
}
.driver-card p { color: var(--muted); margin: 0 0 8px; }
.driver-card ul { margin: 0; padding-left: 18px; }
.reward-grid { grid-template-columns: repeat(2, 1fr); }
.reward {
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--panel);
}
.reward strong { display: block; font-size: 13px; }
.reward span { color: var(--muted); font-size: 12px; }
.reward.pass { border-left: 4px solid var(--green); }
.reward.fail { border-left: 4px solid var(--gold); }
.reward.neutral { border-left: 4px solid #8d8d8d; }
.callout { background: #fff4dd; border-left: 4px solid var(--gold); padding: 10px 12px; }
.model-quality-note { margin-top: -6px; margin-bottom: 18px; }
.narrative {
  background: #fbf6e8;
  border-left: 4px solid var(--gold);
  border-radius: 6px;
  padding: 12px 16px;
  margin: 16px 0 22px;
  color: var(--ink);
  font-size: 14px;
  line-height: 1.55;
}
.benchmark-score { font-size: 42px; font-weight: 800; margin: 0; }
.plot-section h3 { border-top: 1px solid var(--rule); padding-top: 16px; color: var(--muted); }
.plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.plot-card {
  margin: 0;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--panel);
  padding: 10px;
}
.plot-card img {
  display: block;
  width: 100%;
  max-height: 460px;
  object-fit: contain;
  background: #fff;
}
.plot-card figcaption { color: var(--muted); font-size: 13px; margin-top: 8px; }
pre {
  overflow: auto;
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 12px;
}
a { color: var(--blue); }
.muted { color: var(--muted); }
.compact-note { font-size: 13px; margin-top: 0; }
@media (max-width: 860px) {
  h1 { font-size: 38px; }
  .hero, .two-col, .kpi-strip, .insight-strip, .plot-grid, .overview-layout,
  .overview-plot-grid { grid-template-columns: 1fr; }
  .hero-score { max-width: none; }
}
"""
