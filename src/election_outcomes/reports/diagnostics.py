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
    ) -> str:
        rewards = (reward_card or {}).get("rewards", {})
        methodology_benchmark = methodology_benchmark or {}
        top = self._topline(race_forecasts)
        css = self._css()
        metric_cards = "\n".join(
            [
                self._metric_card("Projected margin", top["margin"], "mean two-party margin"),
                self._metric_card("Forecast rows", race_forecasts.height, "candidate/option rows"),
                self._metric_card("Sources", source_manifest.height, "manifested inputs"),
                self._metric_card(
                    "Backtest rows",
                    backtest_payload.get("row_count", 0),
                    "rolling-origin sample",
                ),
            ]
        )
        insight_strip = self._insight_strip(
            race_catalog, control_forecasts, ecosystem_forecasts, backtest_payload
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
<main class="page">
  <header class="hero">
    <div>
      <p class="eyebrow">Election forecast diagnostics</p>
      <h1>{html.escape(top["headline"])}</h1>
      <p class="lede">{html.escape(top["lede"])}</p>
    </div>
    <div class="hero-score">
      <span class="score-label">Top probability</span>
      <span class="score-value">{self._pct(top["probability"])}</span>
      <span class="score-subtitle">{html.escape(top["winner_name"])}</span>
    </div>
  </header>

  <section class="card-grid">
    {metric_cards}
  </section>

  <section class="insight-strip">
    {insight_strip}
  </section>

  <section class="panel">
    <div class="section-head">
      <p class="eyebrow">Current forecast</p>
      <h2>Race Probabilities And Vote Share</h2>
    </div>
    {self._forecast_table(race_forecasts)}
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="section-head">
        <p class="eyebrow">Why it moved</p>
        <h2>Model Drivers</h2>
      </div>
      {self._driver_cards(race_forecasts)}
    </div>
    <div class="panel">
      <div class="section-head">
        <p class="eyebrow">Model health</p>
        <h2>Trust Gates</h2>
      </div>
      {self._reward_grid(rewards)}
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="section-head">
        <p class="eyebrow">Backtest</p>
        <h2>Scorecard Snapshot</h2>
      </div>
      {self._backtest_summary(backtest_payload)}
    </div>
    <div class="panel">
      <div class="section-head">
        <p class="eyebrow">Methodology</p>
        <h2>Silver/FiveThirtyEight Benchmark</h2>
      </div>
      {self._benchmark_summary(methodology_benchmark)}
    </div>
  </section>

  <section class="panel">
    <div class="section-head">
      <p class="eyebrow">Visual diagnostics</p>
      <h2>Charts</h2>
    </div>
    {self._plot_sections(plot_manifest or {})}
  </section>

  <section class="panel compact">
    <div class="section-head">
      <p class="eyebrow">Audit trail</p>
      <h2>Run Metadata</h2>
    </div>
    {self._audit_summary(race_catalog, source_manifest, control_forecasts, ecosystem_forecasts)}
  </section>
</main>
</body>
</html>
"""

    @staticmethod
    def _topline(race_forecasts: pl.DataFrame) -> dict[str, Any]:
        frame = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return {
                "headline": "No trusted probability available",
                "lede": "All tracked races are below the probability-publication threshold.",
                "winner_name": "withheld",
                "probability": None,
                "margin": "n/a",
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
            modeled_seats = int(top_control.get("modeled_seats") or 0)
            if threshold and modeled_seats and modeled_seats < threshold:
                control_value = f"{party} {mean_count:.1f}/{modeled_seats}"
                control_detail = f"National threshold {threshold} is outside this modeled slice."
            else:
                control_value = f"{party} {cls._pct(top_control['control_probability'])}"
                control_detail = (
                    f"Mean modeled count: {mean_count:.1f}; threshold: {threshold or 'n/a'}."
                )
            cards.append(
                cls._insight_card(
                    "Control Readout",
                    control_value,
                    control_detail,
                )
            )
        if ecosystem_forecasts is not None and not ecosystem_forecasts.is_empty():
            top_risk = ecosystem_forecasts.sort("recount_probability", descending=True).row(
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
            rows.append(
                "<tr>"
                f"<td><strong>{html.escape(str(row.get('name') or row.get('option_id')))}</strong>"
                f"<span>{html.escape(str(row.get('race_id')))} · {party}</span></td>"
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
        cards = []
        for row in race_forecasts.sort(["race_id", "option_id"]).iter_rows(named=True):
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
        return f'<div class="driver-grid">{"".join(cards)}</div>'

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
                f"<td>{float(row.get('score', 0.0)):.2f}</td>"
                f"<td>{html.escape(str(row.get('current')))}</td>"
                "</tr>"
            )
        return (
            f'<p class="benchmark-score">{float(payload.get("summary_score", 0.0)):.2f}</p>'
            f"<p>{html.escape(str(payload.get('status')))}</p>"
            '<p><a href="silver_benchmark.html">Open source-backed benchmark details</a></p>'
            f"<table><thead><tr><th>Dimension</th><th>Score</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    @staticmethod
    def _plot_sections(plot_manifest: dict[str, list[dict[str, str]]]) -> str:
        sections = []
        for category, entries in plot_manifest.items():
            figures = []
            for entry in entries:
                title = html.escape(entry["title"])
                path = html.escape(entry["path"])
                figures.append(
                    '<figure class="plot-card">'
                    f'<img src="{path}" alt="{title}">'
                    f"<figcaption>{title}</figcaption></figure>"
                )
            if figures:
                sections.append(
                    f'<div class="plot-section"><h3>{html.escape(category.title())}</h3>'
                    f'<div class="plot-grid">{"".join(figures)}</div></div>'
                )
        return "\n".join(sections) if sections else '<p class="muted">No plots generated.</p>'

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
    def _css() -> str:
        return """
:root {
  color-scheme: light;
  --bg: #f6f4ef;
  --paper: #ffffff;
  --ink: #242424;
  --muted: #6b6b6b;
  --rule: #ded9cf;
  --blue: #1f77b4;
  --red: #d62728;
  --gold: #c87922;
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
.page { max-width: 1180px; margin: 0 auto; padding: 36px 22px 64px; }
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 240px;
  gap: 24px;
  align-items: stretch;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 24px;
  margin-bottom: 24px;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  margin: 0 0 8px;
}
h1 { font-size: 54px; line-height: 1; margin: 0; letter-spacing: 0; }
h2 { font-size: 24px; margin: 0; }
h3 { margin: 0 0 8px; }
.lede { color: var(--muted); max-width: 760px; font-size: 18px; }
.hero-score, .metric-card, .panel {
  background: var(--paper);
  border: 1px solid var(--rule);
  box-shadow: 0 1px 0 rgba(0,0,0,.04);
}
.hero-score { padding: 22px; display: flex; flex-direction: column; justify-content: center; }
.score-label, .metric-label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  font-weight: 700;
}
.score-value { font-size: 56px; font-weight: 800; line-height: 1; }
.score-subtitle { color: var(--muted); }
.card-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
.metric-card { padding: 16px; }
.metric-card strong { display: block; font-size: 28px; margin: 6px 0; }
.metric-card span:last-child { color: var(--muted); font-size: 13px; }
.insight-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 18px;
}
.insight-card { background: #242424; color: #fff; padding: 16px; min-height: 132px; }
.insight-card span {
  display: block;
  color: #d6d0c5;
  font-size: 12px;
  text-transform: uppercase;
  font-weight: 700;
}
.insight-card strong { display: block; font-size: 26px; margin: 8px 0; }
.insight-card p { margin: 0; color: #e8e2d8; font-size: 13px; }
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
.forecast-table td:nth-child(2),
.forecast-table td:nth-child(3) { font-size: 20px; font-weight: 700; }
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
.driver-grid, .reward-grid { display: grid; gap: 10px; }
.driver-card { border: 1px solid var(--rule); padding: 14px; background: #fbfaf7; }
.driver-card p { color: var(--muted); margin: 0 0 8px; }
.driver-card ul { margin: 0; padding-left: 18px; }
.reward-grid { grid-template-columns: repeat(2, 1fr); }
.reward { border: 1px solid var(--rule); padding: 10px 12px; background: #fbfaf7; }
.reward strong { display: block; font-size: 13px; }
.reward span { color: var(--muted); font-size: 12px; }
.reward.pass { border-left: 4px solid #3a8f5d; }
.reward.fail { border-left: 4px solid var(--gold); }
.reward.neutral { border-left: 4px solid #8d8d8d; }
.callout { background: #fff4dd; border-left: 4px solid var(--gold); padding: 10px 12px; }
.benchmark-score { font-size: 42px; font-weight: 800; margin: 0; }
.plot-section h3 { border-top: 1px solid var(--rule); padding-top: 16px; color: var(--muted); }
.plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.plot-card { margin: 0; border: 1px solid var(--rule); background: #fbfaf7; padding: 10px; }
.plot-card img { display: block; width: 100%; height: auto; }
.plot-card figcaption { color: var(--muted); font-size: 13px; margin-top: 8px; }
pre { overflow: auto; background: #fbfaf7; border: 1px solid var(--rule); padding: 12px; }
a { color: var(--blue); }
.muted { color: var(--muted); }
@media (max-width: 860px) {
  h1 { font-size: 38px; }
  .hero, .two-col, .card-grid, .insight-strip, .plot-grid { grid-template-columns: 1fr; }
  .hero-score { max-width: none; }
}
"""
