from __future__ import annotations

import html
import json
from typing import Any, ClassVar

import polars as pl


class SilverStyleBenchmark:
    """Compare the current engine with public Silver/FiveThirtyEight methodology traits."""

    SOURCES: ClassVar[list[dict[str, str]]] = [
        {
            "name": "Silver Bulletin polling averages methodology",
            "url": "https://www.natesilver.net/p/silver-bulletin-polling-average-methodology",
            "used_for": "poll inclusion, pollster quality, recency, house effects, uncertainty",
        },
        {
            "name": "Silver Bulletin 2024 presidential forecast landing page",
            "url": "https://www.natesilver.net/p/nate-silver-2024-president-election-polls-model",
            "used_for": "presidential forecast framing, state/national polling inference",
        },
        {
            "name": "FiveThirtyEight congressional model methodology",
            "url": "https://fivethirtyeight.com/methodology/how-fivethirtyeights-house-and-senate-models-work/",
            "used_for": "fundamentals, adjusted polling, correlated errors, simulations",
        },
        {
            "name": "FiveThirtyEight 2020 forecast design",
            "url": "https://fivethirtyeight.com/features/how-we-designed-the-look-of-our-2020-forecast/",
            "used_for": "top-line probability and simulation-result visual framing",
        },
    ]

    def evaluate(
        self,
        model_config: dict[str, Any],
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        backtest_payload: dict[str, Any],
        residual_covariance: pl.DataFrame | None,
        source_manifest: pl.DataFrame,
    ) -> dict[str, Any]:
        covariance_sample = self._covariance_sample_size(residual_covariance)
        modeled_ev = self._modeled_presidential_seats(race_catalog)
        sample_too_small = bool(backtest_payload.get("sample_size_too_small"))
        rows = [
            self._row(
                "Poll inclusion and provenance",
                "Broad professional-poll inclusion with documented exclusions and source history.",
                "functional" if self._has_table(source_manifest, "polls") else "absent",
                (
                    "Poll sources are manifest-tracked; richer pollster-quality metadata is "
                    "still needed."
                ),
            ),
            self._row(
                "Poll weighting and house effects",
                "Pollster quality, recency, sample-size diminishing returns, and house effects.",
                (
                    "functional"
                    if model_config.get("polling", {}).get("half_life_days")
                    else "scaffold"
                ),
                (
                    "Current polling has recency/sample weighting; learned pollster effects "
                    "are pending."
                ),
            ),
            self._row(
                "Fundamentals layer",
                "Partisan lean, incumbency, fundraising, economic and demographic inputs.",
                (
                    "scaffold"
                    if self._has_table(source_manifest, "fundamentals") and sample_too_small
                    else "functional"
                    if self._has_table(source_manifest, "fundamentals")
                    else "absent"
                ),
                (
                    "Fundamentals exist and can ridge-fit, but live state-level fundamentals "
                    "are pending."
                ),
            ),
            self._row(
                "Rolling-origin validation",
                "Fit on prior cycles and score held-out cycles before trusting components.",
                ("functional" if backtest_payload.get("rolling_origin_executed") else "absent"),
                (
                    "Rolling-origin refit now runs; fixture sample remains too small for "
                    "trust rewards."
                ),
            ),
            self._row(
                "Correlated election simulation",
                "National, geographic, and race-level correlated errors in simulations.",
                (
                    "functional"
                    if covariance_sample >= 30
                    else "scaffold"
                    if residual_covariance is not None and not residual_covariance.is_empty()
                    else "functional"
                ),
                (
                    "Simulation can consume residual covariance; broad historical covariance "
                    "is pending."
                ),
            ),
            self._row(
                "Electoral College/top-line reporting",
                (
                    "Translate state probabilities into Electoral College outcomes and "
                    "top-line visuals."
                ),
                (
                    "functional"
                    if modeled_ev >= 270
                    else "scaffold"
                    if self._has_presidential_rows(race_catalog, race_forecasts)
                    else "absent"
                ),
                (
                    f"Presidential reporting is wired, but only {modeled_ev} electoral votes "
                    "are present in this scenario."
                ),
            ),
            self._row(
                "Insight surface",
                "Expose drivers, uncertainty, tipping points, and forecast-vs-actual diagnostics.",
                "functional" if self._has_driver_columns(race_forecasts) else "scaffold",
                "Diagnostics include driver rows and comparison narratives.",
            ),
        ]
        score = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
        return {
            "benchmark_name": "Silver/FiveThirtyEight public-methodology benchmark",
            "summary_score": score,
            "status": self._status(score),
            "rows": rows,
            "sources": self.SOURCES,
            "note": (
                "This is a methodology/readiness benchmark against public descriptions, not a "
                "claim to reproduce proprietary Silver Bulletin or FiveThirtyEight forecasts."
            ),
        }

    @staticmethod
    def html(payload: dict[str, Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(row['dimension'])}</td>"
            f"<td>{html.escape(row['tier'])}</td>"
            f"<td>{html.escape(str(row['score']))}</td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['current'])}</td>"
            "</tr>"
            for row in payload["rows"]
        )
        sources = "".join(
            f'<li><a href="{html.escape(source["url"])}">{html.escape(source["name"])}</a>'
            f": {html.escape(source['used_for'])}</li>"
            for source in payload["sources"]
        )
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Silver/FiveThirtyEight Benchmark</title></head>
<body>
<h1>{html.escape(payload["benchmark_name"])}</h1>
<p>Status: <strong>{html.escape(payload["status"])}</strong></p>
<p>Summary score: {payload["summary_score"]:.3f}</p>
<p>{html.escape(payload["note"])}</p>
<table>
<thead>
<tr><th>Dimension</th><th>Tier</th><th>Score</th><th>Target</th><th>Current</th></tr>
</thead>
<tbody>{rows}</tbody>
</table>
<h2>Source Anchors</h2>
<ul>{sources}</ul>
</body>
</html>
"""

    @staticmethod
    def _row(dimension: str, target: str, tier: str, current: str) -> dict[str, Any]:
        score_by_tier = {
            "absent": 0.0,
            "scaffold": 0.33,
            "functional": 0.66,
            "production": 1.0,
        }
        normalized_tier = tier if tier in score_by_tier else "absent"
        return {
            "dimension": dimension,
            "target": target,
            "tier": normalized_tier,
            "score": score_by_tier[normalized_tier],
            "current": current,
        }

    @staticmethod
    def _status(score: float) -> str:
        if score >= 0.85:
            return "near public-methodology parity"
        if score >= 0.65:
            return "partial parity; core validation still maturing"
        if score >= 0.45:
            return "partial parity; core validation maturing"
        return "early parity; major statistical layers pending"

    @staticmethod
    def _has_table(source_manifest: pl.DataFrame, table: str) -> bool:
        return (
            not source_manifest.is_empty()
            and "table" in source_manifest.columns
            and source_manifest.filter(pl.col("table") == table).height > 0
        )

    @staticmethod
    def _has_presidential_rows(race_catalog: pl.DataFrame, race_forecasts: pl.DataFrame) -> bool:
        return (
            not race_catalog.is_empty()
            and not race_forecasts.is_empty()
            and race_catalog.filter(pl.col("office_type") == "president").height > 0
        )

    @staticmethod
    def _has_driver_columns(race_forecasts: pl.DataFrame) -> bool:
        required = {"top_drivers", "component_contributions", "uncertainty_explanation"}
        return not race_forecasts.is_empty() and required.issubset(set(race_forecasts.columns))

    @staticmethod
    def _modeled_presidential_seats(race_catalog: pl.DataFrame) -> int:
        if race_catalog.is_empty() or "seats" not in race_catalog.columns:
            return 0
        presidential = race_catalog.filter(pl.col("office_type") == "president")
        return 0 if presidential.is_empty() else int(presidential["seats"].sum())

    @staticmethod
    def _covariance_sample_size(residual_covariance: pl.DataFrame | None) -> int:
        if (
            residual_covariance is None
            or residual_covariance.is_empty()
            or "sample_size" not in residual_covariance.columns
        ):
            return 0
        return int(residual_covariance["sample_size"].max())


def benchmark_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
