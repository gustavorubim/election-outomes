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
        rows = [
            self._row(
                "Poll inclusion and provenance",
                "Broad professional-poll inclusion with documented exclusions and source history.",
                self._has_table(source_manifest, "polls"),
                (
                    "Poll sources are manifest-tracked; richer pollster-quality metadata is "
                    "still needed."
                ),
            ),
            self._row(
                "Poll weighting and house effects",
                "Pollster quality, recency, sample-size diminishing returns, and house effects.",
                bool(model_config.get("polling", {}).get("pollster_house_effects")),
                (
                    "Current polling has recency/sample weighting; learned pollster effects "
                    "are pending."
                ),
            ),
            self._row(
                "Fundamentals layer",
                "Partisan lean, incumbency, fundraising, economic and demographic inputs.",
                self._has_table(source_manifest, "fundamentals"),
                (
                    "Fundamentals exist and can ridge-fit, but live state-level fundamentals "
                    "are pending."
                ),
            ),
            self._row(
                "Rolling-origin validation",
                "Fit on prior cycles and score held-out cycles before trusting components.",
                bool(backtest_payload.get("rolling_origin_executed")),
                (
                    "Rolling-origin refit now runs; fixture sample remains too small for "
                    "trust rewards."
                ),
            ),
            self._row(
                "Correlated election simulation",
                "National, geographic, and race-level correlated errors in simulations.",
                residual_covariance is not None and not residual_covariance.is_empty(),
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
                self._has_presidential_rows(race_catalog, race_forecasts),
                "Presidential scenario emits EC distribution for available state races.",
            ),
            self._row(
                "Insight surface",
                "Expose drivers, uncertainty, tipping points, and forecast-vs-actual diagnostics.",
                self._has_driver_columns(race_forecasts),
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
<thead><tr><th>Dimension</th><th>Score</th><th>Target</th><th>Current Engine</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Source Anchors</h2>
<ul>{sources}</ul>
</body>
</html>
"""

    @staticmethod
    def _row(dimension: str, target: str, implemented: bool, current: str) -> dict[str, Any]:
        return {
            "dimension": dimension,
            "target": target,
            "score": 1.0 if implemented else 0.35,
            "current": current,
        }

    @staticmethod
    def _status(score: float) -> str:
        if score >= 0.85:
            return "near public-methodology parity"
        if score >= 0.65:
            return "partial parity; core validation still maturing"
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


def benchmark_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
