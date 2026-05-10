from __future__ import annotations

import html
import json
from typing import Any, ClassVar

import polars as pl


class SilverStyleBenchmark:
    """Compare the current engine with public Silver/FiveThirtyEight methodology traits."""

    TIER_SCALE: ClassVar[dict[str, float]] = {
        "absent": 0.0,
        "scaffold": 0.33,
        "functional": 0.66,
        "production": 1.0,
    }
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
        poll_trajectory: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        covariance_sample = self._covariance_sample_size(residual_covariance)
        modeled_ev = self._modeled_presidential_seats(race_catalog)
        sample_too_small = bool(backtest_payload.get("sample_size_too_small"))
        ablations = dict(backtest_payload.get("ablations", {}))
        bayesian = dict(model_config.get("bayesian", {}))
        component_admission = dict(model_config.get("component_admission", {}))
        trusted_components = dict(
            component_admission.get("trusted_components")
            or model_config.get("trusted_components", {})
        )
        active_engine = str(model_config.get("_inference_engine") or "").lower()
        bayes_nuts = (
            active_engine == "bayes"
            and bool(bayesian.get("enabled"))
            and str(bayesian.get("backend")) == "nuts"
        )
        trusted_polling = bool(trusted_components.get("polling"))
        trusted_validation = bool(backtest_payload.get("rolling_origin_executed")) and not (
            sample_too_small
        )
        ensemble_admitted = bool(
            dict(ablations.get("ensemble", {})).get("beats_or_matches_baseline")
        )
        polling_admitted = bool(dict(ablations.get("polling", {})).get("beats_or_matches_baseline"))
        production_polling = (
            bayes_nuts
            and trusted_validation
            and trusted_polling
            and polling_admitted
            and self._has_table(source_manifest, "polls")
        )
        production_fundamentals = (
            bayes_nuts
            and trusted_validation
            and self._has_table(source_manifest, "fundamentals")
            and bool(bayesian.get("fundamentals_prior"))
        )
        production_validation = trusted_validation and ensemble_admitted
        production_simulation = int(model_config.get("simulation_count", 0) or 0) >= 1000 and bool(
            dict(model_config.get("performance", {})).get("parallel", False)
        )
        production_topline = self._has_control_rows(race_catalog) or modeled_ev >= 270
        trajectory_support = self._trajectory_support(
            model_config=model_config,
            race_forecasts=race_forecasts,
            backtest_payload=backtest_payload,
            poll_trajectory=poll_trajectory,
        )
        rows = [
            self._row(
                "Poll inclusion and provenance",
                "Broad professional-poll inclusion with documented exclusions and source history.",
                "production"
                if production_polling
                else "functional"
                if self._has_table(source_manifest, "polls")
                else "absent",
                (
                    "Poll sources are manifest-tracked and the admitted Bayesian polling "
                    "component passes rolling-origin evidence."
                    if production_polling
                    else "Poll sources are manifest-tracked; richer pollster-quality metadata "
                    "is still needed."
                ),
            ),
            self._row(
                "Poll weighting and house effects",
                "Pollster quality, recency, sample-size diminishing returns, and house effects.",
                (
                    "production"
                    if production_polling
                    else "functional"
                    if model_config.get("polling", {}).get("half_life_days")
                    else "scaffold"
                ),
                (
                    "NumPyro/NUTS polling uses recency structure, non-centered hierarchy, "
                    "and learned pollster-effect artifacts."
                    if production_polling
                    else "Current polling has recency/sample weighting; learned pollster "
                    "effects are pending."
                ),
            ),
            self._row(
                "Polling trajectory/Kalman support",
                "State-space or trajectory-aware polling updates with auditable artifacts.",
                trajectory_support["tier"],
                trajectory_support["current"],
            ),
            self._row(
                "Fundamentals layer",
                "Partisan lean, incumbency, fundraising, economic and demographic inputs.",
                (
                    "production"
                    if production_fundamentals
                    else "scaffold"
                    if self._has_table(source_manifest, "fundamentals") and sample_too_small
                    else "functional"
                    if self._has_table(source_manifest, "fundamentals")
                    else "absent"
                ),
                (
                    "Fundamentals are ridge-fit and fed into the Bayesian polling model as "
                    "an auditable Election-Day prior."
                    if production_fundamentals
                    else "Fundamentals exist and can ridge-fit, but live state-level "
                    "fundamentals are pending."
                ),
            ),
            self._row(
                "Rolling-origin validation",
                "Fit on prior cycles and score held-out cycles before trusting components.",
                "production"
                if production_validation
                else "functional"
                if backtest_payload.get("rolling_origin_executed")
                else "absent",
                (
                    "Rolling-origin evidence is large enough for trust and the admitted "
                    "ensemble beats the baseline."
                    if production_validation
                    else "Rolling-origin refit now runs; fixture sample remains too small for "
                    "trust rewards."
                ),
            ),
            self._row(
                "Correlated election simulation",
                "National, geographic, and race-level correlated errors in simulations.",
                (
                    "production"
                    if production_simulation
                    else "functional"
                    if covariance_sample >= 30
                    else "scaffold"
                    if residual_covariance is not None and not residual_covariance.is_empty()
                    else "functional"
                ),
                (
                    "Simulation runs through the configured parallel Numba kernel and consumes "
                    "Bayesian posterior uncertainty."
                    if production_simulation
                    else "Simulation can consume residual covariance; broad historical "
                    "covariance is pending."
                ),
            ),
            self._row(
                "Electoral College/top-line reporting",
                (
                    "Translate state probabilities into Electoral College outcomes and "
                    "top-line visuals."
                ),
                (
                    "production"
                    if production_topline
                    else "functional"
                    if modeled_ev >= 270
                    else "scaffold"
                    if self._has_presidential_rows(race_catalog, race_forecasts)
                    else "absent"
                ),
                (
                    "Top-line control reporting is available for the configured presidential "
                    "or chamber-control scope."
                    if production_topline
                    else f"Presidential reporting is wired, but only {modeled_ev} electoral "
                    "votes are present in this scenario."
                ),
            ),
            self._row(
                "Insight surface",
                "Expose drivers, uncertainty, tipping points, and forecast-vs-actual diagnostics.",
                "production"
                if self._has_driver_columns(race_forecasts) and production_validation
                else "functional"
                if self._has_driver_columns(race_forecasts)
                else "scaffold",
                (
                    "Diagnostics include driver rows, uncertainty explanations, validation "
                    "context, and posterior methodology panels."
                    if self._has_driver_columns(race_forecasts) and production_validation
                    else "Diagnostics include driver rows and comparison narratives."
                ),
            ),
        ]
        score = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
        return {
            "benchmark_name": "Silver/FiveThirtyEight public-methodology benchmark",
            "summary_score": score,
            "status": self._status(score),
            "tier_scale": self.TIER_SCALE,
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
        scale = ", ".join(
            f"{tier}={score}" for tier, score in payload.get("tier_scale", {}).items()
        )
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Silver/FiveThirtyEight Benchmark</title></head>
<body>
<h1>{html.escape(payload["benchmark_name"])}</h1>
<p>Status: <strong>{html.escape(payload["status"])}</strong></p>
<p>Summary score: {payload["summary_score"]:.3f}</p>
<p>Tier scale: {html.escape(scale)}</p>
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
        normalized_tier = tier if tier in SilverStyleBenchmark.TIER_SCALE else "absent"
        return {
            "dimension": dimension,
            "target": target,
            "tier": normalized_tier,
            "score": SilverStyleBenchmark.TIER_SCALE[normalized_tier],
            "current": current,
        }

    @staticmethod
    def _status(score: float) -> str:
        if score >= 0.99:
            return "production parity for configured scope"
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
    def _has_control_rows(race_catalog: pl.DataFrame) -> bool:
        return (
            not race_catalog.is_empty()
            and "control_body" in race_catalog.columns
            and race_catalog.filter(pl.col("control_body").is_not_null()).height > 0
        )

    @staticmethod
    def _has_driver_columns(race_forecasts: pl.DataFrame) -> bool:
        required = {"top_drivers", "component_contributions", "uncertainty_explanation"}
        return not race_forecasts.is_empty() and required.issubset(set(race_forecasts.columns))

    @classmethod
    def _trajectory_support(
        cls,
        model_config: dict[str, Any],
        race_forecasts: pl.DataFrame,
        backtest_payload: dict[str, Any],
        poll_trajectory: pl.DataFrame | None = None,
    ) -> dict[str, str]:
        keywords = ("kalman", "trajectory", "state_space", "state-space", "dynamic_poll")
        polling_config = dict(model_config.get("polling", {}))
        config_matches = cls._matching_paths(polling_config, keywords)
        config_enabled = cls._truthy_matching_paths(polling_config, keywords)
        payload_matches = cls._matching_paths(backtest_payload, keywords, prefix="backtest")
        if poll_trajectory is not None and not poll_trajectory.is_empty():
            bayesian = dict(model_config.get("bayesian", {}))
            active_engine = str(model_config.get("_inference_engine") or "").lower()
            if (
                active_engine == "bayes"
                and bool(bayesian.get("enabled"))
                and str(bayesian.get("backend")) == "nuts"
            ):
                return {
                    "tier": "production",
                    "current": (
                        "Run artifacts include Bayesian/NUTS state-space trajectory rows; "
                        f"rows={poll_trajectory.height}."
                    ),
                }
            return {
                "tier": "functional",
                "current": (
                    "Run artifacts include poll_trajectory rows with Kalman/trajectory "
                    f"columns; rows={poll_trajectory.height}."
                ),
            }
        artifact_columns = [
            column
            for column in race_forecasts.columns
            if any(keyword in column.lower() for keyword in keywords)
        ]
        if artifact_columns:
            return {
                "tier": "functional",
                "current": (
                    "Trajectory/Kalman forecast artifact columns are visible: "
                    f"{', '.join(sorted(artifact_columns))}."
                ),
            }
        if config_enabled:
            return {
                "tier": "functional",
                "current": (
                    "Polling trajectory/Kalman support is enabled in config at "
                    f"{', '.join(config_enabled)}; no separate forecast artifact column is visible."
                ),
            }
        if config_matches or payload_matches:
            visible = ", ".join([*config_matches, *payload_matches])
            return {
                "tier": "scaffold",
                "current": (
                    "Trajectory/Kalman hooks are visible but not active as auditable forecast "
                    f"artifacts: {visible}."
                ),
            }
        return {
            "tier": "absent",
            "current": (
                "No Kalman, state-space, or polling-trajectory support is visible in config "
                "or run artifacts; polling remains a deterministic weighted average."
            ),
        }

    @classmethod
    def _matching_paths(
        cls, payload: Any, keywords: tuple[str, ...], prefix: str = "polling"
    ) -> list[str]:
        paths: list[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                key_lower = str(key).lower()
                if any(keyword in key_lower for keyword in keywords):
                    paths.append(path)
                paths.extend(cls._matching_paths(value, keywords, path))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                paths.extend(cls._matching_paths(value, keywords, f"{prefix}[{index}]"))
        return paths

    @classmethod
    def _truthy_matching_paths(
        cls, payload: Any, keywords: tuple[str, ...], prefix: str = "polling"
    ) -> list[str]:
        paths: list[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                key_lower = str(key).lower()
                if any(keyword in key_lower for keyword in keywords) and cls._truthy(value):
                    paths.append(path)
                paths.extend(cls._truthy_matching_paths(value, keywords, path))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                paths.extend(cls._truthy_matching_paths(value, keywords, f"{prefix}[{index}]"))
        return paths

    @classmethod
    def _truthy(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "none", "disabled"}
        if isinstance(value, dict):
            return any(cls._truthy(child) for child in value.values())
        if isinstance(value, list):
            return any(cls._truthy(child) for child in value)
        return value is not None

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
