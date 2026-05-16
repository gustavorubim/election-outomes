from __future__ import annotations

import polars as pl

from civic_signal.reports.diagnostics import DiagnosticsReport


def _race_catalog(office_type: str = "senate") -> pl.DataFrame:
    seats = [16, 10] if office_type == "president" else [1, 1]
    return pl.DataFrame(
        {
            "race_id": [f"US-{office_type.upper()}-GA-2026", f"US-{office_type.upper()}-MN-2026"],
            "office_type": [office_type, office_type],
            "seats": seats,
            "tier": ["A", "A"],
        }
    )


def _race_forecasts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_id": [
                "US-SEN-GA-2026",
                "US-SEN-GA-2026",
                "US-SEN-MN-2026",
                "US-SEN-MN-2026",
            ],
            "option_id": ["ga-dem", "ga-rep", "mn-dem", "mn-rep"],
            "name": [
                "Georgia Democrat",
                "Georgia Republican",
                "Minnesota Democrat",
                "Minnesota Republican",
            ],
            "party": ["DEM", "REP", "DEM", "REP"],
            "winner_probability": [0.48, 0.52, 0.61, 0.39],
            "vote_share_mean": [0.49, 0.51, 0.55, 0.45],
            "vote_share_p05": [0.44, 0.46, 0.50, 0.40],
            "vote_share_p95": [0.54, 0.56, 0.60, 0.50],
            "top_drivers": ["polling blend"] * 4,
            "component_contributions": [
                '{"polling":{"vote_share":0.49,"weight":0.70}}',
                '{"polling":{"vote_share":0.51,"weight":0.70}}',
                '{"fundamentals":{"vote_share":0.55,"weight":0.30}}',
                '{"fundamentals":{"vote_share":0.45,"weight":0.30}}',
            ],
        }
    )


def _source_manifest() -> pl.DataFrame:
    return pl.DataFrame({"source_id": ["fixture-polls", "fixture-results"]})


def _backtest_payload(sample_size_too_small: bool = False) -> dict[str, object]:
    return {
        "row_count": 42,
        "sample_size_too_small": sample_size_too_small,
        "metrics": {
            "ensemble": {
                "brier": 0.081,
                "log_score": 0.22,
                "expected_calibration_error": 0.031,
            },
            "ignored-note": "not a metric row",
        },
        "rolling_origin": {"cycles": [2020, 2024]},
    }


def _control_forecasts(body: str = "senate") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "control_body": [body, body],
            "party": ["REP", "DEM"],
            "control_probability": [0.62, 0.38],
            "seat_count_mean": [52.4, 47.6],
            "seat_count_p10": [50.0, 45.0],
            "seat_count_p90": [55.0, 50.0],
            "control_threshold": [51, 51],
            "modeled_seats": [35, 35],
            "holdover_seats": [65, 65],
        }
    )


def _plot_manifest() -> dict[str, list[dict[str, str]]]:
    return {
        "distribution": [
            {"path": "plots/seat_count_histogram.png", "title": "Seat Count Histogram"},
            {"path": "plots/vote_share_density.png", "title": "Vote Share Density"},
            {
                "path": "plots/electoral_college_distribution.png",
                "title": "Electoral College Distribution",
            },
        ],
        "projection": [
            {"path": "plots/control_projection.png", "title": "Control Projection"},
            {"path": "plots/topline_electoral_swarm.png", "title": "Electoral Swarm"},
            {"path": "plots/race_probability_bars.png", "title": "Race Probability Bars"},
        ],
        "drivers": [{"path": "plots/tipping_points.png", "title": "Tipping Points"}],
        "model_quality": [{"path": "plots/calibration_curve.png", "title": "Calibration Curve"}],
    }


def test_control_dashboard_prioritizes_chamber_overview_plots() -> None:
    html = DiagnosticsReport().render(
        run_id="senate-unit",
        race_catalog=_race_catalog(),
        race_forecasts=_race_forecasts(),
        source_manifest=_source_manifest(),
        backtest_payload=_backtest_payload(),
        reward_card={"rewards": {"R0_build": {"passed": None}, "R2_provenance": {"passed": True}}},
        plot_manifest=_plot_manifest(),
        methodology_benchmark={
            "summary_score": 0.61,
            "status": "functional",
            "rows": [{"dimension": "Calibration", "tier": "functional", "score": 0.66}],
        },
        control_forecasts=_control_forecasts(),
        ecosystem_forecasts=pl.DataFrame(
            {"race_id": ["US-SEN-GA-2026"], "recount_probability": [0.12]}
        ),
    )

    assert "Republicans favored for Senate control" in html
    assert "Control probability" in html
    assert "overview-plot-grid" in html
    assert html.index("plots/seat_count_histogram.png") < html.index("Where The Forecast Lives")
    assert html.index("plots/control_projection.png") < html.index("Projection Views")
    assert "plots/vote_share_density.png" in html
    assert "Scenario Scope" in html
    assert "Closest Contests" in html
    assert "forecast-shell" in html
    assert "Top-line plot was not generated" not in html


def test_presidential_dashboard_prioritizes_electoral_college_overview_plots() -> None:
    control = _control_forecasts("president").with_columns(
        pl.Series("control_threshold", [270, 270]),
        pl.Series("seat_count_mean", [291.4, 246.6]),
        pl.Series("seat_count_p10", [257.0, 211.0]),
        pl.Series("seat_count_p90", [331.0, 281.0]),
        pl.Series("party", ["DEM", "REP"]),
    )
    html = DiagnosticsReport().render(
        run_id="pres-unit",
        race_catalog=_race_catalog("president"),
        race_forecasts=_race_forecasts(),
        source_manifest=_source_manifest(),
        backtest_payload=_backtest_payload(),
        plot_manifest=_plot_manifest(),
        control_forecasts=control,
    )

    assert "Democrats favored in the Electoral College" in html
    assert "DEM EC win" in html
    assert html.index("plots/electoral_college_distribution.png") < html.index(
        "Where The Forecast Lives"
    )
    assert html.index("plots/topline_electoral_swarm.png") < html.index("Projection Views")
    assert "plots/seat_count_histogram.png" in html


def test_diagnostics_helpers_cover_empty_and_legacy_paths() -> None:
    report = DiagnosticsReport()
    empty_forecast = pl.DataFrame(schema={"race_id": pl.Utf8, "winner_probability": pl.Float64})
    invalid_driver_row = pl.DataFrame(
        {
            "race_id": ["US-SEN-GA-2026"],
            "option_id": ["ga-dem"],
            "name": ["Georgia Democrat"],
            "top_drivers": ["polling blend"],
            "winner_probability": [0.51],
            "component_contributions": ["{bad json"],
        }
    )
    driver_row_without_probability = pl.DataFrame(
        {
            "race_id": ["US-SEN-GA-2026"],
            "option_id": ["ga-dem"],
            "name": ["Georgia Democrat"],
            "top_drivers": ["fundamentals"],
            "component_contributions": ['{"fundamentals":{"vote_share":0.53,"weight":0.40}}'],
        }
    )
    no_probability_row = pl.DataFrame(
        {"race_id": ["US-SEN-GA-2026"], "winner_probability": [None]},
        schema={"race_id": pl.Utf8, "winner_probability": pl.Float64},
    )
    insight_strip = report._insight_strip(
        race_catalog=_race_catalog("president"),
        control_forecasts=_control_forecasts("president"),
        ecosystem_forecasts=pl.DataFrame(
            {"race_id": ["US-SEN-GA-2026"], "recount_probability": [0.2]}
        ),
        backtest_payload=_backtest_payload(True),
    )

    assert report._topline(empty_forecast)["headline"] == "No trusted probability available"
    assert report._topline(_race_forecasts(), None)["margin"] == "+10.0 pts"
    assert "42.0%" in report._metric_card("Rate", 0.42, "fixture")
    assert "26 modeled EV" in insight_strip
    assert "Control Readout" in insight_strip
    assert "Closest-Race Risk" in insight_strip
    assert "below threshold" in insight_strip
    assert "No chamber/control forecast generated" in report._control_table(None)
    assert "No race forecast rows available" in report._closest_race_list(pl.DataFrame())
    assert "No competitive races available" in report._closest_race_list(no_probability_row)
    assert "No forecast rows were generated" in report._forecast_table(empty_forecast)
    assert "No driver rows available" in report._driver_cards(pl.DataFrame())
    assert "No admitted component contribution" in report._driver_cards(invalid_driver_row)
    assert "fundamentals" in report._driver_cards(driver_row_without_probability)
    assert "No reward card generated" in report._reward_grid({})
    assert "configured trust threshold" in report._backtest_summary(_backtest_payload(True))
    assert "No methodology benchmark generated" in report._benchmark_summary({})
    assert "No plots generated" in report._plot_sections(
        {"drivers": _plot_manifest()["drivers"]}, exclude_categories=["drivers"]
    )
    assert "summary-plot-grid" in report._single_plot_cards(
        _plot_manifest(), ["control_projection.png", "missing.png"]
    )
    assert "Top-line plot was not generated: missing.png" in report._single_plot_card(
        _plot_manifest(), "missing.png"
    )
    assert "&quot;control_rows&quot;: 0" in report._audit_summary(
        race_catalog=pl.DataFrame(),
        source_manifest=_source_manifest(),
        control_forecasts=None,
        ecosystem_forecasts=None,
    )
    assert report._pct("not-a-rate") == "not-a-rate"
