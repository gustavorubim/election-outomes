from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl

from civic_signal.features.builder import FeatureBundle


def subset_bundle(bundle: FeatureBundle, race_catalog: pl.DataFrame) -> FeatureBundle:
    active_ids = race_catalog["race_id"].to_list() if "race_id" in race_catalog.columns else []

    def by_race(frame: pl.DataFrame) -> pl.DataFrame:
        if "race_id" not in frame.columns:
            return frame
        return frame.filter(pl.col("race_id").is_in(active_ids))

    return replace(
        bundle,
        races=by_race(bundle.races),
        options=by_race(bundle.options),
        polls=by_race(bundle.polls),
        markets=by_race(bundle.markets),
        public_signals=by_race(bundle.public_signals),
        fundamentals=by_race(bundle.fundamentals),
        results=by_race(bundle.results),
        backtest_predictions=by_race(bundle.backtest_predictions),
        race_catalog=race_catalog,
    )


def filter_bundle_by_date(bundle: FeatureBundle, as_of: str) -> FeatureBundle:
    cutoff = date.fromisoformat(as_of)

    def by_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
        if column not in frame.columns:
            return frame
        dates = pl.col(column)
        if frame.schema[column] != pl.Date:
            dates = (
                pl.col(column).cast(pl.Utf8).str.slice(0, 10).str.strptime(pl.Date, strict=False)
            )
        return frame.filter(dates <= cutoff)

    return replace(
        bundle,
        polls=by_date(bundle.polls, "end_date"),
        markets=by_date(bundle.markets, "observed_at"),
        public_signals=by_date(bundle.public_signals, "observed_at"),
        fundamentals=by_date(bundle.fundamentals, "as_of"),
    )


def filter_results_before_cycle(bundle: FeatureBundle, target_cycle: int) -> FeatureBundle:
    historical_ids = (
        bundle.race_catalog.filter(pl.col("cycle") < target_cycle)["race_id"].to_list()
        if "cycle" in bundle.race_catalog.columns
        else []
    )

    def historical(frame: pl.DataFrame) -> pl.DataFrame:
        if "race_id" not in frame.columns:
            return frame
        return frame.filter(pl.col("race_id").is_in(historical_ids))

    return replace(
        bundle,
        races=historical(bundle.races),
        options=historical(bundle.options),
        polls=historical(bundle.polls),
        markets=historical(bundle.markets),
        public_signals=historical(bundle.public_signals),
        fundamentals=historical(bundle.fundamentals),
        results=historical(bundle.results),
        backtest_predictions=historical(bundle.backtest_predictions),
        race_catalog=bundle.race_catalog.filter(pl.col("race_id").is_in(historical_ids)),
    )
