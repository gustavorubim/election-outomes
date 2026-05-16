from __future__ import annotations

import polars as pl


class TierAssessor:
    def __init__(self, tier_config: dict[str, object]) -> None:
        self.config = tier_config

    def assign(
        self,
        races: pl.DataFrame,
        polls: pl.DataFrame,
        markets: pl.DataFrame,
        fundamentals: pl.DataFrame,
        public_signals: pl.DataFrame,
    ) -> pl.DataFrame:
        counts = self._counts(races, polls, markets, fundamentals, public_signals)
        tier_a = dict(self.config.get("tier_a", {}))
        tier_b = dict(self.config.get("tier_b", {}))
        min_polls = int(tier_a.get("min_polls", 2))
        min_pollsters = int(tier_a.get("min_pollsters", 2))
        min_market_quotes = int(tier_a.get("min_market_quotes", 1))
        min_fundamental_rows = int(tier_a.get("min_fundamental_rows", 1))
        min_any_signal_rows = int(tier_b.get("min_any_signal_rows", 1))
        tier_c_reason = str(
            self.config.get("tier_c", {}).get("reason", "Insufficient validated data.")
        )
        has_a_polling = (pl.col("poll_count") >= min_polls) & (
            pl.col("pollster_count") >= min_pollsters
        )
        has_a_market = pl.col("market_count") >= min_market_quotes
        has_fundamentals = pl.col("fundamental_count") >= min_fundamental_rows
        any_signal = (
            pl.col("poll_count")
            + pl.col("market_count")
            + pl.col("fundamental_count")
            + pl.col("public_signal_count")
        )
        tier_a_condition = has_fundamentals & (has_a_polling | has_a_market)
        tier_b_condition = (any_signal >= min_any_signal_rows) & has_fundamentals
        return (
            races.join(counts, on="race_id", how="left")
            .with_columns(
                [
                    pl.col("poll_count").fill_null(0),
                    pl.col("pollster_count").fill_null(0),
                    pl.col("market_count").fill_null(0),
                    pl.col("fundamental_count").fill_null(0),
                    pl.col("public_signal_count").fill_null(0),
                ]
            )
            .with_columns(
                pl.when(tier_a_condition)
                .then(pl.lit("A"))
                .when(tier_b_condition)
                .then(pl.lit("B"))
                .otherwise(pl.lit("C"))
                .alias("tier"),
                pl.when(tier_a_condition)
                .then(pl.lit("Validated polls/markets plus fundamentals."))
                .when(tier_b_condition)
                .then(pl.lit("Sparse forecast with fundamentals and wide uncertainty."))
                .otherwise(pl.lit(tier_c_reason))
                .alias("tier_reason"),
            )
        )

    def _counts(
        self,
        races: pl.DataFrame,
        polls: pl.DataFrame,
        markets: pl.DataFrame,
        fundamentals: pl.DataFrame,
        public_signals: pl.DataFrame,
    ) -> pl.DataFrame:
        frames = [races.select("race_id").unique()]
        frames.append(
            polls.group_by("race_id").agg(
                pl.len().alias("poll_count"),
                pl.col("pollster").n_unique().alias("pollster_count"),
            )
        )
        frames.append(markets.group_by("race_id").agg(pl.len().alias("market_count")))
        frames.append(fundamentals.group_by("race_id").agg(pl.len().alias("fundamental_count")))
        frames.append(public_signals.group_by("race_id").agg(pl.len().alias("public_signal_count")))
        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, on="race_id", how="left")
        return result
