from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.storage.io import write_parquet


@dataclass(frozen=True)
class CuratedBuildResult:
    tables: dict[str, pl.DataFrame]


class CuratedDataBuilder:
    """Normalize raw snapshots into typed curated Parquet tables."""

    NUMERIC_COLUMNS: ClassVar[set[str]] = {
        "cycle",
        "seats",
        "measure_threshold",
        "incumbent",
        "previous_vote_share",
        "fundraising_usd",
        "sample_size",
        "pct",
        "probability",
        "spread",
        "volume",
        "open_interest",
        "value",
        "z_score",
        "partisan_lean",
        "incumbency_advantage",
        "economic_index",
        "demographic_turnout_index",
        "historical_turnout_rate",
        "registered_voters",
        "vote_share",
        "turnout",
        "winner",
        "actual_winner",
        "actual_vote_share",
        "baseline_probability",
        "polls_probability",
        "fundamentals_probability",
        "markets_probability",
        "public_signals_probability",
        "ensemble_probability",
        "predicted_vote_share",
        "lower_90",
        "upper_90",
    }
    BOOL_COLUMNS: ClassVar[set[str]] = {"incumbent", "winner", "actual_winner", "leakage_checked"}
    INT_COLUMNS: ClassVar[set[str]] = {"cycle", "seats", "sample_size"}

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def run(self) -> CuratedBuildResult:
        manifest_path = self.context.raw_dir / "source_manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError("Run sync before build-features")
        manifest = pl.read_parquet(manifest_path).filter(pl.col("status") != "failed")
        table_frames: dict[str, list[pl.DataFrame]] = {}
        self.context.curated_dir.mkdir(parents=True, exist_ok=True)

        for row in manifest.iter_rows(named=True):
            table = str(row["table"])
            frame = self._read_source(row)
            table_frames.setdefault(table, []).append(frame)
        tables = {
            table: self._canonical_order(
                pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
            )
            for table, frames in table_frames.items()
        }
        for table, frame in tables.items():
            write_parquet(frame, self.context.curated_dir / f"{table}.parquet")
        usage_manifest = manifest.with_columns(
            pl.concat_str([pl.lit("curated:"), pl.col("table")]).alias("downstream_usage")
        )
        write_parquet(usage_manifest, self.context.curated_dir / "source_manifest.parquet")
        return CuratedBuildResult(tables)

    def _read_source(self, row: dict[str, object]) -> pl.DataFrame:
        frame = pl.read_csv(
            str(row["raw_path"]),
            infer_schema_length=1000,
            null_values=["", "null", "None"],
            try_parse_dates=True,
        )
        if frame.columns:
            frame = frame.filter(pl.col(frame.columns[0]).is_not_null())
        parser_version = str(row["parser_version"])
        if parser_version == "fivethirtyeight-president-polls-v1":
            frame = self._normalize_538_president_polls(frame)
        frame = self._coerce(frame)
        return frame.with_columns(
            pl.lit(row["source_id"]).alias("source_id"),
            pl.lit(row["content_hash"]).alias("source_hash"),
            pl.lit(row["parser_version"]).alias("parser_version"),
        )

    @staticmethod
    def _canonical_order(frame: pl.DataFrame) -> pl.DataFrame:
        priority = [
            "race_id",
            "option_id",
            "poll_id",
            "market_id",
            "signal_id",
            "as_of",
            "end_date",
            "observed_at",
        ]
        sort_columns = [column for column in priority if column in frame.columns]
        return frame.sort(sort_columns) if sort_columns else frame

    def _normalize_538_president_polls(self, frame: pl.DataFrame) -> pl.DataFrame:
        required = {
            "cycle",
            "state",
            "pollster",
            "poll_id",
            "question_id",
            "start_date",
            "end_date",
            "sample_size",
            "population",
            "methodology",
            "internal",
            "partisan",
            "stage",
            "answer",
            "candidate_party",
            "pct",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"FiveThirtyEight president polls missing columns: {sorted(missing)}")
        return (
            frame.with_columns(
                pl.col("cycle").cast(pl.Int64, strict=False),
                pl.col("candidate_party").str.to_uppercase().alias("candidate_party"),
                pl.col("state").str.to_lowercase().alias("_state_lower"),
                pl.col("stage").str.to_lowercase().alias("_stage_lower"),
                self._date_expr("start_date"),
                self._date_expr("end_date"),
            )
            .filter(
                (pl.col("cycle") == 2020)
                & (pl.col("_state_lower") == "wisconsin")
                & (pl.col("_stage_lower") == "general")
                & pl.col("candidate_party").is_in(["DEM", "REP"])
                & pl.col("pct").is_not_null()
            )
            .select(
                pl.concat_str(
                    [
                        pl.lit("538-"),
                        pl.col("poll_id").cast(pl.Utf8),
                        pl.lit("-"),
                        pl.col("question_id").cast(pl.Utf8),
                        pl.lit("-"),
                        pl.col("candidate_party"),
                    ]
                ).alias("poll_id"),
                pl.lit("US-PRES-WI-2020").alias("race_id"),
                pl.col("pollster").fill_null("unknown").alias("pollster"),
                "start_date",
                "end_date",
                pl.col("population")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .replace({"v": "lv"})
                .fill_null("a")
                .alias("population"),
                pl.col("sample_size").cast(pl.Float64, strict=False).alias("sample_size"),
                pl.when(pl.col("internal").cast(pl.Boolean, strict=False))
                .then(pl.lit("internal"))
                .when(pl.col("partisan").is_not_null())
                .then(pl.lit("partisan"))
                .otherwise(pl.lit("nonpartisan"))
                .alias("sponsor_class"),
                self._methodology_expr().alias("methodology"),
                pl.concat_str(
                    [pl.lit("US-PRES-WI-2020-"), pl.col("candidate_party").str.slice(0, 1)]
                ).alias("option_id"),
                pl.col("pct").cast(pl.Float64, strict=False).alias("pct"),
            )
            .unique(subset=["poll_id", "race_id", "option_id"], keep="last", maintain_order=True)
            .sort(["race_id", "option_id", "end_date", "poll_id"])
        )

    @staticmethod
    def _date_expr(column: str) -> pl.Expr:
        text = pl.col(column).cast(pl.Utf8)
        return pl.coalesce(
            text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            text.str.strptime(pl.Date, "%m/%d/%y", strict=False),
            text.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        ).alias(column)

    @staticmethod
    def _methodology_expr() -> pl.Expr:
        method = pl.col("methodology").cast(pl.Utf8).str.to_lowercase()
        return (
            pl.when(method.str.contains("phone|live", literal=False))
            .then(pl.lit("live_phone"))
            .when(method.str.contains("mixed|text|ivr|mail|online panel", literal=False))
            .then(pl.lit("mixed"))
            .when(method.str.contains("online|web|internet", literal=False))
            .then(pl.lit("online"))
            .otherwise(pl.lit("mixed"))
        )

    def _coerce(self, frame: pl.DataFrame) -> pl.DataFrame:
        expressions = []
        for column in frame.columns:
            if column in self.BOOL_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Boolean, strict=False))
            elif column in self.INT_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Int64, strict=False))
            elif column in self.NUMERIC_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Float64, strict=False))
        return frame.with_columns(expressions) if expressions else frame
