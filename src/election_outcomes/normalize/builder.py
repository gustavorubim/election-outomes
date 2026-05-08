from __future__ import annotations

import json
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
    UNIQUE_KEYS_BY_TABLE: ClassVar[dict[str, list[str]]] = {
        "races": ["race_id"],
        "options": ["race_id", "option_id"],
        "polls": ["poll_id", "race_id", "option_id"],
        "market_quotes": ["market_id", "race_id", "option_id", "observed_at"],
        "public_signals": ["signal_id", "race_id", "option_id", "observed_at"],
        "fundamentals": ["race_id", "as_of"],
        "results": ["race_id", "option_id"],
        "backtest_predictions": ["race_id", "option_id", "cycle"],
    }
    PRESIDENT_STATE_PANEL_TABLES: ClassVar[dict[str, str]] = {
        "president-state-panel-races-v1": "races",
        "president-state-panel-options-v1": "options",
        "president-state-panel-results-v1": "results",
        "president-state-panel-fundamentals-v1": "fundamentals",
        "president-state-panel-polls-v1": "polls",
    }
    PRESIDENT_STATE_PANEL_BASE_COLUMNS: ClassVar[set[str]] = {
        "cycle",
        "state",
        "election_date",
    }
    PRESIDENT_STATE_PANEL_OPTION_COLUMNS: ClassVar[set[str]] = {
        "dem_name",
        "rep_name",
        "dem_incumbent",
        "rep_incumbent",
        "dem_previous_vote_share",
        "rep_previous_vote_share",
        "dem_fundraising_usd",
        "rep_fundraising_usd",
    }
    PRESIDENT_STATE_PANEL_RESULT_COLUMNS: ClassVar[set[str]] = {
        "dem_vote_share",
        "rep_vote_share",
        "turnout",
    }
    PRESIDENT_STATE_PANEL_FUNDAMENTAL_COLUMNS: ClassVar[set[str]] = {
        "partisan_lean",
        "incumbency_advantage",
        "economic_index",
        "demographic_turnout_index",
        "historical_turnout_rate",
        "registered_voters",
    }
    PRESIDENT_STATE_PANEL_POLL_COLUMNS: ClassVar[set[str]] = {
        "pollster",
        "poll_sample_size",
        "poll_population",
        "poll_sponsor_class",
        "poll_methodology",
        "dem_poll_pct",
        "rep_poll_pct",
    }

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
                self._dedupe(
                    pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0],
                    table,
                )
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
        table = str(row["table"])
        parser_version = str(row["parser_version"])
        parser_args = self._parser_args_from_row(row)
        if parser_version == "fivethirtyeight-president-polls-v1":
            frame = self._normalize_538_president_polls(frame, parser_args)
        elif parser_version in self.PRESIDENT_STATE_PANEL_TABLES:
            expected_table = self.PRESIDENT_STATE_PANEL_TABLES[parser_version]
            if table != expected_table:
                raise ValueError(
                    f"{parser_version} must be registered for table {expected_table!r}; "
                    f"got {table!r}"
                )
            frame = self._normalize_president_state_panel(frame, parser_version, parser_args)
        frame = self._coerce(frame)
        return frame.with_columns(
            pl.lit(row["source_id"]).alias("source_id"),
            pl.lit(row["content_hash"]).alias("source_hash"),
            pl.lit(row["parser_version"]).alias("parser_version"),
        )

    @staticmethod
    def _parser_args_from_row(row: dict[str, object]) -> dict[str, object]:
        raw = row.get("parser_args")
        if not raw:
            return {}
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            source_id = row.get("source_id", "unknown")
            raise ValueError(f"Malformed parser_args JSON for source {source_id}") from exc
        if not isinstance(payload, dict):
            source_id = row.get("source_id", "unknown")
            raise ValueError(f"parser_args for source {source_id} must decode to a mapping")
        return payload

    @classmethod
    def _dedupe(cls, frame: pl.DataFrame, table: str) -> pl.DataFrame:
        keys = cls.UNIQUE_KEYS_BY_TABLE.get(table)
        if not keys:
            return frame
        present = [key for key in keys if key in frame.columns]
        if not present:
            return frame
        return frame.unique(subset=present, keep="first", maintain_order=True)

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

    def _normalize_president_state_panel(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        if parser_version == "president-state-panel-races-v1":
            return self._president_state_panel_races(frame, parser_version, parser_args)
        if parser_version == "president-state-panel-options-v1":
            return self._president_state_panel_options(frame, parser_version, parser_args)
        if parser_version == "president-state-panel-results-v1":
            return self._president_state_panel_results(frame, parser_version, parser_args)
        if parser_version == "president-state-panel-fundamentals-v1":
            return self._president_state_panel_fundamentals(frame, parser_version, parser_args)
        if parser_version == "president-state-panel-polls-v1":
            return self._president_state_panel_polls(frame, parser_version, parser_args)
        raise ValueError(f"Unsupported president state panel parser: {parser_version}")

    def _president_state_panel_races(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        base = self._president_state_panel_base(
            frame, {"electoral_votes"}, parser_version, parser_args
        )
        return base.select(
            "race_id",
            "cycle",
            "state",
            "election_date",
            pl.lit("state").alias("geography_type"),
            pl.col("state").alias("geography"),
            pl.lit("president").alias("office_type"),
            pl.lit("candidate").alias("race_type"),
            pl.lit("general").alias("stage"),
            pl.col("electoral_votes").cast(pl.Int64, strict=False).alias("seats"),
            pl.lit("president").alias("control_body"),
            pl.lit(None, dtype=pl.Float64).alias("measure_threshold"),
        )

    def _president_state_panel_options(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        base = self._president_state_panel_base(
            frame, self.PRESIDENT_STATE_PANEL_OPTION_COLUMNS, parser_version, parser_args
        )
        dem_ids, rep_ids = self._president_state_option_id_columns(base)
        return self._president_state_party_rows(
            base,
            [
                "cycle",
                "state",
                "race_id",
                "option_id",
                "name",
                "party",
                "incumbent",
                "previous_vote_share",
                "fundraising_usd",
            ],
            dem_columns={
                **dem_ids,
                "name": "dem_name",
                "incumbent": "dem_incumbent",
                "previous_vote_share": "dem_previous_vote_share",
                "fundraising_usd": "dem_fundraising_usd",
            },
            rep_columns={
                **rep_ids,
                "name": "rep_name",
                "incumbent": "rep_incumbent",
                "previous_vote_share": "rep_previous_vote_share",
                "fundraising_usd": "rep_fundraising_usd",
            },
        )

    def _president_state_panel_results(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        base = self._president_state_panel_base(
            frame, self.PRESIDENT_STATE_PANEL_RESULT_COLUMNS, parser_version, parser_args
        )
        dem_ids, rep_ids = self._president_state_option_id_columns(base)
        results = self._president_state_party_rows(
            base,
            [
                "cycle",
                "state",
                "race_id",
                "option_id",
                "vote_share",
                "turnout",
                "winner",
            ],
            dem_columns={**dem_ids, "vote_share": "dem_vote_share", "turnout": "turnout"},
            rep_columns={**rep_ids, "vote_share": "rep_vote_share", "turnout": "turnout"},
        ).with_columns(pl.col("vote_share").cast(pl.Float64, strict=False))
        results = results.filter(pl.col("vote_share").is_not_null())
        if results.is_empty():
            return results
        return results.with_columns(
            (pl.col("vote_share") == pl.col("vote_share").max().over("race_id")).alias("winner")
        )

    def _president_state_panel_fundamentals(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        offsets = self._panel_as_of_offsets(parser_args, parser_version)
        required = set(self.PRESIDENT_STATE_PANEL_FUNDAMENTAL_COLUMNS)
        if not offsets:
            required.add("as_of")
        base = self._president_state_panel_base(frame, required, parser_version, parser_args)
        if offsets:
            base = self._expand_panel_as_of_offsets(base, offsets)
        else:
            base = base.with_columns(
                self._date_expr("as_of"),
                pl.lit(None, dtype=pl.Int64).alias("as_of_offset_days"),
            )
        self._require_non_nulls(base, ["as_of"], parser_version)
        return base.select(
            "cycle",
            "state",
            "race_id",
            "as_of",
            "as_of_offset_days",
            "partisan_lean",
            "incumbency_advantage",
            "economic_index",
            "demographic_turnout_index",
            "historical_turnout_rate",
            "registered_voters",
        )

    def _president_state_panel_polls(
        self, frame: pl.DataFrame, parser_version: str, parser_args: dict[str, object]
    ) -> pl.DataFrame:
        offsets = self._panel_as_of_offsets(parser_args, parser_version)
        required = set(self.PRESIDENT_STATE_PANEL_POLL_COLUMNS)
        if not offsets:
            required.update({"poll_start_date", "poll_end_date"})
        base = self._president_state_panel_base(frame, required, parser_version, parser_args)
        if offsets:
            duration_days = self._panel_poll_duration_days(parser_args, parser_version)
            base = self._expand_panel_as_of_offsets(base, offsets).with_columns(
                pl.col("as_of").alias("end_date"),
                (pl.col("as_of") - pl.duration(days=duration_days)).alias("start_date"),
            )
        else:
            base = base.with_columns(
                self._date_expr("poll_start_date"),
                self._date_expr("poll_end_date"),
                pl.lit(None, dtype=pl.Int64).alias("as_of_offset_days"),
            ).with_columns(
                pl.col("poll_start_date").alias("start_date"),
                pl.col("poll_end_date").alias("end_date"),
            )
        self._require_non_nulls(base, ["start_date", "end_date"], parser_version)
        base = base.with_columns(
            pl.when(pl.col("as_of_offset_days").is_not_null())
            .then(pl.concat_str([pl.lit("t"), pl.col("as_of_offset_days").cast(pl.Utf8)]))
            .otherwise(pl.col("end_date").cast(pl.Utf8))
            .alias("_poll_key")
        )
        dem_ids, rep_ids = self._president_state_option_id_columns(base)
        polls = self._president_state_party_rows(
            base,
            [
                "poll_id",
                "cycle",
                "state",
                "race_id",
                "pollster",
                "start_date",
                "end_date",
                "population",
                "sample_size",
                "sponsor_class",
                "methodology",
                "option_id",
                "pct",
                "as_of_offset_days",
            ],
            dem_columns={**dem_ids, "pct": "dem_poll_pct"},
            rep_columns={**rep_ids, "pct": "rep_poll_pct"},
            common_columns={
                "pollster": "pollster",
                "start_date": "start_date",
                "end_date": "end_date",
                "population": "poll_population",
                "sample_size": "poll_sample_size",
                "sponsor_class": "poll_sponsor_class",
                "methodology": "poll_methodology",
                "as_of_offset_days": "as_of_offset_days",
            },
            poll_key_column="_poll_key",
        ).with_columns(
            pl.col("pct").cast(pl.Float64, strict=False),
            pl.col("population").cast(pl.Utf8).str.to_lowercase(),
            pl.col("sponsor_class").cast(pl.Utf8).str.to_lowercase(),
            pl.col("methodology").cast(pl.Utf8).str.to_lowercase(),
        )
        return polls.filter(pl.col("pct").is_not_null())

    def _president_state_panel_base(
        self,
        frame: pl.DataFrame,
        required_columns: set[str],
        parser_version: str,
        parser_args: dict[str, object],
    ) -> pl.DataFrame:
        required = self.PRESIDENT_STATE_PANEL_BASE_COLUMNS | required_columns
        self._require_columns(frame, required, parser_version)
        generated_race_id = pl.concat_str(
            [
                pl.lit("US-PRES-"),
                pl.col("state"),
                pl.lit("-"),
                pl.col("cycle").cast(pl.Utf8),
            ]
        )
        race_id = (
            pl.coalesce(pl.col("race_id").cast(pl.Utf8), generated_race_id)
            if "race_id" in frame.columns
            else generated_race_id
        )
        base = frame.with_columns(
            pl.col("cycle").cast(pl.Int64, strict=False),
            pl.col("state").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("state"),
            self._date_expr("election_date"),
        ).with_columns(race_id.alias("race_id"))
        self._require_non_nulls(
            base, ["cycle", "state", "election_date", "race_id"], parser_version
        )
        cycles = self._panel_cycles(parser_args, parser_version)
        if cycles:
            base = base.filter(pl.col("cycle").is_in(cycles))
        return base

    @staticmethod
    def _president_state_option_id_columns(
        frame: pl.DataFrame,
    ) -> tuple[dict[str, str], dict[str, str]]:
        dem = {"option_id": "dem_option_id"} if "dem_option_id" in frame.columns else {}
        rep = {"option_id": "rep_option_id"} if "rep_option_id" in frame.columns else {}
        return dem, rep

    def _president_state_party_rows(
        self,
        frame: pl.DataFrame,
        output_columns: list[str],
        *,
        dem_columns: dict[str, str],
        rep_columns: dict[str, str],
        common_columns: dict[str, str] | None = None,
        poll_key_column: str | None = None,
    ) -> pl.DataFrame:
        common = common_columns or {}
        return pl.concat(
            [
                self._president_state_party_select(
                    frame,
                    output_columns,
                    party="DEM",
                    suffix="D",
                    column_map={**common, **dem_columns},
                    poll_key_column=poll_key_column,
                ),
                self._president_state_party_select(
                    frame,
                    output_columns,
                    party="REP",
                    suffix="R",
                    column_map={**common, **rep_columns},
                    poll_key_column=poll_key_column,
                ),
            ],
            how="diagonal_relaxed",
        )

    @staticmethod
    def _president_state_party_select(
        frame: pl.DataFrame,
        output_columns: list[str],
        *,
        party: str,
        suffix: str,
        column_map: dict[str, str],
        poll_key_column: str | None,
    ) -> pl.DataFrame:
        expressions: list[pl.Expr | str] = []
        for column in output_columns:
            if column == "party":
                expressions.append(pl.lit(party).alias(column))
            elif column == "option_id":
                source_column = column_map.get("option_id")
                if source_column is None:
                    expressions.append(
                        pl.concat_str([pl.col("race_id"), pl.lit(f"-{suffix}")]).alias(column)
                    )
                else:
                    expressions.append(pl.col(source_column).alias(column))
            elif column == "poll_id":
                if poll_key_column is None:
                    raise ValueError("poll_key_column is required when selecting poll_id")
                expressions.append(
                    pl.concat_str(
                        [
                            pl.lit("panel-"),
                            pl.col("race_id"),
                            pl.lit("-"),
                            pl.col(poll_key_column),
                            pl.lit(f"-{suffix}"),
                        ]
                    ).alias(column)
                )
            elif column == "winner":
                expressions.append(pl.lit(False).alias(column))
            elif column in column_map:
                expressions.append(pl.col(column_map[column]).alias(column))
            else:
                expressions.append(column)
        return frame.select(expressions)

    @staticmethod
    def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    @staticmethod
    def _require_non_nulls(frame: pl.DataFrame, columns: list[str], label: str) -> None:
        counts = frame.select(
            [pl.col(column).is_null().sum().alias(column) for column in columns]
        ).row(0, named=True)
        missing_values = sorted(column for column, count in counts.items() if count)
        if missing_values:
            raise ValueError(f"{label} has null values in required columns: {missing_values}")

    @staticmethod
    def _panel_as_of_offsets(parser_args: dict[str, object], parser_version: str) -> list[int]:
        raw_offsets = parser_args.get("as_of_offsets_days")
        if raw_offsets is None:
            return []
        if not isinstance(raw_offsets, list) or not raw_offsets:
            raise ValueError(
                f"{parser_version} parser_args.as_of_offsets_days must be a non-empty list"
            )
        offsets: list[int] = []
        for value in raw_offsets:
            try:
                offset = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parser_version} parser_args.as_of_offsets_days must contain integers"
                ) from exc
            if offset < 0:
                raise ValueError(
                    f"{parser_version} parser_args.as_of_offsets_days must be non-negative"
                )
            if offset not in offsets:
                offsets.append(offset)
        return offsets

    @staticmethod
    def _panel_cycles(parser_args: dict[str, object], parser_version: str) -> list[int]:
        raw_cycles = parser_args.get("cycles")
        if raw_cycles is None:
            return []
        if not isinstance(raw_cycles, list) or not raw_cycles:
            raise ValueError(f"{parser_version} parser_args.cycles must be a non-empty list")
        cycles: list[int] = []
        for value in raw_cycles:
            try:
                cycle = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parser_version} parser_args.cycles must contain integers"
                ) from exc
            if cycle not in cycles:
                cycles.append(cycle)
        return cycles

    @staticmethod
    def _panel_poll_duration_days(parser_args: dict[str, object], parser_version: str) -> int:
        raw = parser_args.get("poll_duration_days", 3)
        try:
            duration_days = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{parser_version} parser_args.poll_duration_days must be an integer"
            ) from exc
        if duration_days < 0:
            raise ValueError(
                f"{parser_version} parser_args.poll_duration_days must be non-negative"
            )
        return duration_days

    @staticmethod
    def _expand_panel_as_of_offsets(frame: pl.DataFrame, offsets: list[int]) -> pl.DataFrame:
        return frame.join(pl.DataFrame({"as_of_offset_days": offsets}), how="cross").with_columns(
            (pl.col("election_date") - pl.duration(days=pl.col("as_of_offset_days"))).alias("as_of")
        )

    def _normalize_538_president_polls(
        self, frame: pl.DataFrame, parser_args: dict[str, object] | None = None
    ) -> pl.DataFrame:
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
        args = parser_args or {}
        required_args = {"cycle", "state", "stage", "race_id", "parties"}
        missing_args = required_args.difference(args)
        if missing_args:
            raise ValueError(
                "FiveThirtyEight president parser_args missing required keys: "
                f"{sorted(missing_args)}"
            )
        cycle = int(args["cycle"])
        state_lower = str(args["state"]).lower()
        stage_lower = str(args["stage"]).lower()
        race_id = str(args["race_id"])
        parties = [str(party).upper() for party in list(args["parties"])]
        if not parties:
            raise ValueError("FiveThirtyEight president parser_args parties must be non-empty")
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
                (pl.col("cycle") == cycle)
                & (pl.col("_state_lower") == state_lower)
                & (pl.col("_stage_lower") == stage_lower)
                & pl.col("candidate_party").is_in(parties)
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
                pl.lit(race_id).alias("race_id"),
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
                    [pl.lit(f"{race_id}-"), pl.col("candidate_party").str.slice(0, 1)]
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
