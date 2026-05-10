from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from election_outcomes.config import ProjectContext, Scenario, ScenarioRegistry
from election_outcomes.inference.failover import FailoverPolicy, exercise_timeout_failover
from election_outcomes.pipeline import ForecastPipeline
from election_outcomes.storage.io import write_json
from election_outcomes.verification.checklist import VisualQAChecklist


@dataclass(frozen=True)
class Phase8VerificationRunner:
    """Run the fixture-backed Phase 8 multi-office verification sequence."""

    context: ProjectContext

    def run(
        self,
        *,
        run_id: str | None = None,
        scenario: str = "2026-multioffice-verification",
        as_of: str | None = None,
        inference_engine: str = "bayes",
        bayesian_backend: str | None = None,
        quiet: bool = True,
        reproducibility_check: bool = True,
        daily_update: bool = True,
    ) -> dict[str, Any]:
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        if scenario_obj is None:
            raise ValueError("Phase 8 verification requires an explicit scenario")
        as_of = as_of or scenario_obj.default_as_of
        if as_of is None:
            raise ValueError("as_of is required unless the selected scenario defines default_as_of")
        run_id = run_id or datetime.now(UTC).strftime("phase8-%Y%m%dT%H%M%SZ")
        pipeline = ForecastPipeline(self.context)

        out_dir = pipeline.run_forecast(
            as_of=as_of,
            run_id=run_id,
            scenario=scenario,
            inference_engine=inference_engine,
            bayesian_backend=bayesian_backend,
            quiet=quiet,
        )
        if reproducibility_check:
            out_dir = pipeline.run_forecast(
                as_of=as_of,
                run_id=run_id,
                scenario=scenario,
                inference_engine=inference_engine,
                bayesian_backend=bayesian_backend,
                quiet=quiet,
            )

        daily_update_payload = None
        if daily_update and inference_engine.lower().strip() == "bayes":
            update_as_of = (date.fromisoformat(as_of) + timedelta(days=1)).isoformat()
            daily_update_payload = pipeline.run_daily_update(
                anchor_run_id=run_id,
                as_of=update_as_of,
            )

        timeout_audit = exercise_timeout_failover(
            FailoverPolicy.from_config(self.context.read_yaml("model.yaml"))
        )
        write_json(timeout_audit, out_dir / "timeout_failover_audit.json")
        artifact_verification = pipeline.verify_run(run_id)
        visual_qa = VisualQAChecklist().run(
            out_dir,
            expected_offices=self._expected_offices(scenario_obj),
        )
        live_scope = self._live_2026_source_scope(scenario_obj, as_of)
        payload = {
            "run_id": run_id,
            "scenario": scenario,
            "as_of": as_of,
            "inference_engine": inference_engine,
            "bayesian_backend": bayesian_backend,
            "output_dir": str(out_dir),
            "fixture_scope": {
                "status": "fixture_verification",
                "expected_offices": self._expected_offices(scenario_obj),
                "governor_status": "fixture_covered",
                "president_tracker_status": "fixture_non_control_tracker",
                "live_2026_status": live_scope["status"],
                "live_source_scope": live_scope,
            },
            "artifact_verification": artifact_verification,
            "visual_qa": visual_qa,
            "daily_update": daily_update_payload,
            "timeout_failover_audit": timeout_audit,
            "passed": bool(artifact_verification["passed"])
            and bool(visual_qa["passed"])
            and bool(timeout_audit["passed"]),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        write_json(payload, out_dir / "phase8_verification.json")
        return payload

    @staticmethod
    def _expected_offices(scenario: Scenario) -> list[str]:
        for key in ("expected_offices", "office_types", "offices"):
            raw = scenario.payload.get(key)
            if isinstance(raw, list):
                return [str(item).lower() for item in raw]
            if isinstance(raw, str) and raw.strip():
                return [part.strip().lower() for part in raw.split(",") if part.strip()]
        office = scenario.payload.get("office_type")
        return [str(office).lower()] if office else []

    def _live_2026_source_scope(self, scenario: Scenario, as_of: str) -> dict[str, Any]:
        manifest_path = self.context.curated_dir / "source_manifest.parquet"
        expected_offices = set(self._live_source_required_offices(scenario))
        target_year = int(as_of[:4])
        if not manifest_path.exists():
            return {
                "status": "not_claimed",
                "reason": "curated source manifest is missing",
                "target_year": target_year,
                "live_source_ids": [],
                "live_2026_rows": 0,
                "model_signal_2026_rows": 0,
                "rows_by_table": {},
                "model_signal_rows_by_table": {},
                "covered_offices": [],
                "model_signal_covered_offices": [],
                "expected_offices": sorted(expected_offices),
            }
        manifest = pl.read_parquet(manifest_path)
        live_sources = manifest.filter(
            ~pl.col("url").cast(pl.Utf8).str.starts_with("file://") & (pl.col("status") != "failed")
        )
        live_source_ids = [str(value) for value in live_sources["source_id"].unique().to_list()]
        if not live_source_ids:
            return {
                "status": "not_claimed",
                "reason": "no successful non-file sources in source manifest",
                "target_year": target_year,
                "live_source_ids": [],
                "live_2026_rows": 0,
                "model_signal_2026_rows": 0,
                "rows_by_table": {},
                "model_signal_rows_by_table": {},
                "covered_offices": [],
                "model_signal_covered_offices": [],
                "expected_offices": sorted(expected_offices),
            }

        races = self._curated_table("races")
        live_2026_rows = 0
        signal_2026_rows = 0
        covered_offices: set[str] = set()
        signal_covered_offices: set[str] = set()
        by_table: dict[str, int] = {}
        signal_by_table: dict[str, int] = {}
        signal_tables = self._model_signal_tables()
        for table in [
            "races",
            "options",
            "polls",
            "fundamentals",
            "market_quotes",
            "public_signals",
        ]:
            frame = self._curated_table(table)
            if frame.is_empty() or "source_id" not in frame.columns:
                continue
            frame = frame.filter(pl.col("source_id").is_in(live_source_ids))
            if frame.is_empty():
                continue
            scoped = self._with_race_scope(frame, races)
            if "cycle" in scoped.columns:
                scoped = scoped.filter(pl.col("cycle") == target_year)
            if scoped.is_empty():
                continue
            row_count = scoped.height
            by_table[table] = row_count
            live_2026_rows += row_count
            if "office_type" in scoped.columns:
                table_offices = {
                    str(value).lower()
                    for value in scoped["office_type"].drop_nulls().unique().to_list()
                }
                covered_offices.update(table_offices)
            else:
                table_offices = set()

            signal_scoped = self._signal_scoped_frame(table, scoped, signal_tables)
            if not signal_scoped.is_empty():
                signal_2026_rows += signal_scoped.height
                signal_by_table[table] = signal_scoped.height
                if "office_type" in signal_scoped.columns:
                    signal_covered_offices.update(
                        str(value).lower()
                        for value in signal_scoped["office_type"].drop_nulls().unique().to_list()
                    )
                else:
                    signal_covered_offices.update(table_offices)

        status = (
            "claimed"
            if signal_2026_rows > 0 and expected_offices.issubset(signal_covered_offices)
            else "metadata_only"
            if live_2026_rows > 0 and expected_offices.issubset(covered_offices)
            else "partial"
            if live_2026_rows > 0
            else "not_claimed"
        )
        return {
            "status": status,
            "reason": (
                "live 2026 rows cover every expected office"
                if status == "claimed"
                else (
                    "live 2026 rows only cover every expected office through metadata "
                    "or non-admitted signals"
                )
                if status == "metadata_only"
                else "live rows do not cover every expected 2026 office"
                if status == "partial"
                else "no successful non-file source contributed 2026 rows"
            ),
            "target_year": target_year,
            "live_source_ids": sorted(live_source_ids),
            "live_2026_rows": live_2026_rows,
            "model_signal_2026_rows": signal_2026_rows,
            "rows_by_table": by_table,
            "model_signal_rows_by_table": signal_by_table,
            "covered_offices": sorted(covered_offices),
            "model_signal_covered_offices": sorted(signal_covered_offices),
            "expected_offices": sorted(expected_offices),
        }

    def _curated_table(self, table: str) -> pl.DataFrame:
        path = self.context.curated_dir / f"{table}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    def _live_source_required_offices(self, scenario: Scenario) -> list[str]:
        raw = scenario.payload.get("live_source_required_offices")
        if isinstance(raw, list):
            return [str(item).lower() for item in raw]
        if isinstance(raw, str) and raw.strip():
            return [part.strip().lower() for part in raw.split(",") if part.strip()]
        return self._expected_offices(scenario)

    def _model_signal_tables(self) -> set[str]:
        trusted = dict(self.context.read_yaml("model.yaml").get("trusted_components", {}))
        tables = {"polls", "fundamentals", "market_quotes"}
        if bool(trusted.get("public_signals")):
            tables.add("public_signals")
        return tables

    @staticmethod
    def _signal_scoped_frame(
        table: str, scoped: pl.DataFrame, signal_tables: set[str]
    ) -> pl.DataFrame:
        if table not in signal_tables:
            return pl.DataFrame()
        if table == "public_signals" and "z_score" in scoped.columns:
            return scoped.filter(pl.col("z_score").fill_null(0.0).abs() > 1e-12)
        return scoped

    @staticmethod
    def _with_race_scope(frame: pl.DataFrame, races: pl.DataFrame) -> pl.DataFrame:
        if "race_id" not in frame.columns or races.is_empty() or "race_id" not in races.columns:
            return frame
        race_columns = [
            column for column in ["race_id", "cycle", "office_type"] if column in races.columns
        ]
        if len(race_columns) == 1:
            return frame
        race_scope = races.select(race_columns).rename(
            {column: f"race_scope_{column}" for column in race_columns if column != "race_id"}
        )
        scoped = frame.join(race_scope, on="race_id", how="left")
        drop_columns: list[str] = []
        for column in race_columns:
            if column == "race_id":
                continue
            scope_column = f"race_scope_{column}"
            if scope_column not in scoped.columns:
                continue
            if column in scoped.columns:
                scoped = scoped.with_columns(
                    pl.coalesce([pl.col(column), pl.col(scope_column)]).alias(column)
                )
                drop_columns.append(scope_column)
            else:
                scoped = scoped.rename({scope_column: column})
        return scoped.drop(drop_columns) if drop_columns else scoped
