from __future__ import annotations

import hashlib
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.ingest.sources import SourceDefinition, SourceRegistry
from election_outcomes.storage.io import read_json, write_json, write_parquet


@dataclass(frozen=True)
class SyncResult:
    manifest: pl.DataFrame
    fetched_sources: int
    skipped_sources: int
    failed_sources: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SyncRunner:
    """Incremental local sync for source definitions."""

    def __init__(self, context: ProjectContext, registry: SourceRegistry | None = None) -> None:
        self.context = context
        self.registry = registry or SourceRegistry.from_context(context)

    def run(self) -> SyncResult:
        self.context.raw_dir.mkdir(parents=True, exist_ok=True)
        self.context.state_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.context.state_dir / "sync_state.json"
        previous = read_json(state_path) if state_path.exists() else {}

        rows: list[dict[str, object]] = []
        state: dict[str, str] = {}
        fetched = skipped = failed = 0
        retrieved_at = datetime.now(UTC).isoformat()

        for source in self.registry.sources:
            try:
                row, did_fetch = self._sync_one(source, previous, retrieved_at)
                fetched += int(did_fetch)
                skipped += int(not did_fetch)
                state[source.id] = str(row["content_hash"])
            except Exception as exc:  # pragma: no cover - defensive manifest path
                failed += 1
                row = self._failure_row(source, retrieved_at, exc)
            rows.append(row)

        manifest = pl.DataFrame(rows)
        write_parquet(manifest, self.context.raw_dir / "source_manifest.parquet")
        write_json(state, state_path)
        return SyncResult(manifest, fetched, skipped, failed)

    def _sync_one(
        self,
        source: SourceDefinition,
        previous: dict[str, str],
        retrieved_at: str,
    ) -> tuple[dict[str, object], bool]:
        if source.type == "http_csv":
            return self._sync_http(source, previous, retrieved_at)
        if source.type != "fixture":
            raise ValueError(f"Unsupported source type: {source.type}")
        if source.path is None:
            raise ValueError(f"Fixture source {source.id} requires a local path")
        content_hash = _sha256(source.path)
        raw_path = self.context.raw_dir / source.id / f"{content_hash}{source.path.suffix}"
        did_fetch = previous.get(source.id) != content_hash or not raw_path.exists()
        if did_fetch:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.path, raw_path)
        return (
            {
                "source_id": source.id,
                "table": source.table,
                "url": source.url,
                "raw_path": str(raw_path),
                "retrieved_at": retrieved_at,
                "content_hash": content_hash,
                "license": source.license,
                "parser_version": source.parser_version,
                "auth_mode": source.auth_mode,
                "status": "fetched" if did_fetch else "unchanged",
                "error": "",
                "downstream_usage": "",
            },
            did_fetch,
        )

    def _sync_http(
        self,
        source: SourceDefinition,
        previous: dict[str, str],
        retrieved_at: str,
    ) -> tuple[dict[str, object], bool]:
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "election-outcomes/0.1 (+research forecast sync)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        content_hash = hashlib.sha256(payload).hexdigest()
        suffix = self._http_suffix(source)
        raw_path = self.context.raw_dir / source.id / f"{content_hash}{suffix}"
        did_fetch = previous.get(source.id) != content_hash or not raw_path.exists()
        if did_fetch:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
        return (
            {
                "source_id": source.id,
                "table": source.table,
                "url": source.url,
                "raw_path": str(raw_path),
                "retrieved_at": retrieved_at,
                "content_hash": content_hash,
                "license": source.license,
                "parser_version": source.parser_version,
                "auth_mode": source.auth_mode,
                "status": "fetched" if did_fetch else "unchanged",
                "error": "",
                "downstream_usage": "",
            },
            did_fetch,
        )

    @staticmethod
    def _http_suffix(source: SourceDefinition) -> str:
        if source.path and source.path.suffix:
            return source.path.suffix
        parsed = urllib.parse.urlparse(source.url)
        suffix = Path(parsed.path).suffix
        return suffix or ".dat"

    @staticmethod
    def _failure_row(
        source: SourceDefinition,
        retrieved_at: str,
        exc: Exception,
    ) -> dict[str, object]:
        return {
            "source_id": source.id,
            "table": source.table,
            "url": source.url,
            "raw_path": "",
            "retrieved_at": retrieved_at,
            "content_hash": "",
            "license": source.license,
            "parser_version": source.parser_version,
            "auth_mode": source.auth_mode,
            "status": "failed",
            "error": str(exc),
            "downstream_usage": "",
        }
