from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from election_outcomes.storage.io import read_json, write_json


@dataclass(frozen=True)
class VisualQAChecklist:
    """Automated Phase 8 dashboard/readiness checklist."""

    max_plot_bytes: int = 2_000_000

    def run(self, run_dir: Path, expected_offices: list[str] | None = None) -> dict[str, Any]:
        expected = [office.lower() for office in (expected_offices or [])]
        checks = [
            self._plot_manifest(run_dir),
            self._diagnostics_html(run_dir),
            self._model_card(run_dir),
            self._reward_card(run_dir),
            self._fingerprint(run_dir),
            self._office_artifacts(run_dir, expected),
            self._silver_benchmark(run_dir),
        ]
        payload = {
            "passed": all(bool(check["passed"]) for check in checks),
            "checks": checks,
        }
        write_json(payload, run_dir / "visual_qa_checklist.json")
        return payload

    def _plot_manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "plot_manifest.json"
        if not path.exists():
            return {"name": "plot_manifest_visual_qa", "passed": False, "detail": "missing"}
        manifest = read_json(path)
        entries = [
            entry
            for values in manifest.values()
            if isinstance(values, list)
            for entry in values
            if isinstance(entry, dict)
        ]
        missing = []
        empty = []
        oversized = []
        untitled = []
        for entry in entries:
            rel_path = str(entry.get("path", ""))
            title = str(entry.get("title", "")).strip()
            plot_path = run_dir / rel_path
            if not title:
                untitled.append(rel_path)
            if not plot_path.exists():
                missing.append(rel_path)
                continue
            size = plot_path.stat().st_size
            if size == 0:
                empty.append(rel_path)
            if size > self.max_plot_bytes:
                oversized.append({"path": rel_path, "bytes": size})
        return {
            "name": "plot_manifest_visual_qa",
            "passed": bool(entries)
            and not missing
            and not empty
            and not oversized
            and not untitled,
            "detail": {
                "plot_count": len(entries),
                "missing": missing,
                "empty": empty,
                "oversized": oversized,
                "untitled": untitled,
            },
        }

    @staticmethod
    def _diagnostics_html(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "diagnostics.html"
        if not path.exists():
            return {"name": "diagnostics_html_visual_qa", "passed": False, "detail": "missing"}
        html = path.read_text(encoding="utf-8")
        required = ["posterior_diagnostics", "fundamentals_prior", "source_audit"]
        missing = [anchor for anchor in required if f'id="{anchor}"' not in html]
        return {
            "name": "diagnostics_html_visual_qa",
            "passed": not missing and path.stat().st_size > 0,
            "detail": {"missing_anchors": missing, "bytes": path.stat().st_size},
        }

    @staticmethod
    def _model_card(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "model_card.md"
        if not path.exists():
            return {"name": "model_card_office_methodology", "passed": False, "detail": "missing"}
        text = path.read_text(encoding="utf-8")
        required = ["office_methodology", "senate_joint", "house_hierarchical", "cross_office"]
        missing = [token for token in required if token not in text]
        return {
            "name": "model_card_office_methodology",
            "passed": not missing,
            "detail": {"missing_tokens": missing},
        }

    @staticmethod
    def _reward_card(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "reward_card.json"
        if not path.exists():
            return {"name": "reward_card_r0_r15", "passed": False, "detail": "missing"}
        rewards = read_json(path).get("rewards", {})
        expected = [f"R{index}_" for index in range(16)]
        present = sorted(str(key) for key in rewards)
        missing = [
            prefix for prefix in expected if not any(key.startswith(prefix) for key in present)
        ]
        return {
            "name": "reward_card_r0_r15",
            "passed": not missing,
            "detail": {"missing_prefixes": missing, "reward_count": len(present)},
        }

    @staticmethod
    def _fingerprint(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "reproducibility_fingerprint.json"
        if not path.exists():
            return {"name": "reproducibility_fingerprint", "passed": False, "detail": "missing"}
        payload = read_json(path)
        return {
            "name": "reproducibility_fingerprint",
            "passed": bool(payload.get("cross_run_verified")),
            "detail": {
                "compared_to_previous": bool(payload.get("compared_to_previous")),
                "cross_run_verified": bool(payload.get("cross_run_verified")),
            },
        }

    @staticmethod
    def _office_artifacts(run_dir: Path, expected_offices: list[str]) -> dict[str, Any]:
        required = []
        if "senate" in expected_offices:
            required.extend(["senate_joint_posterior.parquet", "senate_seat_posterior.parquet"])
        if "house" in expected_offices:
            required.extend(
                ["house_hierarchical_posterior.parquet", "house_seat_posterior.parquet"]
            )
        if "governor" in expected_offices:
            required.append("governor_seat_posterior.parquet")
        if len(set(expected_offices) & {"senate", "house", "governor"}) >= 2:
            required.append("cross_office_posterior.parquet")
        missing = [name for name in required if not (run_dir / name).exists()]
        empty = [
            name
            for name in required
            if (run_dir / name).exists() and (run_dir / name).stat().st_size == 0
        ]
        return {
            "name": "office_methodology_artifacts",
            "passed": not missing and not empty,
            "detail": {"required": required, "missing": missing, "empty": empty},
        }

    @staticmethod
    def _silver_benchmark(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "silver_benchmark.json"
        if not path.exists():
            return {"name": "silver_benchmark", "passed": False, "detail": "missing"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "name": "silver_benchmark",
            "passed": bool(payload.get("status")) and bool(payload.get("rows")),
            "detail": {
                "status": payload.get("status"),
                "summary_score": payload.get("summary_score"),
            },
        }
