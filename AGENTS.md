# Agent Notes

This repo is a Python `src/` layout project managed with `uv`. The canonical design
contract is `SPEC.md`; keep implementation, docs, tests, and artifacts aligned with it.

## Non-Negotiable Checks

For every task, the agent must ensure the repository still satisfies:

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest --cov=src/civic_signal --cov-fail-under=90
```

Do not lower the coverage gate. If a task cannot run these commands, explain exactly why
and state the residual risk.

## Documentation Rule

Every task must update `README.md` when user-facing commands, artifacts, model behavior,
configuration, source requirements, or operational workflow changes. If the task changes
the design contract, update `SPEC.md` too.

## Implementation Rules

- Keep model changes auditable through config and tests.
- Preserve source provenance: source URL/path, retrieval time, content hash, parser
  version, status, and downstream usage must remain traceable.
- Keep Tier C sparse races honest: track them, but withhold trusted probabilities.
- Do not commit `data/` or `artifacts/`; they are generated local run products.
- Prefer deterministic fixture-backed tests before adding live-source behavior.
- When adding live adapters, record auth mode and terms/limits assumptions in docs and
  source manifests.
- Preserve `uv`, `ruff`, and the 90% coverage gate as repo standards.
- Preserve performance expectations: use vectorized Polars/DuckDB for table work, use
  Numba/parallel kernels for repeated numerical loops when practical, and run
  `uv run civic-signal benchmark run --as-of 2026-05-08 --run-id perf` after
  substantial simulation or scoring changes.
