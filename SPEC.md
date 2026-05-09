# Election Outcomes Specification

## Summary

Build a U.S.-only, research-grade election forecasting engine that can be run manually
from time to time. Each run refreshes public data incrementally, snapshots source
provenance, builds a race catalog, runs a hybrid statistical ensemble, backtests trusted
components, and emits auditable artifacts with measurable rewards.

The default implementation is fixture-backed so the full modeling, artifact, reward,
plotting, and validation contract is deterministic. A first opt-in live registry
(`configs/sources_live.yaml`) adds keyless HTTP CSV polling ingestion for one real race.

## Required Run Outcomes

Every `forecast run` must create `artifacts/runs/<run_id>/` with:

- `race_catalog.parquet`: discovered races with Tier A/B/C status and tier reasons.
- `race_forecasts.parquet`: winner probabilities, vote-share means/medians, interval
  columns, model tier, data-quality flags, driver attribution, uncertainty explanation,
  source lineage, and model-config lineage.
- `forecast_draws.parquet`: simulation draws by race, option, turnout, vote share,
  winner flag, and correlated-error draw id.
- `control_forecasts.parquet`: seat/control probabilities, seat distributions, and
  tipping-point races.
- `ecosystem_forecasts.parquet`: turnout, demographic-composition support, ballot-measure
  support, recount risk, and certification-risk probabilities.
- `source_manifest.parquet`: source ids, URLs/paths, retrieval timestamps, content hashes,
  parser versions, license/terms notes, status, and downstream usage.
- `diagnostics.html`: top-line summary, Electoral College distribution, scorecards,
  reward status, source coverage, model-quality section, and embedded plots.
- `reward_card.json`: machine-readable reward checks.
- `methodology_snapshot.md`: model version, config, source coverage, and limitations.
- `model_card.md`: learned/configured/placeholder parameter status, component admission,
  backtest sample status, covariance status, and source coverage for the run.
- `silver_benchmark.json` and `silver_benchmark.html`: methodology-readiness comparison
  against public Silver/FiveThirtyEight forecast traits and source anchors, scored on
  the explicit four-tier `absent`/`scaffold`/`functional`/`production` scale.
- `reproducibility_fingerprint.json`: stable artifact hashes excluding volatile
  retrieval/status fields, with same-run-id comparison status when available.
- `plot_manifest.json`: projection, calibration, trajectory, stability, model-quality,
  and benchmark plot index.
- `plots/`: static PNG diagnostics.
- `performance.json`: requested acceleration engine, actual engine, parallel mode,
  Numba availability, thread count, and simulation count.
- `comparisons/<comparison_id>/`: optional forecast-vs-actual comparison artifacts
  created by `results compare`.

Forecasts must distinguish:

- Candidate races: president, Senate, House, governors, state offices, local offices,
  primaries, runoffs, and ranked-choice races where data permits.
- Ballot measures: yes/no vote-share and pass probability.
- Control outcomes: seat-count and governing-control distributions.
- Ecosystem outcomes: turnout, demographic turnout support, recount risk, and
  certification-delay risk.

## Verifiable Rewards

Use vector rewards so the system cannot hide weak behavior behind one aggregate score:

- `R0_build`: `uv sync`, `ruff check`, `ruff format --check`, and
  `pytest --cov=src/election_outcomes --cov-fail-under=90` pass.
- `R1_reproducibility`: fixed inputs and run config produce the same stable artifact
  fingerprint when the same `run_id` is rerun, excluding wall-clock retrieval metadata
  and incremental-sync status fields.
- `R2_provenance`: forecast rows trace to source hashes and model-config hashes.
- `R3_sync_integrity`: incremental sync fetches new/changed sources only, dedupes records,
  and records failures explicitly.
- `R4_calibration`: backtests report Brier score, log score, calibration line, expected
  calibration error, and interval coverage.
- `R5_baseline_competition`: trusted ensemble beats or matches declared baselines on a
  rolling-origin holdout with enough historical rows; otherwise it is labeled
  experimental.
- `R6_component_admission`: polls, fundamentals, markets, and public signals enter trusted
  output only when ablation evidence supports them.
- `R7_sparse_honesty`: Tier C races are tracked but do not receive trusted probabilities.
- `R8_uncertainty_quality`: reported forecast intervals have empirical coverage within
  configured tolerance on rolling-origin historical samples large enough to trust.
- `R9_public_signal_discipline`: news/pageview/social-like signals remain experimental
  until leakage and ablation checks pass.
- `R10_explainability`: forecast rows include tier reason, data-quality flags, top
  drivers, component contributions, and uncertainty explanation.
- `R11_plot_contract`: calibration and projection plots are generated and referenced by
  `plot_manifest.json`.
- `R12_performance_contract`: forecast runs record the configured acceleration path and
  use the Numba engine when it is requested and available.

Primary baselines:

- Historical partisan lean/fundamentals only.
- Polling average where polls exist.
- Market-implied where liquid markets exist.
- Incumbent/party prior for sparse races.
- Previous-cycle swing baseline.

## Repository Design

```text
election-outomes/
  pyproject.toml
  README.md
  AGENTS.md
  SPEC.md
  configs/
    sources.yaml
    model.yaml
    backtests.yaml
    tiers.yaml
  fixtures/
    *.csv
  schemas/
    raw_contracts/
    curated_tables/
    artifact_contracts/
  src/election_outcomes/
    cli.py
    config/
    ingest/
    normalize/
    storage/
    features/
    models/
    scoring/
    reports/
  tests/
    unit/
    integration/
    golden_fixtures/
  data/       # gitignored raw/cache/curated local lake
  artifacts/  # gitignored run outputs
  docs/
```

Important config contracts:

- `configs/sources.yaml`: default fixture source registry and parser metadata.
- `configs/sources_live.yaml`: opt-in live source overlay for HTTP CSV/API adapters.
- `configs/scenarios.yaml`: scenario filters and defaults such as 2024 presidential
  state-level runs.
- `configs/model.yaml`: model version, seed, simulation count, component weights,
  trusted-component flags, uncertainty settings, performance settings, and reward
  thresholds.
- `configs/tiers.yaml`: Tier A/B/C thresholds and sparse-race policy.
- `configs/backtests.yaml`: rolling-origin settings, as-of date sweep, metrics, and
  baselines.

Current implementation note: the repo runs a rolling-origin component refit harness and
writes `rolling_predictions.parquet`, `component_admission.json`, and
`residual_covariance.parquet`. The rolling-origin harness evaluates multiple pre-election
as-of cuts when data exists (`T-90/T-60/T-30/T-7/T-1` by default). It must not certify
`R5`, `R6`, or `R8` until the historical race store reaches the configured sample
threshold.

## Modeling Specification

For the detailed statistical rationale, see
[`docs/technical_appendix.md`](docs/technical_appendix.md).

Canonical latent targets:

- Candidate races: latent election-day vote-share simplex and major-party margin.
- Ballot measures: latent yes-share and pass probability.
- Control: derived from correlated race-level simulations.
- Turnout: separate turnout-rate and vote-count projections by geography and election type.

Component models:

- Polling model: sample-size inverse-variance style polling estimates with methodology,
  population, sponsor, time-decay, house-effect hooks, and posterior uncertainty proxy.
- Fundamentals model: historical vote share, partisan lean, incumbency, finance, economy,
  demographics, turnout history, and election type through a standardized ridge fit when
  enough prior-cycle rows exist, otherwise explicit defaults.
- Market model: public read-only market probabilities adjusted for liquidity and spread,
  then mapped to vote-share proxy through a configurable normal inverse-CDF scale.
- Public-signal model: news/pageview/official-release features, experimental by default.
- Ensemble: weighted blend of trusted components with vote-share normalization by race
  and unnormalized marginal win-probability reporting.
- Simulation: structured-factor election-error draws with national, residual-covariance,
  region, and office factors plus heavy-tailed local error, race-level winners, vote
  shares, turnout, recount risk, certification-risk proxy, and thresholded control
  outcomes.
- Performance: two-option race draw generation uses a Numba parallel kernel with a Python
  fallback, while table transforms should stay vectorized through Polars/DuckDB.

Planned statistical upgrade path:

- Replace deterministic polling component with a hierarchical Bayesian model using
  `cmdstanpy` or NumPyro behind the same component/artifact schema.
- Calibrate component weights with rolling-origin backtests rather than static config.
- Expand the current residual-covariance shrinkage model with a deeper historical
  state/down-ballot residual panel; single-observation covariance must be withheld rather
  than invented.
- Extend live source adapters while preserving raw-source hash and curated-table
  contracts.
- Extend Numba kernels to multi-option/ranked-choice simulation and score aggregation when
  those paths become measurable bottlenecks.

## Plotting Specification

Every forecast run must emit calibration and projection visuals:

- Calibration curve.
- Brier score by component.
- Historical interval coverage.
- Winner-probability bars.
- Vote-share interval projections.
- Seat/control projections.
- Turnout/recount-risk projections.
- Forecast coverage by tier.
- Electoral College distribution and representative simulation swarm for presidential
  scenarios.
- Polling probability trajectories when rolling-origin polling probability and as-of
  cut columns are available.
- Simulation probability convergence when draw-level winner rows are available.
- MCMC-style posterior simulation chain traces for Electoral College totals.
- Kalman posterior uncertainty traces for state-space polling fits.

Plots are generated from local artifacts and do not require API credentials.

## Performance Specification

Performance settings live under `performance` in `configs/model.yaml`:

- `engine`: requested acceleration engine, currently `numba` or `python`.
- `parallel`: enables parallel Numba execution.
- `numba_threads`: optional positive thread cap; `0` uses Numba's default.
- `benchmark_draws` and `benchmark_repeats`: benchmark CLI defaults.

Benchmark command:

```bash
uv run election-outcomes benchmark run --as-of 2026-05-08 --run-id perf
```

Benchmark output:

```text
artifacts/benchmarks/<run_id>/performance_benchmark.json
```

## Historical Result Comparison

An existing forecast run can be compared against curated actual results:

```bash
uv run election-outcomes results compare \
  --forecast-run-id 2024-presidential \
  --comparison-id 2024-presidential-actuals \
  --cycle 2024 \
  --office-type president
```

Comparison output:

```text
artifacts/runs/<forecast_run_id>/comparisons/<comparison_id>/
  result_comparison.parquet
  race_outcomes.parquet
  largest_misses.parquet
  result_comparison_summary.json
  result_comparison.html
  narrative.md
  plots/
```

The comparison reports winner accuracy, mean absolute vote-share error, Brier score, and
upset count over the filtered races/options. Presidential comparisons also report
state-level winner accuracy, modeled Electoral College winner accuracy with an explicit
`full_electoral_college` or `modeled_state_slice` scope, actual-winner probabilities,
and the largest option-level vote-share misses.

A same-date presidential cycle evaluation should be available as a first-class command:

```bash
uv run election-outcomes results cycle-eval \
  --run-id oct5-presidential-cycle-eval \
  --cycles 2008,2012,2016,2020,2024 \
  --as-of-mm-dd 10-05
```

Cycle-eval output:

```text
artifacts/cycle_evals/<run_id>/
  cycle_summary.parquet
  cycle_summary.json
  cycle_eval.html
  narrative.md
  plots/
```

The summary must report Electoral College winner probability, EV p10/p50/p90, EC winner
accuracy, state accuracy, Brier score, vote-share MAE, upset count, missed states, and
links to each cycle's diagnostics and comparison report.

## API Credentials

No external credentials are required for fixture-backed runs, backtests, or plots.

Likely live-ingestion credentials:

- `GOOGLE_CIVIC_API_KEY` for Google Civic Information API.
- `CENSUS_API_KEY` for higher-volume Census API usage.
- `GDELT_API_KEY` for GDELT Cloud endpoints that require bearer auth.

Usually public/read-only first:

- Polymarket market/event data.
- Kalshi public market data.
- Wikimedia pageviews.
- FEC and official election-office downloads where available.

Every live adapter must re-check current terms/rate limits and write auth mode, URL,
retrieval time, content hash, parser version, and failures to the source manifest.

## Acceptance Criteria

The repo is healthy only when all of these pass:

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest --cov=src/election_outcomes --cov-fail-under=90
```

A working forecast smoke test is:

```bash
uv run election-outcomes forecast run --as-of 2026-05-08 --run-id smoke
uv run election-outcomes benchmark run --as-of 2026-05-08 --run-id perf
```

The smoke run must create all required Parquet/JSON/HTML/Markdown/PNG artifacts and the
reward card must pass all implemented rewards except `R0_build`, which is validated by
the external commands above.
