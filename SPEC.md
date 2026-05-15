# Election Outcomes Specification

## Summary

Build a U.S.-only, research-grade election forecasting engine that can be run manually
from time to time. Each run refreshes public data incrementally, snapshots source
provenance, builds a race catalog, runs a hybrid statistical ensemble, backtests trusted
components, and emits auditable artifacts with measurable rewards.

The default implementation is fixture-backed so the full modeling, artifact, reward,
plotting, and validation contract is deterministic. The opt-in live registry
(`configs/sources_live.yaml`) adds keyless HTTP CSV polling ingestion for the 2020
Wisconsin presidential archive, 2026 Senate/Governor/House poll streams when upstream
FiveThirtyEight/Datasette rows exist, keyless FRED UNRATE macro fundamentals for the
compact 2026 multi-office smoke races, and neutral Wikipedia race-presence metadata.

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
  support, and close-margin administrative-risk fields that are withheld unless explicitly
  enabled as experimental output.
- `source_manifest.parquet`: source ids, URLs/paths, retrieval timestamps, content hashes,
  parser versions, license/terms notes, status, and downstream usage.
- `diagnostics.html`: top-line summary, paired Electoral College distribution and
  simulation swarm, scorecards, reward status, source coverage, model-quality section,
  and embedded plots.
- `reward_card.json`: machine-readable reward checks.
- `methodology_snapshot.md`: model version, config, source coverage, and limitations.
- `model_card.md`: learned/configured/placeholder parameter status, component admission,
  backtest sample status, covariance status, and source coverage for the run.
- `silver_benchmark.json` and `silver_benchmark.html`: methodology-readiness comparison
  against public Silver/FiveThirtyEight forecast traits and source anchors, scored on
  the explicit four-tier `absent`/`scaffold`/`functional`/`production` scale for the
  configured run scope, not for unmodeled nationwide live coverage.
- `reproducibility_fingerprint.json`: stable artifact hashes excluding volatile
  retrieval/status fields, with same-run-id comparison status when available.
- `plot_manifest.json`: projection, calibration, trajectory, stability, model-quality,
  and benchmark plot index.
- `plots/`: static PNG diagnostics.
- `performance.json`: requested acceleration engine, actual engine, parallel mode,
  Numba availability, thread count, and simulation count.
- `recalibration_map.parquet`: persisted Platt/logit probability calibration map when
  a latest rolling-origin backtest map is available and applied to the run.
- `posterior_draws.parquet`: race-constrained Bayesian election-day latent-share
  posterior draws unless `forecast run --inference-engine kalman` is used.
- `state_space_trajectory.parquet`: Bayesian trajectory summaries by race,
  option, and date with model/source lineage hashes.
- `pollster_house_effects.parquet`: Bayesian empirical-Bayes pollster house-effect
  estimates used by the Bayesian polling bridge.
- `posterior_diagnostics.json`: Bayesian posterior diagnostics, draw count,
  parameterization, fallback status, and lineage hashes.
- `fundamentals_prior.parquet`: Election-Day fundamentals prior used by the
  Bayesian polling bridge.
- `seat_posterior.parquet`: draw-level seat/control posterior summaries. Senate,
  House, and Governor body-specific posterior files are emitted when those rows exist.
- `posterior_history.parquet`, `latest_daily_update.json`, and `updates/<as-of>/`:
  created by `forecast update` from a Bayesian anchor run.
- `timeout_failover_audit.json`: Phase 8 forced-timeout audit showing the configured
  Bayesian NUTS fallback order without marking the forecast itself as a fallback.
- `phase8_verification.json` and `visual_qa_checklist.json`: created by
  `verify run --scenario ...` after orchestrating the fixture-backed multi-office
  verification path.
- `methodology_readiness.json`: created by `verify readiness` under
  `artifacts/readiness/<run_id>/` to audit the Bayesian default-switch contract against
  dependencies, docs/config defaults, Phase 8 artifacts, reward gates, live-source scope,
  and rolling-origin legacy comparison evidence.
- `verification.json`: created by `verify run` after required artifact, schema, plot,
  and reward checks.
- `senate_joint_posterior.parquet`: Phase 4 office-methodology artifact when Senate
  posterior rows exist, with shared Senate environment, class effect, state deviation,
  and holdover-aware seat posterior summaries. NUTS runs label this as a decomposition
  of the fitted state-space draw stream; analytic runs label it as a bridge.
- `house_hierarchical_posterior.parquet`: Phase 5 office-methodology artifact when
  House posterior rows exist, with redistricting-era partition, state effects, district
  idiosyncrasy, sparse-district flags, and non-dense covariance method.
- `cross_office_posterior.parquet`: Phase 7 office-methodology artifact when at least
  two midterm offices share posterior draws, with national environment and per-office
  offsets on the common draw stream.
- `comparisons/<comparison_id>/`: optional forecast-vs-actual comparison artifacts
  created by `results compare`.

Forecasts must distinguish:

- Candidate races: president, Senate, House, governors, state offices, local offices,
  primaries, runoffs, and ranked-choice races where data permits.
- Ballot measures: yes/no vote-share and pass probability.
- Control outcomes: seat-count and governing-control distributions.
- Ecosystem outcomes: turnout and demographic turnout support. Recount and
  certification-delay proxies stay withheld unless the experimental close-margin proxy is
  explicitly enabled or replaced by a calibrated administrative-risk model.

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
  calibration error, interval coverage, learned ensemble weights, and the probability
  calibration transform applied to published marginal race probabilities.
  `verify historical-calibration` also writes a compact 2022 Senate/House/Governor
  audit under `artifacts/historical_calibration/<run_id>/` with per-office ECE gates for
  the Phase 4, Phase 5, and Phase 7 office-methodology plan. The optional
  `sources_historical_panels.yaml` registry expands that audit to production-dimension
  synthetic Senate and House panels without changing default fixture runtime.
- `R5_baseline_competition`: trusted ensemble beats or matches declared baselines on a
  rolling-origin holdout with enough historical rows; otherwise it is labeled
  experimental.
- `R6_component_admission`: polls, fundamentals, markets, and public signals enter trusted
  output only when ablation evidence supports them. Forecast runs apply the current
  rolling-origin admission artifact before ensemble weighting; components rejected by
  admission can still appear as diagnostics or priors, but are not counted as trusted
  ensemble inputs.
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
- `R13_posterior_quality`: Bayesian runs emit posterior diagnostics with sufficient
  draws, no divergences, and valid R-hat/ESS checks when MCMC diagnostics are available.
- `R14_calibrated_publication`: published probabilities either use a persisted
  recalibration map or demonstrate acceptable rolling-origin calibration without a map.
- `R15_daily_update_quality`: daily Bayesian updates pass strategy-specific quality
  gates and do not require a full refit.

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
    inference/
    models/
    observability/
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
- `configs/sources_live.yaml`: opt-in live source overlay for HTTP CSV/text/API adapters.
- `configs/scenarios.yaml`: scenario filters and defaults such as 2024 presidential
  state-level runs.
- `configs/model.yaml`: model version, seed, simulation count, component weights,
  trusted-component flags, uncertainty settings, performance settings, and reward
  thresholds.
- `configs/tiers.yaml`: Tier A/B/C thresholds and sparse-race policy.
- `configs/backtests.yaml`: rolling-origin settings, as-of date sweep, metrics, and
  baselines.

Current implementation note: the repo runs a rolling-origin component refit harness and
writes `rolling_predictions.parquet`, `component_admission.json`, `ensemble_learning.json`,
`probability_calibration.json`, `recalibration_map.parquet`,
`bayesian_hyperpriors.json`, and `residual_covariance.parquet`. The rolling-origin
harness evaluates multiple pre-election as-of cuts when data exists
(`T-90/T-60/T-30/T-7/T-1` by default). It must not certify `R5`, `R6`, or `R8` until the
historical race store reaches the configured sample threshold. Latest trusted backtest
artifacts under `artifacts/backtests/latest/` are consumed by later forecast runs. The
same harness supports `backtest run --inference-engine bayes` plus
`--bayesian-backend analytic|nuts` so the Bayesian bridge and production NUTS backend
can both be scored against the legacy Kalman path without changing global model
configuration.

`backtest refresh-hyperpriors` is the scheduled refresh surface for hyperprior drift.
It writes candidate artifacts under `artifacts/hyperprior_refreshes/<run_id>/`,
including `hyperprior_refresh_manifest.json`, scenario-local candidate hyperpriors, and
a comparison report. It must not update `artifacts/backtests/latest/`; candidate
promotion requires a separate explicit review. The refresh command accepts the same
Bayesian backend override as `backtest run`.

Phase 0 methodology spikes write `artifacts/spikes/<run_id>/comparison.json`,
`phase0_comparison.parquet`, per-engine rolling predictions, and per-engine scorecards.
The current spike compares the legacy Kalman and opt-in Bayes polling engines on the
configured presidential holdout cycle and records the Bayes-minus-Kalman log-loss gate.
Phase 0b methodology spikes write `phase0b_summary.json`,
`geometry_comparison.parquet`, and `acceleration_bakeoff.parquet`. That artifact is the
gate for centered-vs-non-centered posterior geometry and for accepting any global SMC
daily-update path. The configured production update strategy remains cached posterior
reweighting unless Phase 0b proves another non-global strategy dominates and records
fallback semantics.

## Modeling Specification

For the detailed statistical rationale, see
[`docs/technical_appendix.md`](docs/technical_appendix.md).

Canonical latent targets:

- Candidate races: latent election-day vote-share simplex and major-party margin.
- Ballot measures: latent yes-share and pass probability.
- Control: derived from correlated race-level simulations.
- Turnout: separate turnout-rate and vote-count projections by geography and election type.

Component models:

- Polling model: the Bayesian path is the production default. The legacy deterministic
  Kalman/state-space polling estimates remain available through
  `--inference-engine kalman`; they are initialized from previous vote share when
  available, with sample-size observation variance, methodology/population/sponsor
  effective-sample adjustments, iterative empirical-Bayes pollster house-effect
  shrinkage, and posterior uncertainty proxy. Forecast and backtest commands resolve
  their default inference engine from `configs/model.yaml`. The Bayesian bridge exports
  logit-normal posterior draws, state-space trajectory summaries, posterior diagnostics,
  and pollster house-effect artifacts behind the same component schema. Candidate
  offices without eligible polls may receive fundamentals-prior-only posterior draws so
  sparse House/Senate races still produce auditable uncertainty artifacts and sparse
  forecast rows. The Bayesian
  backend defaults to compact hierarchical NumPyro/NUTS with two vectorized chains, 500
  warmup iterations, 2,000 sampling iterations per chain, and a `0.99` target
  acceptance probability; `--bayesian-backend analytic` selects the deterministic bridge
  for fast smoke runs. The JAX/NumPyro/ArviZ dependencies are base dependencies so the
  NUTS path is available after plain `uv sync`. The NUTS backend pools options through
  non-centered office, geography, and race-level effects plus pollster effects. Poll
  observations use empirical-Bayes pollster house-effect adjustment, a Bayesian-specific
  7-day recency half-life, population screen, methodology weights, and poll-age process
  variance so stale or lower-quality polls do not dominate the as-of latent state. The
  exported posterior draw artifact inflates that state from `as_of` to election day with
  `bayesian.state_space.forecast_drift_sd_per_sqrt_day` and constrains all options
  within each race to sum to one before ensemble calibration.
- Fundamentals model: historical vote share, partisan lean, incumbency, finance, economy,
  demographics, turnout history, and election type through a standardized ridge fit when
  enough prior-cycle rows exist, otherwise explicit defaults. Bayesian runs convert the
  fitted fundamentals model into an Election-Day prior artifact.
- Market model: public read-only market probabilities adjusted for liquidity and spread,
  then mapped to vote-share proxy through a configurable normal inverse-CDF scale.
- Public-signal model: news/pageview/official-release features, experimental by default.
- Ensemble: weighted blend of trusted components with vote-share normalization by race,
  component-disagreement tracking, rolling-origin simplex weight learning when the
  backtest is trustworthy, calibrated marginal winner probabilities, and a persisted
  recalibration map when a latest rolling-origin calibration artifact is available.
  If a learned trusted component has no current estimates for the forecast scope, the
  run records a runtime admission fallback and uses the first available component in
  polling/fundamentals/markets/public-signals order rather than publishing an all-null
  forecast.
- Simulation: structured-factor election-error draws. When a residual covariance artifact
  is available, that covariance replaces the configured national/region/office layers;
  otherwise the engine falls back to national, region, and office factors plus
  heavy-tailed local error. Bayesian posterior draws seed the race-level simulation
  center for Bayesian races, but the simulator still applies national, region, office,
  and heavy-tailed local forecast-error layers. Race-level winners, vote shares, and
  turnout are always emitted; thresholded control outcomes are emitted only for races with a
  configured `control_body`, so non-control tracker rows can participate in posterior
  and cross-office artifacts without changing seat-count math.
- Daily update: `forecast update --from-anchor <run_id> --as-of <date>` appends
  posterior summaries from a Bayesian anchor run, writes update diagnostics, and refreshes
  `R15_daily_update_quality`. The current implemented strategy is the Phase 0b-selected
  cached-posterior reweighting/summary path with full-refit fallback semantics; full SVI
  or SMC strategies remain gated until Phase 0b accepts them for the target scope.
- NUTS failover: `bayesian.nuts.wall_clock_timeout_seconds` and
  `bayesian.nuts.failover.fallback_order` define production timeout semantics. Phase 8
  exercises this policy on a fixture timeout and records the audit separately from the
  forecast-level `fallback_used` field.
- Performance: two-option race draw generation uses a Numba parallel kernel with a Python
  fallback, while table transforms should stay vectorized through Polars/DuckDB.
- Live-source readiness: Phase 8 records live 2026 scope from the actual curated source
  manifest and curated tables. It may only report `claimed` when successful non-file
  sources contribute model-bearing target-year rows for every expected verification
  office. Neutral race-presence or other metadata-only rows must be reported separately
  as `metadata_only` and must not unlock a production-default switch.
- Production promotion: the Bayesian path is the production default in config for
  operational forecasts, and the plan is accepted as production-promoted only when the
  broader rolling-origin readiness comparison shows Bayes/NUTS beating the legacy
  Kalman scorecard without interval-coverage degradation.

Planned statistical upgrade path:

- Replace the opt-in analytic Bayesian bridge with the full hierarchical NumPyro/NUTS
  state-space model specified in `plan/04-bayesian-polling-model.md`.
- Run Phase 0b before any production default switch: non-centered geometry stress tests
  and a SMC/SVI/reweighting daily-update bakeoff.
- Expand the Bayesian model from POTUS-style polling to Senate joint, House
  hierarchical, and cross-office midterm scopes using low-rank or sparse covariance
  structures where required by the plan. The production NUTS backend emits
  office-specific decomposition artifacts from the fitted shared state-space draw stream,
  while the analytic backend remains available as an explicitly labeled bridge. The NUTS
  backend receives the same fitted fundamentals-prior logit means as the analytic bridge,
  so Phase 2 prior construction is shared across Bayesian backends.
- Replace the current rolling-origin simplex/Platt calibration layer with a richer
  hierarchical calibration model once the historical panel is deep enough.
- Until that replacement exists, keep Platt/logit recalibration slope-bounded at
  `ensemble_learning.calibration_max_slope: 1.0` for publication so the calibration
  layer cannot sharpen probabilities from sparse historical panels.
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
- Turnout/recount-risk projections when calibrated or explicitly enabled proxy fields are
  available.
- Forecast coverage by tier.
- Electoral College distribution and representative simulation swarm for presidential
  scenarios. These two top-line presidential views should render together in the lead
  diagnostics summary, not only in the lower projection grid.
- Polling probability trajectories when rolling-origin polling probability and as-of
  cut columns are available.
- Simulation probability convergence when draw-level winner rows are available.
- MCMC-style posterior simulation chain traces for Electoral College totals.
- Kalman posterior uncertainty traces for state-space polling fits.
- Bayesian posterior latent-share intervals and posterior diagnostics when
  `--inference-engine bayes` is used.
- Fundamentals-prior interval plots when Bayesian runs emit `fundamentals_prior.parquet`.
- Daily posterior history panels once `forecast update` has produced
  `posterior_history.parquet`.

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

This benchmark is a simulation-throughput contract, not a sampler benchmark. The setup
ensemble uses deterministic Kalman polling so the reported throughput isolates
`SimulationEngine` and the configured Numba/Python draw backend even when the production
forecast default is Bayesian/NUTS.

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

The summary must report simulated/control Electoral College winner probability, EV
p10/p50/p90, simulated/control EC winner accuracy, deterministic state-topline EC winner
as an audit field, state accuracy, Brier score, vote-share MAE, upset count, missed
states, and links to each cycle's diagnostics and comparison report.

Cycle evaluation must validate all requested scenario keys and cycle-specific dates
before starting any forecast. It may support explicit artifact reuse, but reuse must be
operator-selected rather than silent.

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
uv run election-outcomes verify run --run-id smoke
uv run election-outcomes benchmark run --as-of 2026-05-08 --run-id perf
```

An explicit Bayesian smoke test is:

```bash
uv run election-outcomes forecast run \
  --as-of 2026-05-08 \
  --run-id bayes-smoke \
  --inference-engine bayes \
  --bayesian-backend nuts \
  --quiet
uv run election-outcomes forecast update --from-anchor bayes-smoke --as-of 2026-05-09
uv run election-outcomes verify run --run-id bayes-smoke
```

A Phase 0 methodology spike smoke test is:

```bash
uv run election-outcomes spike phase-0 \
  --scenario president_state \
  --holdout-cycle 2024 \
  --run-id phase0-smoke \
  --bayesian-backend nuts
```

A Phase 0b acceleration spike smoke test is:

```bash
uv run election-outcomes spike phase-0b --run-id phase0b-smoke
```

A fixture-backed Phase 8 multi-office smoke test is:

```bash
uv run election-outcomes verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-smoke \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --quiet
```

The same Phase 8 harness can exercise the compact hierarchical NumPyro/NUTS backend:

```bash
uv run election-outcomes verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-nuts-smoke \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --bayesian-backend nuts \
  --quiet
```

The Phase 4/5/7 historical calibration gate is:

```bash
uv run election-outcomes verify historical-calibration \
  --run-id midterm-2022-calibration \
  --bayesian-backend nuts \
  --quiet
```

It writes per-office Senate, House, and Governor calibration metrics plus explicit
Phase 4, Phase 5, and Phase 7 gate results. This is a compact fixture gate; production
claims still require a broader historical panel.

For production-dimension synthetic Senate and House coverage:

```bash
uv run election-outcomes verify historical-calibration \
  --run-id historical-panels-2022-nuts \
  --sources-config sources_historical_panels.yaml \
  --data-dir data/historical-panels-nuts \
  --artifacts-dir artifacts/historical-panels-nuts \
  --bayesian-backend nuts \
  --quiet
```

A production-default readiness audit is:

```bash
uv run election-outcomes verify readiness \
  --run-id bayes-default-readiness \
  --forecast-run-id phase8-smoke \
  --bayes-backtest-run-id president-state-bayes-backtest \
  --legacy-backtest-run-id president-state-backtest
```

The smoke run must create all required Parquet/JSON/HTML/Markdown/PNG artifacts, and
`verify run` must pass artifact/schema/plot checks. The reward card must record every
implemented reward state. `R0_build` is validated by the external commands above, `R1`
passes only after a same-run-id reproducibility rerun, and experimental component gates
may fail only when the failure is explicit in `reward_card.json` and `diagnostics.html`.
