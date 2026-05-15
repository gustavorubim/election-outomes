# Election Outcomes

CLI-first U.S. election forecasting engine. The package syncs public or fixture-backed
sources, builds curated race tables, runs polling/fundamentals/market/public-signal
components, simulates correlated outcomes, and writes auditable diagnostics.

The current default run is deterministic and fixture-backed so the full artifact,
plotting, reward, and benchmark contract can be tested before broader live adapters are
added. The presidential benchmark includes 50 states plus DC for 2000-2024 and supports
full Electoral College simulation.

Core project documents:

- [`SPEC.md`](SPEC.md): durable implementation contract.
- [`AGENTS.md`](AGENTS.md): required agent operating rules.
- [`docs/technical_appendix.md`](docs/technical_appendix.md): model details.
- [`docs/performance.md`](docs/performance.md): Numba and performance contract.
- [`docs/api_requirements.md`](docs/api_requirements.md): live-ingestion API notes.
- [`docs/plan_completion_audit.md`](docs/plan_completion_audit.md): current evidence
  against the plan's definition of done and remaining blockers.

## Package Initialization And Installation

Prerequisites:

- Python managed through `uv`.
- macOS/Linux shell.
- No API keys are required for default fixture-backed runs.

Install the package and local environment:

```bash
uv sync
chflags -R nohidden .venv
```

The `chflags` command is included because this macOS environment has repeatedly hidden
`.venv` metadata after package syncs, which can break editable imports.

Validate the repo:

```bash
uv run ruff check
uv run ruff format --check
PYTHONPATH=src uv run pytest --cov=src/election_outcomes --cov-fail-under=90
```

## Core Workflows

These are the commands to use most often.

### 1. Full Forecast

Run a complete forecast with diagnostics, plots, reward card, and reproducibility
fingerprint:

```bash
uv run election-outcomes forecast run \
  --as-of 2026-05-08 \
  --run-id full-forecast
```

The default polling engine is resolved from `configs/model.yaml`.
The Bayesian path is the production default in config, so operational forecasts emit
Bayesian posterior artifacts. The broad live-scope readiness gate is eligible with the
current Bayes/NUTS run: ensemble log loss `0.124907` versus legacy Kalman `0.125154`,
with 90% interval coverage `0.9662` versus legacy `0.9610`. To force the legacy
Kalman/state-space path, run:

```bash
uv run election-outcomes forecast run \
  --as-of 2026-05-08 \
  --run-id kalman-polling-smoke \
  --inference-engine kalman
```

The Bayesian backend defaults to compact hierarchical NumPyro/NUTS with two vectorized
chains, 500 warmup iterations, 2,000 sampling iterations per chain, and a `0.99` target
acceptance probability. A plain `uv sync` installs JAX, NumPyro, and ArviZ. The
deterministic analytic bridge remains available for fast smoke runs by selecting it
explicitly:

NUTS polling observations use the empirical-Bayes pollster house-effect adjustment, a
Bayesian-specific 7-day half-life, population/methodology quality weights, and
poll-age process variance before fitting the as-of latent state. Exported Bayesian
draws are inflated from `as_of` to election day with
`bayesian.state_space.forecast_drift_sd_per_sqrt_day`, then constrained within each
race so binary shares are anti-correlated and sum to one. The simulator uses those
draws as race-level centers and still applies the configured national, region, office,
and heavy-tailed local forecast-error layers.

```bash
uv run election-outcomes forecast run \
  --as-of 2026-05-08 \
  --run-id analytic-polling-smoke \
  --inference-engine bayes \
  --bayesian-backend analytic
```

If the runtime environment is missing NumPyro/JAX or NUTS exceeds the configured wall-clock timeout, the run
falls back through the configured policy and records that status in
`posterior_diagnostics.json`.

Open the main report:

```bash
open artifacts/runs/full-forecast/diagnostics.html
```

Verify a completed run's artifacts, plots, posterior schemas, and reward gates:

```bash
uv run election-outcomes verify run --run-id full-forecast
```

Run the fixture-backed Phase 8 multi-office verification scenario. This executes a
Bayesian President-tracker+Senate+House+Governor 2026 forecast twice for
reproducibility, runs the daily-update gate, verifies schemas/rewards, and writes the
visual QA checklist. The President tracker is a non-control fixture row, so it appears in
posterior/race/cross-office artifacts but does not contribute Electoral College control:

```bash
uv run election-outcomes verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-verification \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --quiet
```

To make the compact hierarchical NumPyro/NUTS backend explicit through the same Phase 8
harness, add `--bayesian-backend nuts`:

```bash
uv run election-outcomes verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-nuts-verification \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --bayesian-backend nuts \
  --quiet
```

Run a cached-posterior daily update from a Bayesian anchor run:

```bash
uv run election-outcomes forecast update \
  --from-anchor bayes-polling-smoke \
  --as-of 2026-05-09
```

The update appends posterior history, writes run-local update diagnostics, and refreshes
`reward_card.json` so `R15_daily_update_quality` reflects the latest update gate.

Audit whether the Bayesian path is eligible to become the production default:

```bash
uv run election-outcomes verify readiness \
  --run-id bayes-default-readiness \
  --forecast-run-id phase8-verification \
  --bayes-backtest-run-id president-state-bayes-backtest \
  --legacy-backtest-run-id president-state-backtest
```

This writes `artifacts/readiness/<run_id>/methodology_readiness.json` and
`methodology_readiness.md`. The audit is intentionally conservative: it blocks a
default switch unless Bayes dependencies are base dependencies, docs/config declare the
Bayesian default, Phase 8 and hard reward gates pass, live-source scope is claimed, and
rolling-origin Bayes evidence beats the legacy Kalman scorecard without degrading
interval coverage. The calibrated publication layer uses a bounded Platt/logit
transform with `ensemble_learning.calibration_max_slope: 1.0`; this keeps recalibration
from sharpening probabilities on small historical panels.
The live-source scope check inspects the run's curated source manifest and tables; it
only claims live 2026 coverage when successful non-file sources contribute
model-bearing rows for every office listed in the scenario's
`live_source_required_offices`. Neutral metadata rows, such as race-presence rows from
Wikipedia, are reported as `metadata_only` rather than treated as enough to unlock a
production default. The synthetic President tracker remains a fixture-only artifact
exercise and is not a live-source coverage requirement.

Main output:

```text
artifacts/runs/full-forecast/
  race_catalog.parquet
  race_forecasts.parquet
  forecast_draws.parquet
  control_forecasts.parquet
  ecosystem_forecasts.parquet
  source_manifest.parquet
  diagnostics.html
  reward_card.json
  methodology_snapshot.md
  model_card.md
  silver_benchmark.json
  silver_benchmark.html
  reproducibility_fingerprint.json
  performance.json
  plot_manifest.json
  poll_trajectory.parquet
  recalibration_map.parquet      # when a latest backtest calibration map is applied
  posterior_draws.parquet        # Bayesian default; absent on --inference-engine kalman
  state_space_trajectory.parquet # Bayesian default; absent on --inference-engine kalman
  pollster_house_effects.parquet # Bayesian default; absent on --inference-engine kalman
  posterior_diagnostics.json     # Bayesian default; absent on --inference-engine kalman
  fundamentals_prior.parquet     # Bayesian default; absent on --inference-engine kalman
  seat_posterior.parquet         # Bayesian default; absent on --inference-engine kalman
  senate_seat_posterior.parquet  # if senate control rows exist
  house_seat_posterior.parquet   # if house control rows exist
  governor_seat_posterior.parquet # if governor control rows exist
  senate_joint_posterior.parquet # if Senate posterior rows exist
  house_hierarchical_posterior.parquet # if House posterior rows exist
  cross_office_posterior.parquet # if multiple midterm offices share draws
  inference.log                  # Bayesian default; absent on --inference-engine kalman
  inference.html                 # Bayesian default; absent on --inference-engine kalman
  posterior_history.parquet      # after forecast update
  latest_daily_update.json       # after forecast update
  updates/<as-of>/               # after forecast update
  timeout_failover_audit.json    # after Phase 8 verification
  phase8_verification.json       # after verify run --scenario ...
  visual_qa_checklist.json       # after verify run --scenario ...
  verification.json              # after verify run
  stability_metrics.json
  plots/
```

Readiness output:

```text
artifacts/readiness/bayes-default-readiness/
  methodology_readiness.json
  methodology_readiness.md
```

`posterior_draws.parquet` uses the same `model_config_hash` and
`source_manifest_hash` lineage columns as `race_forecasts.parquet`. The bridge is
deterministic for a fixed bundle, config, seed, and `as_of` date. Bayes runs also
write `fundamentals_prior.parquet`, a CV-ridge or fallback Election-Day prior used by
the polling posterior. The analytic bridge and NumPyro/NUTS backend both initialize
their race-option logits from this artifact. Candidate offices with no eligible polls
can still receive prior-only posterior draws from that fundamentals prior so sparse
House/Senate races leave an auditable uncertainty artifact instead of disappearing from
the Bayesian state; these prior-only posterior summaries can also enter the polling
component for sparse forecast rows. Posterior draws are election-day latent centers;
`forecast_draws` adds the full simulation uncertainty stack before probabilities are
published. The legacy Kalman path remains available with
`--inference-engine kalman`.

If learned component admission trusts a component that has no current rows for the
requested scenario, the forecast records `component_admission_runtime_fallback` in the
model config and uses the first available component in polling/fundamentals/markets/
public-signals order instead of publishing an all-null forecast.

Bayesian runs now also write office-methodology artifacts when relevant. When the
upstream posterior is fitted by `--bayesian-backend nuts`, these artifacts are
NUTS-backed decompositions of the shared fitted state-space draw stream; when
`--bayesian-backend analytic` is selected they are labeled as analytic bridge outputs.

- `senate_joint_posterior.parquet`: Phase 4 Senate shared-environment decomposition
  with class effect, state deviation, and holdover-aware seat posterior summaries.
- `house_hierarchical_posterior.parquet`: Phase 5 House hierarchy decomposition with
  redistricting-era partitioning, state effects, district residuals, sparse-district
  flags, and a non-dense covariance method.
- `cross_office_posterior.parquet`: Phase 7 shared midterm draw stream with national
  environment and per-office offsets for Senate/House/Governor offices present in the
  run.

Bayesian runs use Rich progress reporting for the fundamentals-prior and posterior
fit phases. `--quiet` suppresses terminal rendering while still writing
`inference.log` and `inference.html` for run-local inspection.

When a trusted rolling-origin backtest has produced a latest calibration artifact,
forecast runs also copy `recalibration_map.parquet` into the run directory. The reward
card's `R14_calibrated_publication` gate records whether published probabilities used
that persisted map or were already within the configured calibration tolerance.

### 2. Full Backtest

Run the rolling-origin scorecard, component admission, learned ensemble calibration, and
residual covariance pass:

```bash
uv run election-outcomes backtest run \
  --scenario president_state \
  --run-id president-state-backtest
```

The same rolling-origin harness can score the Bayesian polling path explicitly:

```bash
uv run election-outcomes backtest run \
  --scenario president_state \
  --holdout-cycle 2024 \
  --run-id president-state-bayes-backtest \
  --inference-engine bayes \
  --bayesian-backend nuts
```

Use `--bayesian-backend analytic` for a fast deterministic bridge run when you are
debugging scorecard plumbing rather than exercising the production NUTS backend.

Write scheduled hyperprior refresh candidates without changing the production `latest`
artifacts:

```bash
uv run election-outcomes backtest refresh-hyperpriors \
  --run-id monthly-hyperpriors \
  --scenarios president_state \
  --inference-engine bayes \
  --bayesian-backend nuts
```

This writes `artifacts/hyperprior_refreshes/<run_id>/hyperprior_refresh_manifest.json`,
scenario-local candidate hyperpriors, and `comparison_report.md`. The refresh command is
deliberately non-promoting; production forecasts continue reading
`artifacts/backtests/latest/` until a separate explicit review promotes a candidate.

Run the Phase 0 side-by-side methodology spike:

```bash
uv run election-outcomes spike phase-0 \
  --scenario president_state \
  --holdout-cycle 2024 \
  --run-id phase0-potus-2024 \
  --bayesian-backend nuts
```

Run the Phase 0b geometry and daily-update acceleration spike:

```bash
uv run election-outcomes spike phase-0b --run-id phase0b-acceleration
```

Run the compact 2022 Senate/House/Governor historical calibration gate for the
Phase 4/5/7 office-methodology plan:

```bash
uv run election-outcomes verify historical-calibration \
  --run-id midterm-2022-calibration \
  --bayesian-backend nuts \
  --quiet
```

This writes `artifacts/historical_calibration/<run_id>/historical_calibration.json`,
`office_calibration.parquet`, `historical_calibration_comparison.parquet`, and a
Markdown summary. The fixture proves the calibration gate is executable across Senate,
House, and Governor; it does not replace a production-sized historical panel.

For a production-dimension synthetic congressional panel, switch the source registry:

```bash
uv run election-outcomes verify historical-calibration \
  --run-id historical-panels-2022-nuts \
  --sources-config sources_historical_panels.yaml \
  --data-dir data/historical-panels-nuts \
  --artifacts-dir artifacts/historical-panels-nuts \
  --bayesian-backend nuts \
  --quiet
```

That registry loads full Senate and House fixture panels for 2014-2026 while keeping
the default source registry compact for routine tests.

Backtest output:

```text
artifacts/backtests/president-state-backtest/
  scorecard.json
  scorecard.parquet
  rolling_predictions.parquet
  component_admission.json
  ensemble_learning.json
  probability_calibration.json
  recalibration_map.parquet
  bayesian_hyperpriors.json
  residual_covariance.parquet
```

Phase 0 spike output:

```text
artifacts/spikes/phase0-potus-2024/
  comparison.json
  phase0_comparison.parquet
  rolling_predictions_kalman.parquet
  rolling_predictions_bayes.parquet
  scorecard_kalman.json
  scorecard_bayes.json
```

Phase 0b spike output:

```text
artifacts/spikes/phase0b-acceleration/
  phase0b_summary.json
  geometry_comparison.parquet
  acceleration_bakeoff.parquet
```

The presidential-state benchmark evaluates multiple pre-election cuts where data exists:
`T-90`, `T-60`, `T-30`, `T-7`, and `T-1`. When the row-count gate passes, the latest
backtest also writes learned non-negative ensemble weights and a bounded Platt/logit
calibration transform under `artifacts/backtests/latest/`; forecast runs consume those
artifacts before publishing marginal race probabilities. The same backtest pass also
writes `bayesian_hyperpriors.json`, a rolling-origin grid search used by Bayes
runs to set the Election-Day extra-variance and fundamentals-prior strength. The Phase
0b spike records the non-centered parameterization gate and rejects global SMC for the
combined dimensionality ladder unless its ESS/drift thresholds pass; the current
configured daily-update strategy remains cached posterior reweighting with full-refit
fallback semantics. Bayesian NUTS configuration includes a wall-clock timeout and an
ordered fallback policy (`previous_posterior_reuse`, `bayes_svi_fallback`,
`kalman_legacy_fallback`); Phase 8 writes `timeout_failover_audit.json` from a forced
fixture timeout so the policy is exercised without marking the forecast itself as a
fallback run.

### 3. Full Cycle Analysis

Run the same-date historical presidential benchmark across cycles:

```bash
PYTHONPATH=src uv run election-outcomes results cycle-eval \
  --run-id oct5-presidential-cycle-eval \
  --cycles 2008,2012,2016,2020,2024 \
  --as-of-mm-dd 10-05 \
  --data-dir data/cycle-eval \
  --artifacts-dir artifacts/cycle-eval
```

Open the cycle dashboard:

```bash
open artifacts/cycle-eval/cycle_evals/oct5-presidential-cycle-eval/cycle_eval.html
```

Cycle-eval output:

```text
artifacts/cycle-eval/cycle_evals/oct5-presidential-cycle-eval/
  cycle_summary.parquet
  cycle_summary.json
  cycle_eval.html
  narrative.md
  plots/
```

The cycle dashboard reports simulated/control Electoral College winner accuracy, state
accuracy, Brier score, vote-share error, upsets, missed states, and links to each
cycle's diagnostics and comparison report. It also retains the deterministic
state-topline EC winner from `results compare` as an audit field.

### Current Historical Accuracy Snapshot

The latest checked-in workflow artifacts report the following same-date historical
accuracy. Treat race/state winner accuracy and chamber-control accuracy as separate
claims: House district calls are mostly safe seats, while House chamber control remains
fragile in the current fixture-backed panel.

| Scope | Historical cycles | Race/state winner accuracy | Topline winner accuracy | Mean Brier | Vote-share MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Presidential | 2008, 2012, 2016, 2020, 2024 | 90.6% | 100.0% EC winner | 0.0741 | 2.74 pts |
| Senate | 2014, 2016, 2018, 2020, 2022, 2024 | 98.5% | 100.0% chamber winner | 0.0262 | 1.50 pts |
| House | 2014, 2016, 2018, 2020, 2022, 2024 | 98.4% | 66.7% chamber winner | 0.0239 | 2.68 pts |

Interpretation: the current system is a strong race-level classifier on these
deterministic panels and has reasonable Brier scores, but the House aggregate control
call should not be treated as highly reliable until more real-data validation and
cross-cycle stress testing are added.

## 2024 Presidential Benchmark

Run the 2024 presidential scenario at the default pre-election date:

```bash
uv run election-outcomes forecast run \
  --scenario president_2024_state \
  --run-id 2024-presidential
```

Compare against actual 2024 presidential results:

```bash
uv run election-outcomes results compare \
  --forecast-run-id 2024-presidential \
  --comparison-id 2024-presidential-actuals \
  --cycle 2024 \
  --office-type president
```

Open the comparison report:

```bash
open artifacts/runs/2024-presidential/comparisons/2024-presidential-actuals/result_comparison.html
```

The comparison report opens with a KPI strip, fixed-size calibration plots, an
actual-winner probability histogram, an actual-winner probability swarm, compact
largest-miss rows, and collapsed audit tables for the full summary JSON and
option-level comparison rows. The same layout is used for president, Senate, and
House comparisons so large chamber dashboards do not create oversized
race-by-race canvases.

Use this benchmark to inspect misses and calibration. Do not tune directly against 2024
actuals; use cross-cycle evidence and rolling-origin backtests.

## Senate And House Analysis

The engine ships parallel scenarios for U.S. Senate (state-level) and U.S. House
(district-level) races with their own panels, parsers, and source registries. Reports
lead with the **majority story**: each chamber has its own configured threshold (Senate
51, House 218) and the control forecast surfaces holdover seats, modeled seats,
seat-count distribution, and majority probability per party.

### Senate Cycle Analysis

Run rolling-origin same-date evaluation across Senate Class I/II/III rotations from
2014–2024:

```bash
uv run election-outcomes results cycle-eval \
  --cycles 2014,2016,2018,2020,2022,2024 \
  --as-of-mm-dd 11-04 \
  --scenario-template "senate_{cycle}_state" \
  --forecast-run-prefix sen-eval \
  --office-type senate \
  --sources-config sources_senate.yaml \
  --data-dir data/senate \
  --artifacts-dir artifacts/senate \
  --run-id senate-cycles-2014-2024
```

Open the dashboard:

```bash
open artifacts/senate/cycle_evals/senate-cycles-2014-2024/cycle_eval.html
```

Each per-cycle forecast lives at
`artifacts/senate/runs/sen-eval-<cycle>-1104/` with the full diagnostics, model card,
silver benchmark, control forecast, and forecast-vs-actual comparison.

### House Cycle Analysis

Same-date evaluation across the 6 most recent House cycles:

```bash
uv run election-outcomes results cycle-eval \
  --cycles 2014,2016,2018,2020,2022,2024 \
  --as-of-mm-dd 11-04 \
  --scenario-template "house_{cycle}_district" \
  --forecast-run-prefix hou-eval \
  --office-type house \
  --sources-config sources_house.yaml \
  --data-dir data/house \
  --artifacts-dir artifacts/house \
  --run-id house-cycles-2014-2024
```

Open the dashboard:

```bash
open artifacts/house/cycle_evals/house-cycles-2014-2024/cycle_eval.html
```

The House panel covers all 435 districts × 6 cycles spanning two redistricting eras
(`2012_2020` and `2022_plus`). Polls are restricted to competitive districts; safe
seats are forecast through fundamentals only.

### Defining Majority In Reports

`control_forecasts.parquet` for each Senate/House run carries:

- `control_threshold` — 51 for Senate, 218 for House (from `configs/model.yaml`).
- `holdover_seats` — Senate seats not up that cycle, sourced from the scenario.
- `modeled_seats` — number of seats actually being contested.
- `seat_count_modeled_mean` — mean seats won in modeled races (across draws).
- `seat_count_mean`, `seat_count_p10/p50/p90` — total post-cycle seats including
  holdovers, with 80% interval.
- `majority_probability` — `P(seat_count >= control_threshold)`.
- `seats_to_majority_mean` — seats short of majority on average.
- `tipping_point_races`, `pivotal_rates` — most pivotal contests for control.

`cycle_eval.html` and `narrative.md` lead with chamber name + threshold, then per-cycle
DEM/REP majority probabilities, mean seat counts, race accuracy, and missed
states/districts.

### Source Registries For Each Chamber

```text
configs/sources.yaml         # Presidential state-panel + fixture defaults.
configs/sources_senate.yaml  # Senate state-panel (Class rotation), 6 cycles.
configs/sources_house.yaml   # House district-panel, 6 cycles, 2 redistricting eras.
configs/sources_live.yaml    # Live 538 polling and Wikipedia metadata overlay.
```

Each registry supports `extends:` to layer real-data adapters on top of the panel
fixture without duplicating non-conflicting entries.

### Scenarios

```text
president_state, president_2000_state ... president_2024_state
senate_state,    senate_2014_state    ... senate_2024_state, senate_2026_state
house_district,  house_2014_district  ... house_2024_district, house_2026_district
```

Senate scenarios declare `senate_class` (I/II/III) and `holdovers` (DEM/REP/IND seat
counts not up that cycle). House scenarios declare `redistricting_era` so rolling
origin training stays within a comparable boundary regime.

### Synthetic Panel Honesty

The 2014–2024 Senate and House panels are deterministic procedural draws that match
aggregate historical patterns (D/R seat swings, Cook PVI distribution, incumbency
advantage). They are not actual historical results. The harness exists to exercise
the rolling-origin and majority-threshold contracts at scale; production runs should
ingest real returns from MIT Election Lab + 538 senate/house polls. Regenerate panels
with:

```bash
uv run python scripts/generate_senate_panel.py
uv run python scripts/generate_house_panel.py
```

## 2026 Midterm Forecast

The 2026 Senate (Class II rotation) and 2026 House midterm have forecast-only entries
in both panels — fundamentals and polling rows are populated, results columns are
deliberately empty until the election happens.

### Senate 2026

```bash
uv run election-outcomes forecast run \
  --as-of 2026-11-02 \
  --run-id senate-2026-midterm \
  --scenario senate_2026_state \
  --sources-config sources_senate.yaml \
  --data-dir data/senate \
  --artifacts-dir artifacts/senate-2026
```

### House 2026

```bash
uv run election-outcomes forecast run \
  --as-of 2026-11-02 \
  --run-id house-2026-midterm \
  --scenario house_2026_district \
  --sources-config sources_house.yaml \
  --data-dir data/house \
  --artifacts-dir artifacts/house-2026
```

### Inspect The Midterm Outputs

```bash
open artifacts/senate-2026/runs/senate-2026-midterm/diagnostics.html
open artifacts/house-2026/runs/house-2026-midterm/diagnostics.html

# Majority probability summary
uv run python -c "
import polars as pl
for chamber, path in [
    ('SENATE', 'artifacts/senate-2026/runs/senate-2026-midterm/control_forecasts.parquet'),
    ('HOUSE',  'artifacts/house-2026/runs/house-2026-midterm/control_forecasts.parquet'),
]:
    print(chamber)
    print(pl.read_parquet(path).select([
        'party','control_threshold','holdover_seats','modeled_seats',
        'seat_count_mean','seat_count_p10','seat_count_p90',
        'majority_probability','seats_to_majority_mean',
    ]))
"
```

### Forecast-Only Cycle Caveats

- `compare_results` against 2026 returns empty (no actuals yet). R5/R6/R8 still gate
  on the rolling-origin backtest of 2014–2024.
- The 2026 environment seed (D+3.0 for Senate, D+4.0 for House) is a midterm
  out-party assumption baked into `scripts/generate_*_panel.py`. Replace it with real
  poll-aggregate values via `sources_live.yaml` once upstream 538 or other live poll
  streams contribute 2026 model-bearing rows.
- House 2026 inherits the `2022_plus` redistricting era, so rolling-origin training
  uses 2022 + 2024 boundaries.

## What To Inspect

Important forecast artifacts:

- `diagnostics.html`: executive forecast dashboard with an office-aware headline
  (Electoral College for president, chamber control for Senate/House), KPI strip,
  fixed-size overview plots, model drivers, trust gates, backtest snapshot,
  Silver/FiveThirtyEight methodology benchmark, and compact plot grids.
- `race_forecasts.parquet`: per-option probabilities, vote-share intervals, drivers,
  raw and calibrated per-option probabilities, vote-share intervals, drivers, data-quality
  flags, and lineage hashes.
- `forecast_draws.parquet`: race-level simulation draws; Bayesian runs seed the draw
  centers from `posterior_draws.parquet` and still add systematic and heavy-tailed
  forecast errors.
- `recalibration_map.parquet`: persisted Platt/logit recalibration map copied from the
  latest trusted backtest when applied to published probabilities.
- `posterior_draws.parquet`: race-constrained Bayesian election-day latent-share draws
  unless the run forces `--inference-engine kalman`.
- `state_space_trajectory.parquet`: Bayesian trajectory summaries by
  race, option, and day, with the same lineage hashes as forecast rows.
- `pollster_house_effects.parquet`: empirical-Bayes pollster house-effect estimates
  used by the Bayesian polling bridge.
- `posterior_diagnostics.json`: Bayesian polling diagnostics, draw count,
  parameterization, and lineage hashes.
- `fundamentals_prior.parquet`: Bayesian Election-Day prior from the trained
  fundamentals model, including prior method and logit-scale uncertainty.
- `seat_posterior.parquet`: Bayesian draw-level seat/control counts. When
  Senate or House control rows exist, office-specific posterior files are also written.
- `posterior_history.parquet`: append/update history of daily posterior summaries
  created by `forecast update`.
- `senate_joint_posterior.parquet`: opt-in Senate shared-environment artifact with
  Senate class effect and state deviation decomposition.
- `house_hierarchical_posterior.parquet`: opt-in House hierarchy artifact with
  redistricting-era partition, state effect, district idiosyncrasy, and sparse-district
  indicators.
- `cross_office_posterior.parquet`: opt-in cross-office shared midterm environment
  artifact over the common posterior draw stream.
- `control_forecasts.parquet`: EC/control probability, EV/seat distributions, and
  pivotal/tipping information.
- `reward_card.json`: machine-readable reward checks.
- `model_card.md`: learned/configured/placeholder parameter status.

Current trust boundary:

- Default data is deterministic fixture/panel data.
- The default presidential panel is useful for development and benchmark shape, but it
  is not a reproduction of Silver Bulletin or FiveThirtyEight.
- Live polling and metadata ingestion can be enabled with `configs/sources_live.yaml`.
  The 2026 Wikipedia entries are neutral race-presence metadata, not a substitute for
  model-bearing polls, fundamentals, or market observations.
- Close-margin recount/certification fields are withheld by default because they are only
  experimental proxies. Set `experimental_outputs.include_close_margin_ecosystem: true`
  in `configs/model.yaml` only when you explicitly want those uncalibrated proxy fields.

## Repository Map

```mermaid
flowchart TD
    repo["election-outomes/"]
    config{{"configs/*.yaml"}}
    fixtures[/"fixtures/*.csv"/]
    src["src/election_outcomes/"]
    tests["tests/"]
    docs["docs/"]
    schemas["schemas/"]
    data[("data/ local lake")]
    artifacts[("artifacts/ run outputs")]

    repo --> config
    repo --> fixtures
    repo --> src
    repo --> tests
    repo --> docs
    repo --> schemas
    repo --> data
    repo --> artifacts

    src --> cli["cli.py"]
    src --> ingest["ingest/"]
    src --> normalize["normalize/"]
    src --> features["features/"]
    src --> models["models/"]
    src --> scoring["scoring/"]
    src --> reports["reports/"]
    src --> performance["performance/"]
```

## End-To-End Flow

```mermaid
flowchart LR
    start([manual run])
    sync["sync sources"]
    raw[("raw source snapshots")]
    normalize["normalize tables"]
    curated[("curated Parquet")]
    tier{"tier gate"}
    models["component models"]
    ensemble["ensemble blend"]
    sim["correlated simulation"]
    artifacts[/"forecast artifacts"/]
    diagnostics["HTML diagnostics"]
    rewards(("reward card"))
    compare["optional result comparison"]

    start --> sync --> raw --> normalize --> curated --> tier
    tier --> models --> ensemble --> sim --> artifacts
    artifacts --> diagnostics
    artifacts --> rewards
    artifacts --> compare
```

## Control Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Pipeline
    participant Ingest
    participant Features
    participant Models
    participant Reports
    participant Results

    User->>CLI: election-outcomes forecast run
    CLI->>Pipeline: run_forecast(as_of, run_id)
    Pipeline->>Ingest: SyncRunner.run()
    Ingest-->>Pipeline: source_manifest
    Pipeline->>Features: CuratedDataBuilder + FeatureBuilder
    Features-->>Pipeline: FeatureBundle + race_catalog
    Pipeline->>Models: polling/fundamentals/markets/public signals
    Models-->>Pipeline: component estimates
    Pipeline->>Models: ensemble + simulation
    Models-->>Pipeline: draws, forecasts, control, ecosystem
    Pipeline->>Reports: plots, diagnostics, methodology
    Reports-->>User: artifacts/runs/<run_id>/
    User->>CLI: election-outcomes results compare
    CLI->>Results: join forecast artifacts to actual results
    Results-->>User: comparisons/<comparison_id>/
```

## Model Shape

```mermaid
classDiagram
    class PollingModel {
      +Kalman state-space estimate
      +previous-share prior initialization
      +iterative house-effect shrinkage
      +content fingerprint cache
    }
    class FundamentalsModel {
      +standardized ridge fit
      +partisan lean and incumbency
      +economy, finance, demographics
    }
    class MarketModel {
      +public market probability
      +liquidity/spread gate
      +normal probability-to-share map
    }
    class PublicSignalModel {
      +news/pageview-style signals
      +experimental admission
    }
    class EnsembleModel {
      +trusted component weights
      +component admission
    }
    class SimulationEngine {
      +learned covariance path
      +heavy-tailed residuals
      +Numba binary draw kernel
      +thresholded control outputs
    }

    PollingModel --> EnsembleModel
    FundamentalsModel --> EnsembleModel
    MarketModel --> EnsembleModel
    PublicSignalModel --> EnsembleModel
    EnsembleModel --> SimulationEngine
```

## Artifact Relationships

```mermaid
erDiagram
    SOURCE_MANIFEST ||--o{ RACE_CATALOG : "provenance"
    SOURCE_MANIFEST ||--o{ RACE_FORECASTS : "source hash"
    RACE_CATALOG ||--o{ RACE_FORECASTS : "tier and race"
    RACE_CATALOG ||--o{ FORECAST_DRAWS : "race id"
    RACE_FORECASTS ||--o{ FORECAST_DRAWS : "option id"
    FORECAST_DRAWS ||--o{ CONTROL_FORECASTS : "derived"
    FORECAST_DRAWS ||--o{ ECOSYSTEM_FORECASTS : "derived"
    REWARD_CARD ||--|| SOURCE_MANIFEST : "checks"
    REWARD_CARD ||--|| PLOT_MANIFEST : "checks"
    REWARD_CARD ||--|| REPRO_FINGERPRINT : "checks"
    PLOT_MANIFEST ||--o{ PLOTS : "indexes"
    RACE_FORECASTS ||--o{ RESULT_COMPARISON : "compared with actuals"
```

## Appendix A: Expanded Commands

### Inspect Reward Card

```bash
uv run python - <<'PY'
import json
from pathlib import Path

run = Path("artifacts/runs/full-forecast")
rewards = json.loads((run / "reward_card.json").read_text())["rewards"]
for name, payload in rewards.items():
    print(f"{name}: {payload['passed']} | {payload['detail']}")
PY
```

### Inspect Forecast Tables

```bash
uv run python - <<'PY'
from pathlib import Path
import polars as pl

run = Path("artifacts/runs/full-forecast")
print(pl.read_parquet(run / "race_catalog.parquet").select(["race_id", "tier", "tier_reason"]))
print(
    pl.read_parquet(run / "race_forecasts.parquet")
    .select(["race_id", "option_id", "tier", "winner_probability", "data_quality_flags"])
    .sort(["race_id", "option_id"])
)
PY
```

### Inspect Backtest Metrics

```bash
uv run python - <<'PY'
import json
from pathlib import Path

scorecard = json.loads(
    Path("artifacts/backtests/president-state-backtest/scorecard.json").read_text()
)
print(json.dumps(scorecard["metrics"], indent=2, sort_keys=True))
print(json.dumps(scorecard["ablations"], indent=2, sort_keys=True))
PY
```

### One-Month-Before 2024 Scenario

```bash
PYTHONPATH=src uv run election-outcomes forecast run \
  --scenario president_2024_state \
  --as-of 2024-10-05 \
  --run-id 2024-presidential-1mo \
  --data-dir data/run-2024-presidential-1mo \
  --artifacts-dir artifacts/run-2024-presidential-1mo
open artifacts/run-2024-presidential-1mo/runs/2024-presidential-1mo/diagnostics.html
```

### Inspect 2024 Result Comparison

```bash
uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path(
        "artifacts/runs/2024-presidential/"
        "comparisons/2024-presidential-actuals/"
        "result_comparison_summary.json"
    ).read_text()
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

```bash
uv run python - <<'PY'
from pathlib import Path
import polars as pl

comparison = pl.read_parquet(
    Path(
        "artifacts/runs/2024-presidential/"
        "comparisons/2024-presidential-actuals/"
        "result_comparison.parquet"
    )
)
print(
    comparison.select(
        [
            "race_id",
            "option_id",
            "winner_probability",
            "vote_share_mean",
            "actual_vote_share",
            "absolute_vote_share_error",
            "predicted_winner",
            "actual_winner",
        ]
    )
)
PY
```

### Reuse Existing Cycle-Eval Artifacts

```bash
PYTHONPATH=src uv run election-outcomes results cycle-eval \
  --run-id oct5-presidential-cycle-eval \
  --cycles 2008,2012,2016,2020,2024 \
  --as-of-mm-dd 10-05 \
  --data-dir data/cycle-eval \
  --artifacts-dir artifacts/cycle-eval \
  --reuse-existing
```

### Live Source Smoke Run

The first live-ingestion path uses FiveThirtyEight's public Datasette CSV stream for
the 2020 presidential poll archive. The same live registry also includes keyless
FiveThirtyEight/Datasette Senate, Governor, and House polling streams filtered to the
2026 cycle; those streams contribute polling rows only when the upstream tables have
matching 2026 general-election polls. The live registry also includes a keyless FRED
UNRATE CSV adapter that emits model-bearing national macro fundamentals for the compact
2026 Senate/Governor/House smoke races, plus keyless Wikipedia raw-page race-presence
metadata. The Wikipedia rows are neutral `public_signals` with `z_score = 0.0`; they
prove public HTTP text ingestion and provenance, but they do not influence the Bayesian
polling latent state by themselves. The live registry does not need Google Civic.

```bash
uv run election-outcomes forecast run \
  --sources-config sources_live.yaml \
  --data-dir data/live \
  --artifacts-dir artifacts/live \
  --as-of 2020-10-30 \
  --run-id wi-2020-live-polls
```

```bash
uv run election-outcomes results compare \
  --sources-config sources_live.yaml \
  --data-dir data/live \
  --artifacts-dir artifacts/live \
  --forecast-run-id wi-2020-live-polls \
  --comparison-id wi-2020-live-polls-actuals \
  --cycle 2020 \
  --office-type president \
  --race-id US-PRES-WI-2020
```

To probe whether the public 2026 Senate/Governor/House streams currently satisfy the
Phase 8 live-source scope gate, run the verification scenario against the live registry:

```bash
uv run election-outcomes verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-live-scope-analytic \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --bayesian-backend analytic \
  --sources-config sources_live.yaml \
  --data-dir data/live-scope \
  --artifacts-dir artifacts/live-scope \
  --quiet
```

Inspect `artifacts/live-scope/runs/<run_id>/phase8_verification.json`. A production
default switch still requires `fixture_scope.live_source_scope.status == "claimed"`.
The FRED fundamentals adapter can satisfy that live-source scope for the compact smoke
races; neutral Wikipedia `public_signals` rows alone cannot.
If an HTTP refresh stalls after a prior successful sync, the source manifest records
`status = stale_reused` and uses the cached raw snapshot rather than blocking the
verification run indefinitely.

### Performance Benchmark

```bash
uv run election-outcomes benchmark run \
  --as-of 2026-05-08 \
  --run-id full-perf
```

Benchmark output:

```text
artifacts/benchmarks/full-perf/performance_benchmark.json
```

The benchmark isolates simulation throughput. It uses the deterministic Kalman polling
path to build the setup ensemble, then times repeated `SimulationEngine` draws with the
configured performance backend.

## Appendix B: Diagnostics And Plots

Every forecast writes `plot_manifest.json` and PNG plots under `plots/`.

Projection plots:

- `race_probability_bars.png`
- `vote_share_intervals.png`
- `control_projection.png`
- `turnout_recount_risk.png` (only when close-margin ecosystem proxies are enabled)
- `tier_coverage.png`
- `electoral_college_distribution.png`
- `topline_electoral_swarm.png`

Calibration and model-quality plots:

- `calibration_curve.png`
- `brier_by_component.png`
- `interval_coverage.png`
- `polling_kalman_trajectories.png`
- `polling_probability_trajectory.png`
- `simulation_probability_convergence.png`
- `electoral_college_chain_traces.png`
- `kalman_posterior_uncertainty.png`
- `silver_methodology_benchmark.png`

List plot outputs:

```bash
find artifacts/runs/full-forecast/plots -maxdepth 1 -type f | sort
```

View the plot manifest:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("artifacts/runs/full-forecast/plot_manifest.json").read_text()
)
print(json.dumps(manifest, indent=2))
PY
```

## Appendix C: CLI Reference

- `sync`: snapshot configured fixture or HTTP CSV sources into the local raw lake.
- `build-features`: normalize raw snapshots into curated Parquet tables and race tiers.
- `forecast run`: refresh data, rebuild features, run models, simulate outcomes, and
  emit artifacts. The inference engine defaults to `configs/model.yaml`; use
  `--inference-engine kalman` to force the legacy path. Use `--bayesian-backend analytic`
  for the deterministic bridge or `--bayesian-backend nuts` for the production default
  compact hierarchical NumPyro/NUTS backend. Use `--quiet` to suppress the completion
  message.
- `forecast update`: run the configured daily-update strategy from a Bayesian anchor
  run and append `posterior_history.parquet`.
- `backtest run`: refit components by rolling-origin cycle, score baselines, learn
  simplex-constrained ensemble weights, fit probability calibration, and refresh latest
  admission/covariance artifacts. The inference engine defaults to `configs/model.yaml`;
  use `--inference-engine bayes` or `--inference-engine kalman` to override it in the
  same historical harness. Use `--bayesian-backend analytic` or `--bayesian-backend nuts`
  to choose the Bayesian backend for Bayes folds.
- `backtest refresh-hyperpriors`: run the configured scheduled hyperprior refresh and
  write candidate artifacts plus a comparison report under `artifacts/hyperprior_refreshes/`
  without promoting `artifacts/backtests/latest/`. It accepts the same
  `--bayesian-backend` override for candidate backtests.
- `report build`: rebuild diagnostics and methodology files for an existing run. It uses
  the fast legacy Kalman rolling-origin baseline for report-only benchmark context; run
  `backtest run --inference-engine bayes --bayesian-backend nuts` explicitly when you
  need Bayesian training evidence.
- `verify run`: verify required run artifacts, plot files, posterior schemas, and
  reward gates, then write `verification.json` in the run directory. With
  `--scenario 2026-multioffice-verification`, it orchestrates the fixture-backed Phase 8
  Senate+House+Governor verification run and writes `phase8_verification.json` plus
  `visual_qa_checklist.json`. Use `--bayesian-backend nuts` to route Phase 8 through
  the compact hierarchical NumPyro/NUTS backend. The Silver benchmark is scoped to the
  configured run: a compact multi-office Phase 8 run can score `production` for its
  configured scope without claiming nationwide live-source coverage.
- `verify readiness`: audit the default-switch contract and write
  `artifacts/readiness/<run_id>/methodology_readiness.json` plus a Markdown report.
- `verify historical-calibration`: run the compact 2022 Senate/House/Governor
  historical calibration audit, write `artifacts/historical_calibration/<run_id>/`,
  and gate Phase 4 Senate ECE, Phase 5 House ECE, and Phase 7 cross-office per-office
  ECE. It defaults to `--bayesian-backend nuts`; use `analytic` only for fast plumbing
  checks. Use `--sources-config sources_historical_panels.yaml` for the
  production-dimension synthetic Senate/House panel.
- `spike phase-0`: run the Kalman-vs-Bayes rolling-origin comparison harness and
  write the Phase 0 go/no-go artifact under `artifacts/spikes/`. Use
  `--bayesian-backend` to choose the Bayes comparison backend.
- `spike phase-0b`: run centered-vs-non-centered geometry checks and the
  SMC/SVI/reweighting dimensionality bakeoff, then write the Phase 0b strategy artifact.
- `benchmark run`: measure simulation throughput.
- `results compare`: compare one forecast run against known actual results.
- `results cycle-eval`: run same-date historical forecast-vs-actual analysis. Generated
  forecast runs accept `--inference-engine` and `--bayesian-backend` overrides.

## Appendix D: Trust, Rewards, And API Credentials

Reward interpretation:

- `R1_reproducibility` writes a stable artifact fingerprint on every run, but only passes
  after rerunning the same `run_id` with unchanged inputs and matching the previous
  fingerprint.
- `R5_baseline_competition`, `R6_component_admission`, and `R8_uncertainty_quality`
  require enough historical rows and should stay honest when sample sizes are weak.
  `R6` checks the components admitted as trusted by the current rolling-origin evidence;
  components rejected by admission can still be shown as diagnostics or used as priors,
  but they do not get trusted ensemble weight.
- `R2_provenance` is row-level: every forecast row must carry a model-config hash and
  source-manifest hash.

No API credentials are needed to run default fixture-backed forecasts, backtests, plots,
diagnostics, comparisons, or benchmarks. The live sources in `configs/sources_live.yaml`
also run keylessly through public FiveThirtyEight/Datasette CSV streams, FRED CSV
downloads, and Wikipedia raw-page HTTP text reads.

Google Civic is optional. The current live path does not use `GOOGLE_CIVIC_API_KEY`, and
Civic should not block polling, fundamentals, market, or historical-result ingestion.

Current key names expected by the credential template:

```bash
awk -F= '/^[A-Za-z_][A-Za-z0-9_]*=/ {print $1}' .env.example
```

Remaining live-adapter implementation order:

1. Historical results: MIT Election Lab or official state/federal archives.
2. Fundamentals: Census, FRED, BEA/BLS.
3. Polls: public poll feeds and pollster metadata.
4. Markets: read-only Kalshi/Polymarket public data, with no trading credentials.
5. Public signals: GDELT and Wikimedia/pageview-style public attention features.
