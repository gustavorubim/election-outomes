# Election Outcomes

Research-grade scaffold for a U.S.-only election forecasting engine. The package is
CLI-first: it incrementally syncs public data, builds a curated race catalog, runs a
hybrid forecast ensemble, emits auditable artifacts, generates calibration/projection
plots, and reports verifiable rewards.

The current implementation is fixture-backed so the full artifact, model, plotting,
performance, and reward contract can be tested deterministically before live public-data
adapters are added.

Canonical project documents:

- [`SPEC.md`](SPEC.md): durable implementation contract.
- [`AGENTS.md`](AGENTS.md): required agent operating rules.
- [`docs/technical_appendix.md`](docs/technical_appendix.md): detailed model and
  statistical approach.
- [`docs/performance.md`](docs/performance.md): Numba/benchmark performance contract.
- [`docs/api_requirements.md`](docs/api_requirements.md): live-ingestion API notes.

## Full Run

Use this when you want the richest current forecast output: refreshed source snapshots,
curated features, forecasts, posterior-style simulation draws, reward card, diagnostics,
plots, and performance metadata.

```bash
uv sync
chflags -R nohidden .venv
uv run election-outcomes forecast run --as-of 2026-05-08 --run-id full-forecast
```

The `chflags` command is included because this macOS environment has repeatedly hidden
`.venv` metadata after package syncs, which can prevent editable imports from loading.

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
  reproducibility_fingerprint.json
  performance.json
  plot_manifest.json
  plots/
```

Open the run report:

```bash
open artifacts/runs/full-forecast/diagnostics.html
```

Inspect the reward card:

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

Current fixture-backed reward interpretation:

- `R1_reproducibility` writes a stable artifact fingerprint on every run, but only passes
  after rerunning the same `run_id` with unchanged inputs and matching the previous
  fingerprint.
- `R5_baseline_competition`, `R6_component_admission`, and `R8_uncertainty_quality` are
  intentionally not certified from the tiny fixture scorecard. They require a real
  rolling-origin backtest with enough historical races.
- `R2_provenance` is row-level: every forecast row must carry a model-config hash and
  source-manifest hash, and the manifest must contain non-empty source hashes.

Inspect the forecast tables:

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

## Full Backtesting

Run the backtest scorecard and ablation report:

```bash
uv run election-outcomes backtest run --run-id full-backtest
```

Backtest output:

```text
artifacts/backtests/full-backtest/
  scorecard.json
  scorecard.parquet
```

Inspect backtest metrics:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

scorecard = json.loads(
    Path("artifacts/backtests/full-backtest/scorecard.json").read_text()
)
print(json.dumps(scorecard["metrics"], indent=2, sort_keys=True))
print(json.dumps(scorecard["ablations"], indent=2, sort_keys=True))
PY
```

The forecast run also embeds backtest diagnostics in `diagnostics.html` and regenerates
calibration plots from the current backtest fixture. For the richest diagnostic bundle,
run both commands:

```bash
uv run election-outcomes forecast run --as-of 2026-05-08 --run-id full-diagnostic
uv run election-outcomes backtest run --run-id full-diagnostic-backtest
open artifacts/runs/full-diagnostic/diagnostics.html
```

The current `backtest run` command is a fixture scorecard, not a production
rolling-origin fit/evaluate loop. It is useful for exercising metrics, plots, and
artifact contracts; it does not certify model trust rewards until the historical race
store is expanded.

## 2024 Presidential Historical Comparison

Use this workflow when you want to run a pre-election 2024 presidential forecast and
compare the forecast against known actual outcomes. In the current fixture-backed repo,
the available 2024 presidential example is the Wisconsin presidential race
`US-PRES-WI-2024`; live adapters will later expand this to all presidential states and
Electoral College aggregation.

Run the forecast as if it were October 1, 2024:

```bash
uv sync
chflags -R nohidden .venv
uv run election-outcomes forecast run --as-of 2024-10-01 --run-id 2024-presidential
```

Compare the forecast run with actual 2024 presidential results:

```bash
uv run election-outcomes results compare \
  --forecast-run-id 2024-presidential \
  --comparison-id 2024-presidential-actuals \
  --cycle 2024 \
  --office-type president
```

Comparison output:

```text
artifacts/runs/2024-presidential/comparisons/2024-presidential-actuals/
  result_comparison.parquet
  result_comparison_summary.json
  result_comparison.html
  plots/
    vote_share_forecast_vs_actual.png
    winner_probability_vs_actual.png
```

Open the comparison report:

```bash
open artifacts/runs/2024-presidential/comparisons/2024-presidential-actuals/result_comparison.html
```

Inspect the summary metrics:

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

Inspect the row-level comparison:

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

The key interpretation fields are:

- `winner_accuracy`: whether the forecast's top option matched the actual winner.
- `mean_absolute_vote_share_error`: average absolute forecast-share error.
- `brier_score`: probability score against actual winner indicators.
- `upset_count`: number of actual winners assigned less than 50% probability.

## First Live Poll Run

The first live-ingestion path uses FiveThirtyEight's public Datasette CSV stream for
the 2020 presidential poll archive. It does not need Google Civic. The live registry is
kept separate from the deterministic fixture registry so normal tests do not hit the
network.

Run a Wisconsin 2020 presidential forecast with live-downloaded 538 polls:

```bash
uv sync
chflags -R nohidden .venv
uv run election-outcomes forecast run \
  --sources-config sources_live.yaml \
  --data-dir data/live \
  --artifacts-dir artifacts/live \
  --as-of 2020-10-30 \
  --run-id wi-2020-live-polls
```

Rerun the same command once if you want `R1_reproducibility` to compare against the
previous stable fingerprint and pass.

Compare the forecast with actual Wisconsin 2020 results:

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

Outputs:

```text
artifacts/live/runs/wi-2020-live-polls/
artifacts/live/runs/wi-2020-live-polls/comparisons/wi-2020-live-polls-actuals/
```

Current live-source boundary:

- Live data: 538 public presidential poll CSV stream, normalized to `polls`.
- Fixture support data: race catalog, options, fundamentals, and actual result rows.
- Not live yet: Civic race catalog, FEC finance, Census/FRED/BEA/BLS fundamentals,
  public market adapters, GDELT, Wikimedia, and rolling-origin historical backtest
  snapshots.

## Plots And Diagnostics

Every forecast run writes `plot_manifest.json` plus PNG plots under `plots/`.

Calibration plots:

- `calibration_curve.png`: observed win rate versus forecast probability.
- `brier_by_component.png`: Brier score by baseline/component/ensemble.
- `interval_coverage.png`: nominal versus observed interval coverage.

Projection plots:

- `race_probability_bars.png`: winner probabilities by race and option.
- `vote_share_intervals.png`: vote-share means with interval bands.
- `control_projection.png`: modeled seat/control outcomes.
- `turnout_recount_risk.png`: recount-risk projection by race.
- `tier_coverage.png`: race coverage by Tier A/B/C.

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

## Performance Run

Binary two-option race simulation uses a Numba parallel kernel when available, with a
Python fallback. Run a benchmark after changing simulation, scoring, or forecast draw
logic:

```bash
uv run election-outcomes benchmark run --as-of 2026-05-08 --run-id full-perf
```

Benchmark output:

```text
artifacts/benchmarks/full-perf/performance_benchmark.json
```

Inspect performance metadata from a forecast:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

print(
    json.dumps(
        json.loads(Path("artifacts/runs/full-forecast/performance.json").read_text()),
        indent=2,
        sort_keys=True,
    )
)
PY
```

## Required Validation

Every change must keep the repo passing:

```bash
uv sync
chflags -R nohidden .venv
uv run ruff check
uv run ruff format --check
uv run pytest --cov=src/election_outcomes --cov-fail-under=90
```

The coverage gate is part of the project contract. Do not lower it.

## Core Commands

- `sync`: snapshot configured fixture or HTTP CSV sources into the local raw lake.
- `build-features`: normalize raw snapshots into curated Parquet tables and race tiers.
- `forecast run`: refresh data, rebuild features, run models, simulate outcomes, emit
  artifacts, plots, rewards, diagnostics, and performance metadata.
- `backtest run`: score historical forecast fixtures and component ablations.
- `report build`: rebuild diagnostics and methodology files for an existing run.
- `benchmark run`: measure simulation throughput using the configured performance engine.
- `results compare`: compare an existing forecast run against known actual results.

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

## End-To-End Forecast Flow

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

## Forecast State Machine

```mermaid
stateDiagram-v2
    [*] --> SourceSync
    SourceSync --> CuratedBuild: all sources recorded
    SourceSync --> CuratedBuild: failed sources recorded
    CuratedBuild --> Tiering
    Tiering --> Forecastable: Tier A/B
    Tiering --> TrackedOnly: Tier C
    Forecastable --> ComponentModels
    TrackedOnly --> ArtifactWrite
    ComponentModels --> Ensemble
    Ensemble --> Simulation
    Simulation --> ArtifactWrite
    ArtifactWrite --> Diagnostics
    Diagnostics --> Rewards
    Rewards --> [*]
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

## Model Shape

```mermaid
classDiagram
    class PollingModel {
      +weighted polling estimate
      +sample-size weighting
      +time decay and uncertainty
    }
    class FundamentalsModel {
      +partisan lean
      +incumbency and finance
      +turnout history
    }
    class MarketModel {
      +public market probability
      +liquidity/spread gate
      +normal probability-to-share map
    }
    class PublicSignalModel {
      +news and pageview signal
      +experimental admission
    }
    class EnsembleModel {
      +trusted component weights
      +race-level normalization
    }
    class SimulationEngine {
      +national/region/office factors
      +Numba binary draw kernel
      +thresholded control outputs
    }

    PollingModel --> EnsembleModel
    FundamentalsModel --> EnsembleModel
    MarketModel --> EnsembleModel
    PublicSignalModel --> EnsembleModel
    EnsembleModel --> SimulationEngine
```

## Backtest And Reward Flow

```mermaid
flowchart TB
    predictions[/"backtest_predictions.parquet"/]
    metrics["score_predictions"]
    brier["Brier and log score"]
    calibration["calibration and ECE"]
    coverage["interval coverage"]
    ablation{"component ablations"}
    rewards["R4-R8 reward gates"]

    predictions --> metrics
    metrics --> brier
    metrics --> calibration
    metrics --> coverage
    brier --> ablation
    calibration --> rewards
    coverage --> rewards
    ablation --> rewards
```

## Historical Comparison Flow

```mermaid
flowchart LR
    forecast["2024 pre-election forecast run"]
    actuals[("curated actual results")]
    compare["results compare"]
    table[/"result_comparison.parquet"/]
    summary[/"result_comparison_summary.json"/]
    html["result_comparison.html"]
    plots[/"comparison plots"/]

    forecast --> compare
    actuals --> compare
    compare --> table
    compare --> summary
    compare --> plots
    table --> html
    summary --> html
    plots --> html
```

## Performance Flow

```mermaid
flowchart LR
    config{{"performance config"}}
    sim["SimulationEngine"]
    branch{"engine == numba and available?"}
    numba["Numba parallel kernel"]
    python["Python fallback"]
    draws[("forecast_draws.parquet")]
    perf[/"performance.json"/]
    reward(("R12"))

    config --> sim --> branch
    branch -- yes --> numba --> draws
    branch -- no --> python --> draws
    sim --> perf --> reward
```

## Data Quality Tiers

```mermaid
flowchart TD
    race["discovered race"]
    signals{"validated signals?"}
    full{"polls or market plus fundamentals?"}
    sparse{"fundamentals plus any signal?"}
    tierA["Tier A: full probabilistic output"]
    tierB["Tier B: sparse, wide uncertainty"]
    tierC["Tier C: tracked, probability withheld"]

    race --> signals --> full
    full -- yes --> tierA
    full -- no --> sparse
    sparse -- yes --> tierB
    sparse -- no --> tierC
```

## Trust Model

The engine distinguishes tracked races from forecastable races. Tier A/B races receive
probabilistic outputs. Tier C races remain in the catalog, but trusted probabilities are
withheld. The reward card checks provenance, reproducibility fingerprints, sync
integrity, calibration reporting, baseline competition, component admission,
sparse-race honesty, uncertainty quality, public-signal discipline, explainability,
plot generation, and performance metadata. Fixture-limited trust gates remain red rather
than pretending that four historical rows validate the model.

## API Credentials

No API credentials are needed to run the default fixture-backed engine, backtests,
plots, diagnostics, or benchmarks. The first live source in `configs/sources_live.yaml`
also runs keylessly because it consumes a public FiveThirtyEight/Datasette CSV stream.
For live-ingestion expansion, see [`docs/api_requirements.md`](docs/api_requirements.md).

Google Civic is optional. The current live path does not use `GOOGLE_CIVIC_API_KEY`, and
Civic should not block polling, fundamentals, market, or historical-result ingestion.

Current key names expected by the credential template:

```bash
awk -F= '/^[A-Za-z_][A-Za-z0-9_]*=/ {print $1}' .env.example
```

Remaining live-adapter implementation order:

1. Fundamentals: Census, FRED, BEA/BLS, and public election-result archives.
2. Race catalog: Google Civic and official state/federal race metadata.
3. Polls: public poll feeds and pollster metadata.
4. Markets: read-only Kalshi/Polymarket public data, with no trading credentials.
5. Public signals: GDELT and Wikimedia/pageview-style public attention features.

Until those adapters exist, a "full live run" means live polling plus deterministic
support tables, plots, rewards, performance metadata, and the fixture scorecard.
