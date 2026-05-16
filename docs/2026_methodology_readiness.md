# 2026 Methodology Readiness

This note describes the current Phase 8 verification scope. It is a fixture-backed
readiness check, not a claim that the full live 2026 production forecast is complete.

## Verification Command

```bash
uv run civic-signal verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-verification \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --quiet
```

The runner executes the selected scenario twice with the same run id to prove the stable
artifact fingerprint, runs the Bayesian daily-update gate, verifies artifact schemas and
reward gates, and writes:

- `artifacts/runs/<run_id>/phase8_verification.json`
- `artifacts/runs/<run_id>/visual_qa_checklist.json`
- `artifacts/runs/<run_id>/verification.json`
- `artifacts/runs/<run_id>/timeout_failover_audit.json`

## Current Scope

The default fixture scenario covers a synthetic non-control 2026 President tracker plus
2026 Senate, House, and Governor rows. It validates the Bayesian posterior artifact
contract, Senate joint summary, House hierarchy summary, Governor seat posterior,
cross-office shared draw stream, daily-update quality gate, dashboards, model card, and
Silver-style benchmark artifact.
The verification runner also exercises the configured Bayesian NUTS timeout/failover
policy on a forced fixture timeout. That audit proves the fallback path is visible; it
does not mean the forecast itself used a fallback.

The President tracker is not a live off-cycle adapter and is not an Electoral College
forecast. It is included to exercise President rows in the shared posterior and
cross-office latent artifact while keeping `control_body` empty so it does not affect
seat-count or control probabilities.

## Readiness Interpretation

The Phase 8 fixture pass means the orchestration and artifact contract are operational
for the current 2026 midterm panel. The Kalman path remains available through
`--inference-engine kalman`. Bayes is the configured operational default, and broad
production promotion is eligible after the live-scope rolling-origin Bayes/NUTS score
beat the legacy Kalman score without coverage degradation.

The default-switch decision is now machine-audited by:

```bash
uv run civic-signal verify readiness \
  --run-id bayes-default-readiness \
  --forecast-run-id phase8-verification \
  --bayes-backtest-run-id president-state-bayes-backtest \
  --legacy-backtest-run-id president-state-backtest
```

That command writes `artifacts/readiness/<run_id>/methodology_readiness.json` and
`methodology_readiness.md`. It blocks the switch unless Bayes dependencies are base
dependencies, config and docs declare Bayes as the default, Phase 8/hard reward gates
pass, live 2026 source scope is claimed, and rolling-origin Bayes evidence beats the
legacy Kalman scorecard without degrading interval coverage. Current publication
calibration is intentionally slope-bounded at `2.0` to avoid a default-switch decision
being driven by an overfit calibration transform rather than forecasting signal.
The current broad live-scope evidence is sufficient for the implemented gate:
`live-scope-prod-bayes-nuts-houseadj-norm` scored `0.124907` ensemble log loss versus
legacy Kalman `0.125154`, with 90% interval coverage `0.9662` versus legacy `0.9610`.
The live-source scope claim is evidence-based: Phase 8 inspects the curated source
manifest and curated tables and only reports `claimed` when successful non-file sources
contribute model-bearing 2026 rows for every office listed in the scenario's
`live_source_required_offices`. Neutral Wikipedia race-presence rows are reported as
`metadata_only`: they prove keyless HTTP text ingestion and source provenance, but they
are not enough to make Bayes the production default. The President tracker is
deliberately excluded from that list because it is a synthetic non-control artifact
exercise for a midterm year.

## Manual Review Checklist

- Open `diagnostics.html` and confirm the posterior diagnostics, fundamentals prior, and
  run metadata sections render.
- Open `model_card.md` and confirm the office-methodology section names Senate joint,
  House hierarchical, and cross-office artifacts.
- Inspect `reward_card.json` and confirm `R1_reproducibility`,
  `R13_posterior_quality`, and `R15_daily_update_quality` pass after the scenario run.
- Inspect `phase8_verification.json` and confirm `fixture_scope` records the
  President-tracker+Senate+House+Governor scope and keeps `live_2026_status` marked as
  `not_claimed`, `partial`, or `metadata_only` unless model-bearing live rows cover
  every required office.
- Inspect `timeout_failover_audit.json` and confirm the forced fixture timeout records
  the first fallback policy without setting a forecast-level fallback.
- Inspect `visual_qa_checklist.json` and confirm every referenced plot exists, is
  non-empty, has a title, and stays within the configured size budget.
