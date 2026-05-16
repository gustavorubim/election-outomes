# API Requirements

The current repo runs without external credentials because it uses deterministic
fixtures. Live adapters should read optional credentials from `.env` or shell
environment variables and record whether each source ran authenticated or public-only
in `source_manifest.parquet`.

Important current boundary: `.env` may contain real keys, but the first implemented
live path does not need them. The default `configs/sources.yaml` registry is
fixture-only. `configs/sources_live.yaml` adds a keyless HTTP CSV adapter for the
FiveThirtyEight/Datasette 2020 presidential poll stream, plus 2026 Senate, Governor,
and House poll streams. These are normalized into the `polls` table when upstream rows
exist for the configured cycle/stage/party filters. The same live registry includes a
keyless FRED UNRATE CSV adapter that emits model-bearing national macro fundamentals
for the compact 2026 Senate/Governor/House smoke races, plus Wikipedia raw-page
race-presence metadata. The Wikipedia rows are neutral `public_signals` metadata, not
model-bearing poll, fundamentals, or market observations. All other forecast support
tables in that run are still fixtures or configured race priors.

## Not Needed For Current Runs

- Plot generation: no keys. Plots are generated from local Parquet/JSON artifacts.
- Fixture ingestion: no keys.
- Backtests over committed golden fixtures: no keys.

## Recommended Live Credentials

- `GOOGLE_CIVIC_API_KEY`: required for Google Civic Information API requests.
- `CENSUS_API_KEY`: recommended for Census API volume; Census documents a key need for
  mobile/web apps or more than 500 daily queries.
- `GDELT_API_KEY`: needed for GDELT Cloud API endpoints that require bearer auth.

## Usually Public Or Keyless For Read-Only Use

- FiveThirtyEight/Datasette poll CSV streams: current live polling adapters, keyless.
- FRED graph CSV downloads: current UNRATE fundamentals adapter, keyless.
- Wikipedia raw-page race-presence metadata: current `http_text` adapter, keyless.
- Polymarket market data: public market/event endpoints are generally keyless; trading,
  portfolio, and authenticated WebSocket flows require credentials and are out of scope.
- Kalshi market data: public market data can be read without trading credentials; trading
  and account endpoints are out of scope.
- Wikimedia pageviews: public analytics reads are keyless for this use case.
- FEC and official election-office downloads: design adapters to support public downloads
  first and optional keys/rate-limit settings where available.

Before implementing each live adapter, re-check that source's current terms, rate limits,
and authentication requirements. The sync layer must record source URL, retrieval time,
content hash, parser version, and any auth mode in the manifest.

## Adapter Acceptance Contract

Each live adapter should add all of the following before it can influence forecasts:

- A `configs/sources.yaml` entry with source id, table name, parser version, license or
  terms note, URL/API endpoint, and auth mode.
- A raw snapshot with a content hash that can be rerun without silent overwrites.
- A parser that emits the same curated table contract used by the fixture table it
  replaces or extends.
- Tests using golden fixtures that do not require live credentials.
- A source-manifest row for success, unchanged, skipped, and failed states.
- README instructions showing how to run the adapter and which `.env` keys are optional
  or required.

The first production backtest upgrade should not depend on live APIs at runtime. It
should materialize historical snapshots first, then run rolling-origin splits over those
snapshots so `R5`, `R6`, and `R8` are repeatable.

## Current Live Smoke Run

```bash
uv run civic-signal forecast run \
  --sources-config sources_live.yaml \
  --data-dir data/live \
  --artifacts-dir artifacts/live \
  --as-of 2020-10-30 \
  --run-id wi-2020-live-polls

uv run civic-signal results compare \
  --sources-config sources_live.yaml \
  --data-dir data/live \
  --artifacts-dir artifacts/live \
  --forecast-run-id wi-2020-live-polls \
  --comparison-id wi-2020-live-polls-actuals \
  --cycle 2020 \
  --office-type president \
  --race-id US-PRES-WI-2020
```

This run proves API/file consumption, raw hashing, source-manifest provenance, parser
normalization, forecast artifact generation, and forecast-vs-actual comparison for one
real race. The 2026 Senate/Governor/House adapters are included in the same registry,
but the methodology-readiness gate still requires successful non-file model-bearing rows
for every required office before Bayes can become the production default. The FRED
fundamentals adapter can supply that compact smoke-scope evidence; neutral Wikipedia
race-presence rows are reported as metadata coverage only.
