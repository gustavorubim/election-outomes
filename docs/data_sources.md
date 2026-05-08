# Data Sources

The default source registry lives in `configs/sources.yaml`. A first live registry lives
in `configs/sources_live.yaml`. Each source records:

- stable source id
- logical curated table
- retrieval type
- URL or local path
- parser version
- license or terms note

The fixture registry is intentionally shaped like public sources so live adapters can
write the same raw manifest and curated tables. The default fixture registry includes a
compact presidential state-cycle panel for 2000-2024. It is derived from public
presidential returns made available through MIT Election Data and Science Lab-style
state result files mirrored for deterministic offline testing; the 2024 rows include
full 50-state-plus-DC Electoral College weights.

Current source modes:

- `fixture`: copy a local CSV into the raw lake and hash it.
- `http_csv`: download a public CSV endpoint into the raw lake and hash it.

The first live adapter is `fivethirtyeight_president_polls` in
`configs/sources_live.yaml`. It downloads the FiveThirtyEight/Datasette presidential
poll CSV stream and normalizes Wisconsin 2020 Democratic/Republican rows into the
existing `polls` contract for `US-PRES-WI-2020`.

The compact presidential panel is exposed through five parser versions that all point to
the same raw file and emit separate curated contracts:

- `president-state-panel-races-v1`
- `president-state-panel-options-v1`
- `president-state-panel-results-v1`
- `president-state-panel-fundamentals-v1`
- `president-state-panel-polls-v1`

Remaining source families to implement include MIT Election Lab, FEC, VoteHub or other
poll feeds, Census/FRED/BEA/BLS, Kalshi, Polymarket, GDELT, Wikimedia, and optional
Civic-style race catalog enrichment.
