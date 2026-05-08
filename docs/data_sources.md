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
write the same raw manifest and curated tables.

Current source modes:

- `fixture`: copy a local CSV into the raw lake and hash it.
- `http_csv`: download a public CSV endpoint into the raw lake and hash it.

The first live adapter is `fivethirtyeight_president_polls` in
`configs/sources_live.yaml`. It downloads the FiveThirtyEight/Datasette presidential
poll CSV stream and normalizes Wisconsin 2020 Democratic/Republican rows into the
existing `polls` contract for `US-PRES-WI-2020`.

Remaining source families to implement include MIT Election Lab, FEC, VoteHub or other
poll feeds, Census/FRED/BEA/BLS, Kalshi, Polymarket, GDELT, Wikimedia, and optional
Civic-style race catalog enrichment.
