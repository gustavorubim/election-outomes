# Modeling Contract

The core forecast object is a distribution, not a point estimate:

$$
\Pr(W_{ro}=1 \mid \mathcal{D}_{\le a})
\approx
\frac{1}{D}\sum_{d=1}^{D} W^{(d)}_{ro}
$$

where `r` is a race, `o` is an option/candidate, `a` is the forecast `as_of` date, and
`D` is the number of simulation draws.

## Required Row Contract

Forecastable component rows must include:

- `race_id`
- `option_id`
- `component`
- `marginal_win_probability`
- `vote_share`
- `uncertainty`
- `admitted`
- `explanation`

Final forecast rows must additionally carry:

- source-manifest lineage hash
- model-config lineage hash
- tier and tier reason
- data-quality flags
- component contribution JSON
- uncertainty explanation

## Tier Contract

Tier C races are tracked in `race_catalog.parquet` but do not receive trusted winner
probabilities in `race_forecasts.parquet`:

$$
T_r=C \Rightarrow \Pr(W_{ro}=1 \mid \mathcal{D}_{\le a}) \text{ is withheld}
$$

## Simulation Contract

Final forecast artifacts are generated from correlated simulation draws, not from
component point estimates alone. Component marginal probabilities are diagnostics;
published race probabilities are draw frequencies.

## Control Contract

`race_catalog.seats` is interpreted relative to `control_body`:

- House scenarios: House seats.
- Senate scenarios: Senate seats.
- Presidential state scenarios: Electoral College votes.

Control probabilities compare draw-level totals to configured thresholds:

$$
\Pr(\text{control}_{bp})
\approx
\frac{1}{D}\sum_{d=1}^{D}
\mathbb{1}\{S^{(d)}_{bp}\ge \tau_b\}
$$

For `control_body=president`, `tau_b=270` and `S` is summed Electoral College votes.

See [`technical_appendix.md`](technical_appendix.md) for the full notation, likelihoods,
simulation equations, backtest metrics, and reward gates.
