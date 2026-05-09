# Methodology

This engine estimates a joint distribution over election outcomes:

$$
\Pr(\boldsymbol{\theta}, \mathbf{W}, \mathbf{S} \mid \mathcal{D}_{\le a})
$$

where `theta` is race-level vote share, `W` is the winner indicator, `S` is
seat/control or Electoral College count, and `a` is the forecast `as_of` date.

The current implementation is a deterministic, auditable approximation to that posterior:

1. **Polling**: Gaussian state-space/Kalman polling model with sample-size observation
   variance, nonsampling floor, previous-share initialization, and empirical-Bayes
   pollster house-effect shrinkage.
2. **Fundamentals**: standardized ridge model over prior share, partisan lean, economic,
   demographic, incumbency, and finance features when enough historical rows exist;
   otherwise an explicit fallback prior.
3. **Markets**: public read-only market probabilities gated by liquidity and spread,
   then mapped to vote-share proxy through an inverse-normal transform.
4. **Public Signals**: news/pageview/official-release signals, experimental by default
   and admitted only after leakage and rolling-origin ablation checks.
5. **Ensemble**: weighted component blend over admitted components, with contribution
   attribution retained by race and option.
6. **Simulation**: correlated draw engine with national, geography covariance, region,
   office, and heavy-tailed local errors. Published probabilities and intervals come
   from these draws.

The rigorous mathematical contract lives in
[`technical_appendix.md`](technical_appendix.md). That document also identifies which
parts are implemented approximations and which remain frontier targets, including full
hierarchical Bayesian polling, richer turnout modeling, and broader live-source coverage.
