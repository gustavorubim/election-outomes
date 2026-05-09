# Technical Appendix

This appendix is the statistical contract for the forecasting engine. It is written as a
model note rather than a product overview: notation first, equations second, and
implementation status called out explicitly where the current code is still a pragmatic
approximation.

The engine forecasts distributions over outcomes. Point estimates are intermediate
quantities; the published probabilities, intervals, control outcomes, and diagnostics
come from simulated joint draws.

## 1. Notation

Indexes:

- `r`: race.
- `o`: option or candidate within race `r`.
- `t`: calendar date.
- `c`: election cycle.
- `g`: geography, usually state for the presidential benchmark.
- `b`: control body, such as `president`, `senate`, or `house`.
- `d`: simulation draw.
- `k`: model component, such as polling, fundamentals, markets, or public signals.

Observed data:

$$
\mathcal{D}_{\le a}
= \{ \text{polls, fundamentals, markets, public signals, metadata} :
\text{source\_time} \le a \}
$$

where `a` is the forecast `as_of` date. Actual results for target cycle `c` are excluded
from training and forecasting when the forecast is dated before Election Day.

Latent election-day quantities:

$$
\boldsymbol{\theta}_{r}
= (\theta_{r1}, \ldots, \theta_{rO_r}),
\qquad
\sum_{o=1}^{O_r}\theta_{ro}=1,
\qquad
0 \le \theta_{ro} \le 1
$$

For two-option candidate races the margin is:

$$
m_r = \theta_{r1} - \theta_{r2}
$$

Winner indicator:

$$
W_{ro} = \mathbb{1}\{\theta_{ro} = \max_j \theta_{rj}\}
$$

The published win probability is:

$$
\Pr(W_{ro}=1 \mid \mathcal{D}_{\le a})
\approx
\frac{1}{D}\sum_{d=1}^{D} W^{(d)}_{ro}
$$

## 2. Data And Lineage Contract

Every source row is treated as an auditable observation, not as an anonymous feature.
The source registry defines:

$$
s = (\text{id}, \text{table}, \text{adapter}, \text{path/url}, \text{parser\_version},
\text{terms})
$$

Sync produces immutable raw payloads and a source manifest:

$$
\text{manifest}_s =
(\text{retrieved\_at}, \text{content\_hash}, \text{parser\_version},
\text{status}, \text{downstream\_usage})
$$

Forecast rows must carry:

$$
\text{lineage}_{ro} =
(\text{source\_manifest\_hash}, \text{model\_config\_hash})
$$

`R2_provenance` is satisfied only when forecast rows can be traced back to non-empty
source hashes and the model configuration hash used for the run.

```mermaid
flowchart LR
    registry["source registry"]
    raw[("raw snapshots")]
    manifest[/"source_manifest.parquet"/]
    curated[("curated tables")]
    features[("feature bundle")]
    forecast[/"forecast artifacts"/]

    registry --> raw
    raw --> manifest
    manifest --> curated
    curated --> features
    features --> forecast
    manifest --> forecast
```

## 3. Race Eligibility And Sparse Honesty

Let `Q_r` be a data-quality vector for race `r`:

$$
Q_r =
(\text{poll\_count}, \text{poll\_freshness}, \text{fundamental\_coverage},
\text{market\_coverage}, \text{source\_lineage}, \text{metadata\_completeness})
$$

The tier map is:

$$
T_r =
\begin{cases}
A, & Q_r \text{ passes full probability thresholds} \\
B, & Q_r \text{ supports a sparse forecast with wider uncertainty} \\
C, & Q_r \text{ is tracked but not trusted for probability output}
\end{cases}
$$

Tier C semantics are strict:

$$
T_r=C \Rightarrow
\Pr(W_{ro}=1 \mid \mathcal{D}_{\le a}) \text{ is withheld}
$$

This is a core modeling choice. A known race with inadequate data is not silently
converted into a false-precision forecast.

## 4. Polling Model

### 4.1 Observation Equation

For a poll `i` of race `r`, option `o`, and field end date `t_i`, define the observed
share:

$$
y_i \in [0,1]
$$

The intended observation model is:

$$
y_i =
x_{rot_i}
+ h_{j(i),o}
+ \mu_{\text{mode}(i)}
+ \pi_{\text{population}(i)}
+ \sigma_{\text{sponsor}(i)}
+ \epsilon_i
$$

where:

- `x_{rot}` is latent support for option `o` in race `r` at date `t`.
- `h_{j(i),o}` is the pollster-option house effect.
- Mode, population, and sponsor terms represent systematic measurement adjustments.
- `epsilon_i` combines sampling and nonsampling polling error.

The current implementation collapses mode/population/sponsor effects into an effective
sample-size adjustment and keeps the additive estimated house effect:

$$
y_i^\star = y_i - \hat h_{j(i),o}
$$

### 4.2 Effective Sample Size And Poll Variance

The implemented effective sample size is:

$$
n_i^{eff}
= \max\{1,\ n_i
\cdot w_{\text{population}(i)}
\cdot w_{\text{methodology}(i)}
\cdot w_{\text{sponsor}(i)}\}
$$

The observation variance is:

$$
R_i =
\frac{\max\{y_i(1-y_i),\ 10^{-6}\}}{n_i^{eff}}
+ \sigma_{ns}^{2}
$$

where `sigma_ns` is the configured nonsampling-error floor. The small floor prevents
zero sampling variance for degenerate shares.

### 4.3 State Evolution

For each race-option trajectory:

$$
x_{rot} = x_{ro,t-1} + \eta_{rot},
\qquad
\eta_{rot} \sim \mathcal{N}(0,\ q\Delta t)
$$

Current implementation:

- Gaussian random-walk Kalman filter.
- Daily process variance `q`.
- Initial mean from `previous_vote_share` when available, otherwise 0.5.
- Initial variance from config.
- No backward smoother in the current production path.

### 4.4 Kalman Update

Prediction step:

$$
m^-_t = m_{t-1},
\qquad
P^-_t = P_{t-1} + q\Delta t
$$

For each poll observation on date `t`:

$$
K_i = \frac{P^-_t}{P^-_t + R_i}
$$

$$
m_t = m^-_t + K_i(y_i^\star - m^-_t)
$$

$$
P_t = (1-K_i)P^-_t
$$

The polling component reports:

$$
\hat\theta^{poll}_{roa}=m_a
$$

$$
\hat\sigma^{poll}_{roa}
= \max\{\sqrt{P_a},\ \sigma_{ns}\}
$$

and a marginal normal-approximation win probability:

$$
\hat p^{poll}_{roa}
= \Phi\left(
\frac{\hat\theta^{poll}_{roa}-0.5}{\hat\sigma^{poll}_{roa}}
\right)
$$

This marginal probability is a component signal. Final race probabilities are simulation
probabilities, not merely this closed-form value.

### 4.5 Empirical-Bayes House Effects

Let the raw pollster residual for pollster `j` and option `o` be:

$$
\bar e_{jo}
=
\frac{
\sum_{i:j(i)=j,o(i)=o} \omega_i (y_i - \tilde x_{rot_i})
}{
\sum_{i:j(i)=j,o(i)=o} \omega_i
},
\qquad
\omega_i = R_i^{-1}
$$

where `tilde x` is the reference trajectory from the current Kalman pass.

The implemented shrinkage estimator is:

$$
\lambda_{jo} = \frac{n_{jo}}{n_{jo}+\kappa}
$$

$$
\hat h_{jo}
= \text{clip}\left(
\lambda_{jo}\bar e_{jo}
+ (1-\lambda_{jo})h^{prior}_{jo},
\ -h_{max},\ h_{max}
\right)
$$

The model iterates the house-effect estimate against the Kalman trajectory. This is
empirical Bayes, not full hierarchical Bayes.

Planned Bayesian extension:

$$
h_{jo} \sim \mathcal{N}(0,\tau_h^2),
\qquad
\tau_h \sim \text{HalfNormal}(s_h)
$$

with joint posterior inference over trajectories, house effects, and variance terms.

## 5. Fundamentals Model

### 5.1 Training Target

For historical option rows with known outcomes:

$$
\Delta_{ro}
= y^{actual}_{ro} - y^{previous}_{ro}
$$

where `previous_vote_share` is the prior-cycle or prior-result baseline for the option.

Features:

$$
\mathbf{x}_{ro}
= [
\text{partisan\_lean}_{ro},
\text{economic\_index}_{ro},
\text{demographic\_turnout\_index}_{ro},
\text{incumbent}_{ro},
\text{fundraising\_usd}_{ro}
]
$$

Party-signed features are encoded before fitting so that a favorable state environment
raises the aligned party and lowers the opposing party.

### 5.2 Standardized Ridge Fit

If enough prior-cycle rows exist, features are standardized:

$$
z_{roj} = \frac{x_{roj}-\bar x_j}{s_j}
$$

The ridge objective is:

$$
(\hat\alpha,\hat{\boldsymbol\beta})
=
\arg\min_{\alpha,\boldsymbol\beta}
\sum_{(r,o)\in \mathcal{T}}
\left(
\Delta_{ro} - \alpha - \mathbf{z}_{ro}^{\top}\boldsymbol\beta
\right)^2
+ \lambda \|\boldsymbol\beta\|_2^2
$$

The intercept is not penalized. Coefficients are then transformed back to the raw feature
scale for reporting:

$$
\hat\beta^{raw}_j = \frac{\hat\beta_j}{s_j}
$$

$$
\hat\alpha^{raw}
=
\hat\alpha - \sum_j \hat\beta_j\frac{\bar x_j}{s_j}
$$

Predicted fundamentals share:

$$
\hat\theta^{fund}_{ro}
=
\text{clip}\left(
y^{previous}_{ro}
+ \hat\alpha^{raw}
+ \mathbf{x}_{ro}^{\top}\hat{\boldsymbol\beta}^{raw},
\ 0.05,\ 0.95
\right)
$$

Shares are normalized within race:

$$
\tilde\theta^{fund}_{ro}
=
\frac{\hat\theta^{fund}_{ro}}
{\sum_j \hat\theta^{fund}_{rj}}
$$

Component probability:

$$
\hat p^{fund}_{ro}
=
\Phi\left(
\frac{\tilde\theta^{fund}_{ro}-0.5}{\sigma_{fund}}
\right)
$$

### 5.3 Fallback Model

If the training set is too small, the engine uses explicit handpicked coefficients and
marks `fit_status` as a fallback:

$$
\hat\theta^{fund}_{ro}
= y^{previous}_{ro}
+ \sum_j x_{roj}\beta^{default}_j
$$

The model card must surface this distinction. A fallback fundamentals row is a usable
prior, not evidence of a learned structural model.

## 6. Market Model

For an admitted public prediction-market quote:

$$
p^{mkt}_{ro} \in (0,1)
$$

Admission gate:

$$
\mathbb{1}^{mkt}_{ro}
=
\mathbb{1}\{
\text{open\_interest}\ge O_{min}
\land
\text{spread}\le S_{max}
\}
$$

Probability-to-share proxy:

$$
\hat\theta^{mkt}_{ro}
=
0.5
+ \sigma_{mkt}\Phi^{-1}(p^{mkt}_{ro})
- b_{FL}
$$

where `b_FL` is the configured favorite-longshot-bias adjustment. This is a proxy, not a
market microstructure model. Markets are read-only signals and this repository does not
trade.

## 7. Public-Signal Model

Public signals are modeled as experimental features:

$$
\hat p^{signal}_{ro}
= f_{signal}(\text{news}, \text{pageviews}, \text{official releases}, \ldots)
$$

Admission policy:

$$
\text{trusted}_{signal}=1
\Rightarrow
\text{leakage checks pass}
\land
\text{rolling-origin ablation improves score}
$$

Default behavior is conservative: compute where data exists, report the value, but keep
it outside the trusted ensemble unless evidence supports admission.

## 8. Ensemble Model

Let `A_k` be the admission indicator for component `k`, and `w_k` the configured or
learned component weight.

The component-weighted vote-share signal is:

$$
\bar\theta_{ro}
=
\frac{\sum_k A_k w_k \hat\theta^{(k)}_{ro}}
{\sum_k A_k w_k}
$$

Within each race:

$$
\hat\theta^{ens}_{ro}
=
\frac{\bar\theta_{ro}}{\sum_j \bar\theta_{rj}}
$$

The component-weighted marginal probability is:

$$
\hat p^{ens}_{ro}
=
\frac{\sum_k A_k w_k \hat p^{(k)}_{ro}}
{\sum_k A_k w_k}
$$

The component uncertainty proxy is:

$$
\hat\sigma^{ens}_{ro}
=
\frac{\sum_k A_k w_k \hat\sigma^{(k)}_{ro}}
{\sum_k A_k w_k}
$$

Important interpretation:

- `hat p^{ens}` is a component-blend diagnostic.
- The published `winner_probability` comes from simulation draws.
- Marginal probabilities are not renormalized across options.

Per-race driver attribution is the stored contribution vector:

$$
C_{rok}
=
(w_k,\ \hat p^{(k)}_{ro},\ \hat\theta^{(k)}_{ro},
\ w_k\hat p^{(k)}_{ro},\ w_k\hat\theta^{(k)}_{ro})
$$

## 9. Joint Simulation Model

### 9.1 Error Decomposition

For draw `d`, race `r`, and geography `g(r)`, the systematic election error is:

$$
E^{sys}_{rd}
=
N_d
+ G_{g(r)d}
+ R_{\rho(r)d}
+ O_{\omega(r)d}
$$

where:

$$
N_d \sim \mathcal{N}(0,\sigma_N^2)
$$

$$
\mathbf{G}_d \sim \mathcal{N}(\mathbf{0},\ \Sigma_G)
$$

$$
R_{\rho d} \sim \mathcal{N}(0,\sigma_R^2),
\qquad
O_{\omega d} \sim \mathcal{N}(0,\sigma_O^2)
$$

`G` is used when a residual covariance artifact is available. Region and office shocks
remain additive, so the covariance path does not disable those factor layers.

Local residual:

$$
L_{rd}
=
\frac{\sigma_r}{s_\nu} T_{\nu,rd},
\qquad
T_{\nu,rd}\sim t_\nu,
\qquad
s_\nu = \sqrt{\frac{\nu}{\nu-2}}
$$

The scale:

$$
\sigma_r =
\max\left(
\sigma_{tier(T_r)},\ 0.5\hat\sigma^{ens}_{ro}
\right)
$$

### 9.2 Two-Option Draws

For binary races, the first sorted option receives:

$$
\theta^{(d)}_{r1}
=
\text{clip}\left(
\hat\theta^{ens}_{r1}
+ E^{sys}_{rd}
+ L_{rd},
\ 0.02,\ 0.98
\right)
$$

and:

$$
\theta^{(d)}_{r2}=1-\theta^{(d)}_{r1}
$$

Winner:

$$
W^{(d)}_{ro}
=
\mathbb{1}\{\theta^{(d)}_{ro}=\max_j\theta^{(d)}_{rj}\}
$$

### 9.3 Multi-Option Draws

For multi-option races, baseline shares are sampled from a Dirichlet approximation:

$$
\boldsymbol\pi^{(d)}_r
\sim
\text{Dirichlet}(\boldsymbol\alpha_r),
\qquad
\alpha_{ro}=\max(70\hat\theta^{ens}_{ro},1)
$$

The systematic error perturbs centered log shares:

$$
\ell_{ro}^{(d)} = \log \pi_{ro}^{(d)}
$$

$$
\tilde\ell_{ro}^{(d)}
=
\ell_{ro}^{(d)}
+ E_{rd}
\frac{\ell_{ro}^{(d)}-\bar\ell_r^{(d)}}
{\max(\text{sd}(\boldsymbol\ell_r^{(d)}),10^{-3})}
$$

Then:

$$
\theta^{(d)}_{ro}
=
\frac{\exp(\tilde\ell_{ro}^{(d)})}
{\sum_j \exp(\tilde\ell_{rj}^{(d)})}
$$

This is a practical approximation; ranked-choice and high-dimensional multi-candidate
models remain future work.

### 9.4 Turnout Draws

The current turnout base is:

$$
T^{base}_r =
\text{registered\_voters}_r
\cdot
\text{historical\_turnout\_rate}_r
$$

Draw-level turnout is:

$$
T^{(d)}_r
=
\text{round}\left(
T^{base}_r \max(0.6, 1+N_d)
\right)
$$

This is a projection proxy, not a full demographic turnout model.

## 10. Residual Covariance Estimation

Rolling-origin residuals are:

$$
e_{cag}
=
\frac{1}{|R_{cag}|}
\sum_{r\in R_{cag}}
\left(
\hat\theta^{ens}_{r,a}
- \theta^{actual}_{r}
\right)
$$

where residuals are grouped by cycle `c`, as-of cut `a`, and geography `g`.

The residual matrix is:

$$
\mathbf{E} \in \mathbb{R}^{n \times G}
$$

Empirical covariance:

$$
S = \text{cov}(\mathbf{E})
$$

Structured target covariance:

$$
T_{ij}
=
\rho_{ij}\sqrt{S_{ii}S_{jj}}
$$

with:

$$
\rho_{ij}
=
\begin{cases}
1, & i=j \\
\rho_{\text{same region}}, & \text{region}(i)=\text{region}(j) \\
\rho_{\text{cross region}}, & \text{otherwise}
\end{cases}
$$

Shrinkage:

$$
\Sigma_G
=
(1-\lambda)S + \lambda T
$$

Positive-semidefinite projection:

$$
\Sigma_G = Q\max(\Lambda,\epsilon I)Q^\top
$$

If fewer than two residual observations exist, no covariance artifact is emitted. The
simulation then falls back to national, region, office, and local factors.

## 11. Control And Electoral College Outcomes

For control body `b` and party `p`, draw-level seat or electoral-vote count is:

$$
S^{(d)}_{bp}
=
\sum_{r:b(r)=b}
s_r
\mathbb{1}\{W^{(d)}_{r,p}=1\}
$$

where `s_r` is the unit counted by `control_body`:

- House scenarios: House seats.
- Senate scenarios: Senate seats.
- Presidential state scenarios: Electoral College votes.

Control threshold:

$$
\tau_b =
\begin{cases}
\text{configured threshold}, & \text{if present} \\
\lfloor \text{modeled seats}/2 \rfloor + 1, & \text{otherwise}
\end{cases}
$$

Control probability:

$$
\Pr(\text{control}_{bp})
\approx
\frac{1}{D}
\sum_{d=1}^{D}
\mathbb{1}\{S^{(d)}_{bp}\ge\tau_b\}
$$

Pivotal indicator for race `r`:

$$
I^{(d)}_{rbp}
=
\mathbb{1}\{
W^{(d)}_{rp}=1,\ S^{(d)}_{bp}\ge\tau_b,\ S^{(d)}_{bp}-s_r<\tau_b
\}
$$

$$
\quad+
\mathbb{1}\{
W^{(d)}_{rp}=0,\ S^{(d)}_{bp}<\tau_b,\ S^{(d)}_{bp}+s_r\ge\tau_b
\}
$$

Pivotal rate:

$$
\text{pivotal}_{rbp}
=
\frac{1}{D}\sum_d I^{(d)}_{rbp}
$$

For presidential scenarios, `tau_president=270` and the count is the full Electoral
College total when all 50 states plus DC are present.

## 12. Ecosystem Outcomes

Recount proxy:

$$
\Pr(\text{recount}_r)
\approx
\frac{1}{D}\sum_d
\mathbb{1}\{
\theta^{(d)}_{r(1)}-\theta^{(d)}_{r(2)} \le 0.01
\}
$$

Certification-risk proxy:

$$
\Pr(\text{certification risk}_r)
\approx
0.6
\cdot
\frac{1}{D}\sum_d
\mathbb{1}\{
\theta^{(d)}_{r(1)}-\theta^{(d)}_{r(2)} \le 0.005
\}
$$

The multiplier is explicitly a placeholder. The artifact labels this field as
`close_margin_proxy_not_calibrated`. It should not be read as a calibrated
administrative or legal-risk forecast.

Demographic turnout composition is also explicitly marked as not estimated until a
group-level turnout model is implemented.

## 13. Rolling-Origin Backtesting

For target cycle `c`, the training set is:

$$
\mathcal{T}_c =
\{(r,o): \text{cycle}(r) < c\}
$$

The holdout set is:

$$
\mathcal{H}_c =
\{(r,o): \text{cycle}(r) = c\}
$$

For each as-of offset:

$$
a_{c,\delta} = \text{ElectionDay}_c - \delta
$$

the feature bundle is filtered:

$$
\mathcal{D}_{train}
=
\mathcal{D}_{\le a_{c,\delta}}
\cap
\mathcal{T}_c
$$

$$
\mathcal{D}_{test}
=
\mathcal{D}_{\le a_{c,\delta}}
\cap
\mathcal{H}_c
$$

No target-cycle results are available to the component fits.

Default offsets:

$$
\delta \in \{90,60,30,7,1\}
$$

when source rows exist by those dates.

## 14. Scoring And Calibration

Brier score:

$$
\text{Brier}
=
\frac{1}{n}\sum_i(\hat p_i-y_i)^2
$$

Log score:

$$
\text{LogScore}
=
-\frac{1}{n}\sum_i
\left[
y_i\log(\hat p_i)
+(1-y_i)\log(1-\hat p_i)
\right]
$$

Calibration model:

$$
\Pr(y_i=1)
=
\text{logit}^{-1}
\left(
\alpha + \beta\ \text{logit}(\hat p_i)
\right)
$$

The implementation estimates `(alpha, beta)` with a small-ridge logistic regression.
Perfect calibration is approximately:

$$
\alpha=0,\qquad \beta=1
$$

Expected calibration error:

$$
\text{ECE}
=
\sum_{b=1}^{B}
\frac{|I_b|}{n}
\left|
\frac{1}{|I_b|}\sum_{i\in I_b}\hat p_i
-
\frac{1}{|I_b|}\sum_{i\in I_b}y_i
\right|
$$

The current implementation uses quantile-adaptive probability bins when possible.

Interval coverage:

$$
\text{Coverage}_{90}
=
\frac{1}{n}\sum_i
\mathbb{1}\{L_i^{90}\le y_i^{share}\le U_i^{90}\}
$$

Baseline probability:

$$
\hat p^{base}_{ro}
=
\Phi\left(
\frac{y^{previous}_{ro}-0.5}{\sigma_{base}}
\right)
$$

`sigma_base` is empirical from prior-cycle residuals once enough rows exist; otherwise
it falls back to the configured default.

## 15. Component Admission And Rewards

Let `M_min` be the configured minimum rolling-origin row count for trust.

Trustworthy backtest condition:

$$
\mathbb{1}^{trust}
=
\mathbb{1}\{
\text{rolling\_origin\_executed}
\land
n_{rows}\ge M_{min}
\}
$$

For component `k`, the ablation check is:

$$
\Delta_k =
\text{Brier}_k - \text{Brier}_{baseline}
$$

Component admission under trustworthy backtests:

$$
A_k =
\mathbb{1}\{\Delta_k \le 0\}
$$

If the backtest sample is too small, the forecast may use configured defaults for
pragmatic output, but the reward card must mark evidence-based admission as not
certified.

Reward interpretation:

- `R4`: calibration metrics are reported.
- `R5`: ensemble beats or matches baseline with enough rows.
- `R6`: component admission is evidence-based with enough rows.
- `R8`: interval coverage is within tolerance with enough rows.

## 16. Historical Result Comparison

A completed-cycle comparison joins a forecast run to actual results:

$$
\text{comparison}_{ro}
=
(\hat p_{ro}, \hat\theta_{ro}, y^{actual}_{ro}, W^{actual}_{ro})
$$

Race-level state accuracy:

$$
\text{Accuracy}
=
\frac{1}{R}\sum_r
\mathbb{1}\{
\arg\max_o \hat p_{ro}
=
\arg\max_o W^{actual}_{ro}
\}
$$

Vote-share mean absolute error:

$$
\text{MAE}
=
\frac{1}{n}\sum_{ro}
|\hat\theta_{ro}-y^{actual}_{ro}|
$$

Presidential EC winner accuracy compares the modeled Electoral College winner with the
actual winner only when the scenario contains all 538 electoral votes. Otherwise the
comparison is labeled as a modeled slice.

## 17. Diagnostics And Plot Contract

The diagnostics page should make the distributional claims visible before lower-level
details:

1. Top-line win/control probabilities and EC distribution.
2. Simulation swarm or draw trace next to the histogram.
3. State/race probability and margin tables.
4. Driver attribution by race.
5. Calibration and interval coverage.
6. Model-quality section with Kalman trajectory diagnostics, chain-style draw traces,
   convergence plots, residual covariance summaries, and benchmark scores.

Plot families:

- Electoral College distribution.
- Simulation swarm and chain-style EC traces.
- Winner probability bars.
- Vote-share interval projections.
- Polling trajectories with dots and posterior bands.
- Calibration curve and ECE.
- Interval coverage.
- Brier/log score by component.
- Cross-cycle cycle-eval summaries.
- Silver/FiveThirtyEight methodology-readiness benchmark.

## 18. Performance Model

The expensive loop is draw generation. The optimized binary-race path is:

$$
(\hat\theta, T^{base}, N, L)
\rightarrow
\{\theta^{(d)}_{ro}, W^{(d)}_{ro}, T^{(d)}_r\}_{d=1}^{D}
$$

Implementation policy:

- Polars/DuckDB for table transforms.
- NumPy for dense arrays.
- Numba parallel kernels for repeated draw-level loops.
- Python fallback when Numba is unavailable.

`performance.json` records:

$$
(\text{requested engine}, \text{actual engine}, \text{parallel flag},
\text{Numba availability}, \text{thread count}, D)
$$

`R12_performance_contract` verifies the requested acceleration path is recorded and that
Numba is used when requested and available.

## 19. Current Status Versus Frontier Target

Implemented:

- Fixture-backed and presidential-panel forecasting contract.
- Source manifest and row-level forecast lineage.
- Tier A/B/C sparse-race gates.
- Deterministic Kalman polling with empirical-Bayes house-effect shrinkage.
- Standardized ridge fundamentals when enough rows exist.
- Market inverse-normal share proxy with liquidity/spread gates.
- Weighted ensemble with contribution attribution.
- Correlated simulation with national, geography covariance, region, office, and
  heavy-tailed local errors.
- Electoral College control threshold and pivotal-rate calculation.
- Rolling-origin backtesting across cycles and as-of offsets.
- Calibration, interval coverage, result comparison, and cycle-eval dashboards.
- Numba binary simulation path with Python fallback.

Still not frontier:

- No full posterior MCMC/SMC over all polling and election-error parameters.
- No hierarchical Bayesian fundamentals prior.
- No calibrated legal/process model for certification risk.
- No group-level demographic turnout model.
- No real-time nationwide live adapter coverage across every public data source.
- Multi-option and ranked-choice models remain approximations.

The immediate modeling target is not to hide these gaps. It is to make each approximation
measurable, benchmarked, and replaceable behind stable artifacts.

## 20. Acceptance Standard

Documentation changes do not alter the acceptance gate. A valid repo state must pass:

```bash
uv sync
chflags -R nohidden .venv
uv run ruff check
uv run ruff format --check
PYTHONPATH=src uv run pytest --cov=src/election_outcomes --cov-fail-under=90
```
