# Pricing problems

A `PricingProblem` is the answer to "what is this surrogate pricing?".

It owns every fact that depends on the model: how the model is calibrated,
the feature layout, the sampling domain, the label pricer, which slices are
worth plotting, how a trained surrogate is independently validated, and how
future states are simulated for an exposure profile.

The pipeline stages take a problem and ask it. None of them branches on
which model is configured, so

```python
config.data.pricing_model = "basket_black_scholes"
config.payoff.name = "asian_call"
```

is the whole switch.

## The pipeline

```
problem definition -> calibration -> dataset -> training -> validation -> reporting
```

Calibration is part of the problem, not a stage that happens to precede it.
`calibrate_problem(name, config, market_data)` returns a `CalibrationResult`
holding the fitted parameters *and* whatever the data could not determine.

## Adding a model

Two steps: implement, then register.

```python
from surrogate_modeling.pricing_problem import (
    CalibrationResult, PricingProblem, ProblemSpec, register_problem,
)


class HestonProblem(PricingProblem):

    name = "heston"

    @property
    def feature_names(self):
        return ("S", "K", "T", "v0", "kappa", "theta", "xi", "rho")

    def sample_features(self, u):
        ...                      # map a uniform (n, 8) block onto the domain

    def label_price_fn(self):
        return lambda x, key: ...    # twice differentiable in x


def calibrate_heston(config, market_data) -> CalibrationResult:
    ...


register_problem(ProblemSpec("heston", build=..., calibrate=calibrate_heston))
```

`pricing_model = "heston"` now works. The dataset, the surface plots, the
per-dimension Greek table, the archived configuration and the report all
follow the new feature layout on their own.

## What is required, and what is free

Four members are required: `name`, `feature_names`, `sample_features` and
`label_price_fn`. Everything else has a default derived from those.

| Member | Default | Effect of overriding |
|---|---|---|
| `feature_labels` | the terse names | spelled-out plot axes |
| `feature_bounds` | read off `sample_features` at its corners | — |
| `baseline_features` | the middle of the domain | a more meaningful plot anchor |
| `surface_specs` | feature 0 against every other | fewer or different slices |
| `discount_rate` | `0.0` | discounting in the exposure profile |
| `exchangeable_features` | none | a permutation-symmetry check |
| `shape_constraints` | none | sign checks on the learned gradient |
| `underlying_paths` | `None` | the training-path preview |
| `reference_price` | `None` | the independent Monte Carlo benchmark |
| `analytic_price` | `None` | a closed-form column, and an XVA reference |
| `arbitrage_bounds` | `None` | a model-free no-arbitrage check |
| `exposure_paths` | `None` | the risk stage |
| `exposure_strikes` | none | which strikes the exposure is reported at |

A stage whose optional method returns `None` prints that it is skipping and
why. It never falls back to a guess.

## Randomness

`label_price_fn` returns `f(x, key)`. The key is an argument, not a closure,
and `create_sobolev_labels` gives every sample its own.

Common random numbers are kept exactly where they are justified:

- **within one sample** — price, gradient and HVP are differentiated through
  the same path bundle, so the reported gradient really is the gradient of
  the reported price;
- **inside the calibration residual** — re-drawing paths at each solver step
  would make the residual discontinuous in the parameters.

They are *not* shared **across samples**. Derivatives come from automatic
differentiation within a sample, never from differencing two samples, so a
common stream across the dataset buys nothing — and it costs a coherent
error field. Measured over 24 base seeds: pairwise error correlation +0.80
with a shared key against +0.001 with independent keys, and a dataset-wide
mean error that does not shrink with the sample count. At `PRNGKey(0)` that
left every label low by about −0.71 in price units, with 63 of 64 probe
points below a high-accuracy reference.

`simulation.shared_label_keys = True` restores the old behaviour so the bias
stays reproducible; `tests/test_label_bias.py` measures both.

## Sampling scheme

The payoff decides. A terminal-only payoff (`payoff_spec(...).path_dependent
is False`) is drawn from the exact transition law in one step:

```
S_i(T) = S_i(0) exp((r - sigma_i^2/2) T + sigma_i sqrt(T) Z_i)
```

That is the law of a geometric Brownian motion, not an approximation of it,
so it carries no discretisation error and costs one normal per asset per
path instead of `num_steps` of them — measured at **63.7x** faster for a
price-plus-gradient-plus-HVP label. `num_steps` is then unread.

A path-dependent payoff (Asian) needs the whole trajectory and keeps the
stepping scheme.

Draws are **antithetic** by default: every `Z` is paired with `-Z`. Both have
the same law, so the estimator stays unbiased, and the two payoffs are
negatively correlated, which removes most of the level variance. Adapted
from diff-ml's `Bachelier.antithetic_payoff`.

## Validation

`surrogate_modeling.validation.run_reference_validation` is model-agnostic
and runs for every problem. Every statistic is chosen to survive a near-zero
price:

- **Fresh Monte Carlo, at higher accuracy than the labels.** Reported as
  bias, RMSE, MAE, *median* relative error and SMAPE. The bias term is the
  one that matters — a systematic offset is invisible to RMSE and to R².
- **The labels measured against the same reference.** This separates "the
  network is wrong" from "the labels it was fitted to are wrong".
- **Closed form, where one exists**, as the noise floor the surrogate should
  be read against.
- **No-arbitrage bounds**, with a tolerance of three Monte Carlo standard
  errors of the reference. Without it the check flags label noise: at deep
  in-the-money strikes the time value is smaller than one standard error.
- **Shape constraints** on the learned gradient — delta within `[0, w_i]`,
  `dV/dK <= 0`, `dV/dT >= 0`. No reference price needed.
- **Exchangeability**, in absolute price units and as a median relative
  deviation.
- **Diversification**: a basket call must be worth no more than its
  perfectly correlated limit.

## Built-in problems

| Name | Features | Closed form | Exchangeable |
|---|---|---|---|
| `black_scholes` | `S, K, T, sigma, r` | yes | no |
| `basket_black_scholes` | `S_1 … S_n, K, T` | **no** | the spots, when the basket is |
| `bachelier` | `S, K, T, sigma, r` | yes | no |
| `basket_bachelier` | `S_1 … S_n, K, T` | **yes, exact** | the spots, when the basket is |
| `heston` | `S, K, T, v0, kappa, theta, xi, rho` | yes, by Fourier inversion | no |
| `basket_heston` | `S_1 … S_n, K, T` | no | the spots, when the basket is |

A basket surrogate has per-asset deltas and gammas but no vega or rho: the
basket structure is fixed at construction rather than carried in the feature
vector.

### Bachelier versus Black-Scholes

The normal model replaces `dS = rS dt + sigma S dW` with a driftless normal
forward, `dF = sigma dW` and `F_0 = S exp(rT)`. Three consequences matter
for this repository.

**`sigma` changes units.** A Black-Scholes volatility is a percentage; a
normal volatility is a price per square root of time. At a spot of 312 a
0.29 lognormal vol corresponds to roughly 90 in normal units. The sampling
domain follows: the spot range is *additive* (`S ± n sigma sqrt(T)`), not
multiplicative, and the initial calibration guess is scaled by the spot.

**The basket becomes exact.** A weighted sum of jointly normal assets is
normal, so `basket_bachelier` has a closed-form price, gradient and
Hessian:

```
sigma_B = sqrt(w' diag(sigma) C diag(sigma) w)
V       = exp(-rT) [ (F_B - K) Phi(d) + sigma_B sqrt(T) phi(d) ]
dV/dS_i = w_i Phi(d)          d2V/dS_i dS_j = w_i w_j Gamma
```

The spot Hessian is therefore **rank one**. Under Black-Scholes the
analogous sum of lognormals has no closed form and the repository falls
back to a Monte Carlo reference. This makes `basket_bachelier` the better
testbed: a surrogate can be measured against ground truth rather than
against another estimator.

**Euler is exact.** The coefficients are constant, so an Euler increment
*is* the true increment. `num_steps` changes only the path resolution an
Asian payoff sees, never the accuracy of a European one.

The trade-off is that a normal underlying can go negative, which is
unrealistic for equities but is exactly why the model is standard for
rates and spreads.

### The forward convention, and why it matters

`bachelier` quotes from the spot and carries the drift in `F_0 = S exp(rT)`.
Dropping that carry is not a harmless simplification: with `F_0 = S` the
model under-prices, and the calibrator compensates by driving the fitted
rate **negative** (measured at −0.026 against a Black-Scholes fit of
+0.041 on the same chain). Carrying the forward restores +0.038 and keeps
the `r` feature interpretable. The convention is recorded under
`assumptions` in every archived run.

**The basket correlation is an assumption, not a fit.** The option chain is
single-name, so no instrument in it constrains rho. `calibrate_basket`
records it under `assumptions` in `config.json` and in the report, next to
but clearly separated from what was fitted. The one testable consequence —
that a basket is worth no more than its rho = 1 limit — is checked by the
validation stage.

### Heston, and why the variance scheme was chosen by measurement

    dS = r S dt + sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2

The variance can reach zero, and below the Feller condition
`2 kappa theta / xi^2 < 1` it does so with probability one. Every
discretisation therefore needs a positive part before `sqrt(v)`, and that
choice is not cosmetic for Sobolev training: the labels differentiate
through it twice.

Four candidates were measured against the Fourier price, which is exact up
to quadrature and itself differentiable:

| Feller ratio | scheme | dV/dv0 error | d2V/dS2 error |
|---|---|---|---|
| 4.00 | truncation `max(v,0)` | 0.36 % | 1.25 % |
| 4.00 | smooth | 0.36 % | 1.25 % |
| 0.64 | truncation | 2.06 % | 0.50 % |
| 0.64 | **smooth** | **0.55 %** | 0.41 % |
| 0.16 | truncation | **169 206 %**, sign flipped | 0.82 % |
| 0.16 | reflection `abs(v)` | **13 195 %** | 9.24 % |
| 0.16 | **smooth** | **1.11 %** | **0.58 %** |
| 0.16 | Andersen QE | **nan** | 52.6 % |

Hard truncation returned `dV/dv0 = -92730` against a true `54.8`, and
reflection `-7181`.

**The mechanism is the unbounded derivative of the square root, not the
kink in the positive part.** The pathwise derivative passes through
`d sqrt(v)/dv = 1/(2 sqrt(v))`, which diverges as the variance approaches
zero:

| v | `max(v,0)` | `abs(v)` | smooth |
|---|---|---|---|
| 1e-3 | 15.81 | 15.81 | 14.96 |
| 1e-6 | 500.0 | 500.0 | 17.70 |
| 0 | 2.5e+149 | 5.0e+149 | 17.68 |

`abs(v)` is smooth away from a single point and fails just as badly, which
is what rules out smoothness alone as the explanation: it still *reaches*
zero. QE fails for a third reason — its branch selection is not
differentiable and its exponential branch sets `v = 0` outright.

The repository therefore uses

    v+ = 0.5 (v + sqrt(v^2 + w^2)),   w = 0.01 theta

whose two relevant properties are separate. It is **bounded away from
zero**, flooring the variance at `w/2` so the derivative cannot exceed
`1 / (2 sqrt(w/2)) = 35.4` — this is what fixes the gradient. And it is
**twice differentiable**, its second derivative a bump of height `1/(2w)`
rather than a delta — this is what the Hessian-vector labels need. A
floored `max(v, eps)` would give the first without the second.

It costs almost nothing in price: bias `+0.032` against `+0.023` for
truncation on the same case, both under half a percent.
`tests/test_heston.py` asserts both properties so hard truncation cannot
silently return.

The choice is not a corner case: **51.7 % of the shipped Heston training
domain violates the Feller condition** (ratio range 0.352 to 3.121, median
0.984), so most samples sit in the regime where truncation fails.

The basket variant has **no** closed form: a sum of Heston assets has no
tractable characteristic function. Its correlation is the Kronecker product
`[[1, rho], [rho, 1]] (x) C`, positive semi-definite by construction.

## What was adopted from diff-ml

| Adopted | Why | Problem it solves |
|---|---|---|
| Antithetic sampling (`Bachelier.antithetic_payoff`) | same law, negatively correlated pairs | halves label variance at no cost, improving every Sobolev target |
| The Bachelier call formula and its Greeks (`Bachelier.Call.price/delta/gamma/vega`, eq. (3) of arXiv:2104.08686) | already correct and referenced | avoids re-deriving the normal-model analytics; extended here with discounting, puts and the basket |
| Verifying AD against a closed-form differential inside the sampler (`assert jnp.allclose(differentials_analytic, differentials_vjp)`) | catches a broken derivative path immediately | became the finite-difference and shape tests across all four models |
| Drawing a fresh key per `sample()` call | per-sample randomness is the intended JAX idiom | independent confirmation of the label-key fix |

Not adopted: `Normalization`/`Denormalization`/`Normalized` (equivalent to
`SurrogateModel`), `ad.hvp` (equivalent to ours), `sigmoidal_smoothing`
(equivalent to `payoff.sigmoid_smooth`), and `losses.sobolev`'s `loss_balance`
(ours is the same convex combination, stated explicitly in the config).
