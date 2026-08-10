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
| `basket_black_scholes` | `S_1 … S_n, K, T` | no | the spots, when the basket is |

A basket surrogate has per-asset deltas and gammas but no vega or rho: the
basket structure is fixed at construction rather than carried in the feature
vector.

**The basket correlation is an assumption, not a fit.** The option chain is
single-name, so no instrument in it constrains rho. `calibrate_basket`
records it under `assumptions` in `config.json` and in the report, next to
but clearly separated from what was fitted. The one testable consequence —
that a basket is worth no more than its rho = 1 limit — is checked by the
validation stage.

## What was adopted from diff-ml

| Adopted | Why |
|---|---|
| Antithetic sampling (`Bachelier.antithetic_payoff`) | halves label variance at no cost, which directly improves every Sobolev target |
| Verifying AD against a closed-form differential inside the sampler (`assert jnp.allclose(differentials_analytic, differentials_vjp)`) | became the shape and finite-difference tests; catches a broken derivative path immediately |
| Drawing a fresh key per `sample()` call | independent confirmation that per-sample randomness is the intended idiom |

Not adopted: `Normalization`/`Denormalization`/`Normalized` (equivalent to
`SurrogateModel`), `ad.hvp` (equivalent to ours), `sigmoidal_smoothing`
(equivalent to `payoff.sigmoid_smooth`), and `losses.sobolev`'s `loss_balance`
(ours is the same convex combination, stated explicitly in the config).
