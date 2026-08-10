# Pricing problems

A `PricingProblem` is the answer to "what is this surrogate pricing?".

It owns every fact that depends on the model: the feature layout, the
sampling domain, the label pricer, which slices are worth plotting, how a
trained surrogate is independently validated, and how future states are
simulated for an exposure profile.

The pipeline stages take a problem and ask it. None of them branches on
which model is configured, so adding a model does not mean editing the
data generator, the plotting layer, the validation stage and the risk
stage in turn.

```python
config.data.pricing_model = "basket_black_scholes"
```

is the whole switch.

## Adding a model

Two steps: implement, then register.

```python
from surrogate_modeling.pricing_problem import PricingProblem, register_problem


class HestonProblem(PricingProblem):

    name = "heston"

    @property
    def feature_names(self):
        return ("S", "K", "T", "v0", "kappa", "theta", "xi", "rho")

    def sample_features(self, u):
        # map a uniform (n, 8) block onto the training domain
        ...

    def label_price_fn(self):
        # f(x) -> price, twice differentiable, random numbers fixed inside
        ...


register_problem("heston", lambda config, market_data, fitted_params: HestonProblem(...))
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
| `underlying_paths` | `None` | the training-path preview |
| `reference_price` | `None` | the independent Monte Carlo benchmark |
| `analytic_price` | `None` | a closed-form column, and an XVA reference |
| `arbitrage_bounds` | `None` | a model-free sanity check |
| `exposure_paths` | `None` | the risk stage |
| `exposure_strikes` | none | which strikes the exposure is reported at |

A stage whose optional method returns `None` prints that it is skipping
and why. It never falls back to a guess. That is the point: the previous
design assumed `[S, K, T, sigma, r]` in the plotting and risk layers, so a
basket run produced surfaces titled "Volatility σ" that were really
sweeping the strike over 23 to 35 cents, at a third spot of fifty cents and
a maturity below the training floor. Nothing failed; the output was simply
wrong.

## Where the numbers come from

`feature_bounds` is read off `sample_features` by evaluating it at `u = 0`
and `u = 1`, rather than being declared separately. A plot range and the
range the labels were drawn from therefore cannot drift apart. This
requires `sample_features` to be coordinate-wise increasing in `u`, which
every uniform-domain map is.

`baseline_features` defaults to the domain midpoint, so a new problem's
surface anchor is in-domain without the author having to say where that
is.

## Validation

`surrogate_modeling.validation.run_reference_validation` is model-agnostic
and runs for every problem:

- **Fresh Monte Carlo.** `reference_price` re-prices the surrogate's own
  reference points with a seed unrelated to the labels, so agreement is
  not memorised noise.
- **Closed form, where one exists.** Comparing the surrogate against an
  analytic price alone cannot separate "learned the MC operator" from
  "learned the formula". Reporting MC-vs-analytic alongside makes the
  comparison meaningful: the surrogate should be no further from MC than
  MC is from its own benchmark.
- **No-arbitrage bounds.** A call is worth at least its discounted
  intrinsic value and never more than the underlying.
- **Exchangeability.** Where the true price is invariant under permuting a
  group of features, the surrogate should be too. Nothing enforces this
  architecturally, so the residual asymmetry is a real error the pooled
  metrics hide.

The results are written to `metrics.json` and `report.txt` next to the
training metrics.

## Built-in problems

| Name | Features | Closed form | Exchangeable |
|---|---|---|---|
| `black_scholes` | `S, K, T, sigma, r` | yes | no |
| `basket_black_scholes` | `S_1 … S_n, K, T` | no | the spots, when the basket is |

A basket surrogate has per-asset deltas and gammas but no vega or rho: the
basket structure — weights, correlation, per-asset vols and the rate — is
fixed at construction rather than carried in the feature vector.
