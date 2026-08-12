# Adding a pricing model

Two edits. Write a module in `surrogate_modeling/problems/`, and import it from
that package's `__init__.py`. Nothing in `pipeline/`, `risk_visualisierung/` or
`main.py` needs to change - those stages take a `PricingProblem` and ask it,
they never ask which model it is.

## 1. The market model, if it is new

`marktsimulation/pricing_model.py` holds the dynamics. A `PricingModel`
subclass needs:

```python
class MyModel(PricingModel):
    def drift(self, t, state, params): ...
    def diffusion(self, t, state, params): ...

    def terminal_dispersion(self, s0, params, maturity):
        """Standard deviation of whatever the payoff is written on, at T.
        Sets the payoff-smoothing width."""

    # optional: only if the terminal law is known in closed form, which
    # makes a label ~60x cheaper than stepping to it
    def terminal_state(self, s0, params, maturity, num_paths, key, antithetic=True): ...

    # optional: for multi-asset states, says what the payoff is written on
    def basket_value(self, state, params): ...
```

If you skip `terminal_state`, `mc_price` steps through time with the scheme.
If you have a multi-dimensional state and skip `basket_value`, `mc_price`
raises rather than guessing which component to price.

## 2. The problem

Create `surrogate_modeling/problems/my_model.py`:

```python
MY_MODEL = "my_model"


def calibrate_my_model(config, market_data) -> CalibrationResult:
    ...
    return CalibrationResult(
        params=fitted,
        converged=converged,
        diagnostics=calibration_residuals(pricing_fn, fitted, market_data),
        assumptions={},   # anything the market data does NOT determine
    )


class MyModelProblem(MonteCarloProblem):
    """One sentence on what this prices."""

    name = MY_MODEL

    def __init__(self, market_data, calibration, config):
        self.market_data = market_data
        self.calibration = calibration
        self.config = config
        self.params = calibration.params
        self.payoff = config.payoff
        self.simulation = config.simulation
        self.data = config.data
        self.model = MyModel(scheme=EulerMaruyama())

    @property
    def feature_names(self):
        return ("S", "K", "T")

    def sample_features(self, u):
        """Map a uniform (n, n_features) block onto the training domain.
        Must be increasing in u, so feature_bounds can read the corners."""

    def _price(self, x, key, num_paths):
        """One Monte Carlo price at feature row x, twice differentiable in x."""


register_problem(ProblemSpec(MY_MODEL, MyModelProblem, calibrate_my_model))
```

Then add the import to `surrogate_modeling/problems/__init__.py`. That is the
second edit, and the last required one.

`MonteCarloProblem` supplies `discount_rate`, `exposure_strikes`,
`label_price_fn` and `reference_price` from your `_price`. If your model steps
through time and needs its own path budget, override `_budget` the way the
Heston problems do:

```python
    @property
    def _budget(self):
        return self.config.heston
```

## 3. Optional methods - each one switches a stage on

The base class returns `None` for all of these, which switches the
corresponding stage **off**. Implement what your model can support:

| method | switches on |
|---|---|
| `analytic_price` | closed-form check of the Monte Carlo reference |
| `arbitrage_bounds` | the no-arbitrage diagnostic |
| `shape_constraints` | sign checks on the gradient (cheap, model-free) |
| `comonotonic_limit_price` | the diversification check, for baskets |
| `exposure_paths` + `exposure_strikes` | the XVA stage |
| `underlying_paths` | the training-path preview plot |
| `exchangeable_features` | the permutation-symmetry diagnostic |
| `surface_specs` | which 2-D slices get plotted, if the default is wrong |

**The sharp edge:** a method you forgot looks identical to one you left out on
purpose. After the first run, check the console lists every diagnostic you
expected.

## 4. Config, only if you need new knobs

A model with its own settings gets a frozen dataclass in `pipeline/config.py`
and a field on `ExperimentConfig`, then reads `config.my_model` in its
constructor. Most models need nothing here.

## 5. Verify

```
pytest tests/test_reproducibility.py -q
```

Add your model to `EXPECTED` in that file by running it as a script. The other
five hashes must not move - if they do, you changed shared code.

Then a real run:

```
python main.py
```

Check the printed feature names, the sampled domain, the calibration residuals
and which diagnostics ran.

## Worked examples, easiest first

| file | why read it |
|---|---|
| `problems/black_scholes.py` | the simplest: exact terminal law, closed form to check against |
| `problems/bachelier.py` | shows `state_fn`, converting spot to forward before pricing |
| `problems/heston.py` | no closed terminal law: steps through time, own `_budget`, Fourier reference |
