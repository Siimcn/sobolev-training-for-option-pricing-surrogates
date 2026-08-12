# pipeline

`pipeline/` is the orchestration layer. It owns the *order* of the experiment
and nothing else: every stage takes a `PricingProblem` and asks it what to do,
so no module here knows which model is being priced.

`main.py` is the whole experiment, read top to bottom:

```
load_and_calibrate  ->  build_problem  ->  build_dataset  ->  split_dataset
                    ->  build_surrogate -> train_surrogate -> evaluate_surrogate
                    ->  validate_surrogate -> run_risk_analysis -> save_artifacts
```

## The modules

| module | what it does |
|---|---|
| `config.py` | every knob, as frozen dataclasses. The only file to edit to change a run. |
| `market.py` | fetches the option chain and calibrates the model to it |
| `data.py` | samples the domain and prices the Sobolev labels |
| `model.py` | builds the network and the per-dimension derivative scales |
| `training.py` | runs `SobolevTrainer.fit` |
| `evaluation.py` | test-set metrics and the per-dimension Greek breakdown |
| `reporting.py` | plots and the console report |
| `risk.py` | the exposure profile and XVA |

## config.py is the control panel

`ExperimentConfig` nests one dataclass per concern. Three fields decide what a
run *is*:

```python
data.pricing_model   # which of the six models  -> available_problems()
payoff.name          # what is being priced     -> available_payoffs()
data.sobolev_order   # 0 price, 1 +gradient, 2 +curvature
```

Two settings are easy to get wrong and are commented in place:

- **`simulation` vs `heston` path budgets.** Models with an exact terminal law
  (Black-Scholes, Bachelier and their baskets) read `SimulationConfig`. The two
  Heston problems step through time, so their cost scales with
  `num_paths * num_steps` and they read `HestonConfig` instead. The switch is
  the `_budget` property on the problem, not an `if` anywhere in this package.
- **`simulation.shared_label_keys`.** Leave it `False`. `True` prices the whole
  dataset from one random stream, which shifts every label the same way; the
  error then does not average out over the dataset. It exists only so that bug
  stays reproducible in a test.

## What the stages assume

Each optional `PricingProblem` method returning `None` switches its stage off
rather than failing:

| returns `None` | consequence |
|---|---|
| `reference_price` | no independent validation; the run says so |
| `analytic_price` | no closed-form check against the Monte Carlo reference |
| `arbitrage_bounds` | no no-arbitrage diagnostic |
| `comonotonic_limit_price` | no diversification check |
| `exposure_paths` | the risk stage is skipped |
| `underlying_paths` | no training-path preview plot |

This is deliberate - a new model is usable before it is complete - but it has a
sharp edge worth knowing: **a method you forgot to implement looks exactly like
one you deliberately left out.** Check the run's console output lists the
diagnostics you expect.

## Reproducibility

A run is determined by `config.data.seed` (domain sample),
`config.simulation.label_seed` (Monte Carlo labels), `config.network.seed`
(initial weights), `config.training.seed` (batch shuffling) and
`config.validation.seed` (the independent benchmark, deliberately unrelated to
the label seed).

`tests/test_reproducibility.py` hashes the domain, labels, gradients, HVPs and
trained weights for all six problems and compares against recorded values. Run
it after any change that is meant to preserve behaviour:

```
pytest tests/test_reproducibility.py -q
```

It finishes in well under a minute. A changed hash means the numbers moved,
whatever the rest of the suite says.
