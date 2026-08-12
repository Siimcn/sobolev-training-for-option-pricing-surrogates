# Market simulation

## Overview

The `market_simulation` package contains the core for numerical option pricing in this project. Its purpose is to connect the stochastic models, time-stepping schemes, payoff definitions, and Monte Carlo pricing into a single workflow that can be used for analytic benchmarks and (the main focus of this project) for training surrogate models.

The package has three main layers:

1. Model definitions for the underlying stochastic dynamics
2. Numerical schemes for simulating Monte Carlo paths through time
3. Payoff and pricing functions that calculate option values from the simulated states

This enables the project to compare analytical pricing formulas with Monte Carlo estimates and to generate training data for the surrogate pricing models.

---

## Module Workflow

The workflow is structured as follows:

```text
User Input / Model Parameters
        │
        ▼
PricingModel
        │
        ├── drift(state, params, t)
        ├── diffusion(state, params, t)
        └── noise_correlation(params)
        │
        ▼
TimeSteppingScheme
        │
        ├── EulerMaruyama
        └── Milstein
        │
        ▼
MonteCarloPricer / mc_price
        │
        ├── simulate paths
        ├── evaluate payoff
        └── compute discounted expectation
        │
        ▼
Option Price / Greeks as Training Data
```

For European options, the payoff is computed at the terminal state. For path-dependent products such as Asian options, the payoff is evaluated on the path itself and then combined for many Monte Carlo paths.

---

## timesteppingscheme.py

### Purpose

This module is used to propagate underlying states under a stochastic differential equation. It contains the abstract `TimeSteppingScheme` base class and concrete implementations for Euler-Maruyama and Milstein schemes.

### Key Components

- `TimeSteppingScheme`: base class for SDE solving
- `generate_paths(...)`: creates Monte Carlo paths using correlated Brownian increments
- `EulerMaruyama.step(...)`: first-order discretisation
- `Milstein.step(...)`: second-order correction for the diffusion term

### Functionality

The key idea is that a model only supplies the drift and diffusion functions, while the scheme decides how to evolve the state over time. This keeps model separated from the numerical solver and makes it possible to swap schemes for accuracy or performance.

### Notes

The path generator creates Gaussian increments of the form:

- `dW ~ N(0, dt)`

and optionally applies a Cholesky factorisation of the correlation matrix so that multi-dimensional Brownian motions are correlated appropriately.

---

## pricing_model.py

### Purpose

This file defines the stochastic models used in the project. Each model is a subclass of `PricingModel` with different drift, diffusion, and optionally correlation structure for the chosen financial dynamics.

### Model Types

The file implements several pricing models:

- `BachelierModel`: normal model with constant volatility
- `BlackScholesModel`: standard geometric Brownian motion
- `HestonModel`: stochastic volatility model with variance process
- `BasketBlackScholesModel`: multi-asset basket with correlated Black-Scholes
- `BasketBachelierModel`: multi-asset basket under the normal model
- `BasketHestonModel`: multi-asset basket with stochastic volatility

### Parameter Structures

Each model has an associated parameter class using `NamedTuple`:

- `BachelierParams`
- `BlackScholesParams`
- `HestonParams`
- `BasketBlackScholesParams`
- `BasketBachelierParams`
- `BasketHestonParams`

These parameter classes hold the market quantities required by the model, such as interest rate, volatility, mean reversion speed, long-run variance, and asset correlations.

### Design

The model layer is intentionally generic and modular. The current model determines drift and diffusion, while the time-stepping scheme is reusable across all models.

---

## payoff.py

### Purpose

This module defines the payoff functions and smooth approximations used inside Monte Carlo pricing. It is designed to support both plain european payoffs and path-dependent payoffs.

### Main Components

- `Payoff`: abstract payoff base class
- `EuropeanPayoff`: payoff depending on a terminal value
- `AsianPayoff`: payoff depending on the path average
- `EuropeanCall`, `EuropeanPut`
- `AsianCall`, `AsianPut`
- payoff functions:
  - `relu`
  - `sigmoid_smooth`
  - `cubic_spline_smooth`

### Smoothing

The project uses smooth approximations of the usual payoff functions to ensure differentiability and numerical stability in the surrogate-modeling. This is especially important when gradients are computed during pricing.

---

## monte_carlo_pricer.py

### Purpose

This is the central Monte Carlo pricing engine. It evaluates option prices by simulating paths under a chosen model and then applying a payoff function to the relevant state variable.

### Key Responsibilities

- generate state paths under a model and stepping scheme
- evaluate either terminal or path-dependent values
- apply the chosen payoff function
- compute the Monte Carlo expectation
- return the discounted or undiscounted price depending on the caller

### Functionality

The `MonteCarloPricer` takes:

- a model
- a payoff object
- an optional `value_fn`
- whether the payoff is evaluated on the path or at maturity

It then calls the model's `scheme.generate_paths(...)` method and calculates the mean payoff across all simulated paths.

---

## black_scholes.py

### Purpose

This module contains closed-form Black-Scholes formulas for prices and Greeks. It is one of the analytic benchmark modules in the project and is used to validate Monte Carlo results.

### Functions

- `black_scholes_price(...)`: vectorised price across strikes and maturities
- `black_scholes_price_single(...)`: scalar option price
- `delta(...)`, `gamma(...)`, `vega(...)`: option Greeks
- `bs_feature_price(...)`: feature-based pricing interface
- `bs_feature_gradient(...)`, `bs_feature_hessian(...)`: autodiff-based derivatives
- `create_bs_dataset(...)`: produces analytic datasets used in surrogate modeling

---

## black_scholes_mc.py

### Purpose

Black-Scholes-specific Monte Carlo helpers that do not belong in the generic
pricer: exact terminal sampling, the calibration pricer and training-path
generation for the preview plots. Useful for validating against the analytic
counterpart.

### Functions

- `simulate_terminal(...)`: exact terminal price simulation under geometric Brownian motion
- `make_mc_calibration_pricer(...)`: constructs a pricing function for calibration procedures
- `generate_training_paths(...)`: creates training paths for surrogate learning

!!! note
    This module used to also hold `bs_mc_price(...)`, a second Monte Carlo
    pricer for Black-Scholes alone. It was removed once it was shown to be
    bitwise equivalent to `mc_price(...)`; all six models now price through
    the generic path. See `mc_pricing.py` below.

---

## heston.py

### Purpose

This module implements semi-analytic European option pricing with the Heston stochastic-volatility model. Instead of a pure Monte Carlo simulation, it uses a Fourier-inversion approach based on the characteristic function of the log-price process.

### Functions

- `heston_characteristic(...)`: characteristic function of the log-price under Heston dynamics
- `_probability(...)`: probability calculation via Gil-Pelaez inversion
- `heston_price(...)`: European option price by Fourier inversion
- `heston_price_vector(...)`: vectorised multi-quote pricing
- `feller_ratio(...)`: diagnostic to describe whether the variance process is away from the zero-boundary

---

## bachelier.py

### Purpose

Calculates prices and Greeks for the Bachelier model, which is a normal-model analogue of Black-Scholes.

### Functions

- `bachelier_price(...)`: vectorised Bachelier price
- `bachelier_price_single(...)`: scalar price
- `bachelier_delta(...)`, `bachelier_gamma(...)`, `bachelier_vega(...)`: Greeks
- `basket_normal_volatility(...)`: weighted basket volatility under a normal model
- `basket_bachelier_price(...)`: exact basket price
- `basket_bachelier_greeks(...)`: basket Greeks and Hessian

---

## mc_pricing.py

### Purpose

**The Monte Carlo pricing path. Every one of the six pricing problems goes
through it** — there is no model-specific pricer any more. Given any
`PricingModel` and a payoff name it produces one price, differentiable twice
in the feature row.

### Functions

- `mc_price(...)`: generic Monte Carlo price for any model and payoff specification
- `make_feature_price(...)`: builds a pricing function in the feature-layout style used by surrogate training
- `_rate(...)`: extracts the rate field from model parameters if present

### How it decides what to do

| decided by | how |
|---|---|
| exact draw or step through time | the model: `terminal_state` if it exists, otherwise the scheme |
| what the payoff is written on | the model's `basket_value`, or an explicit `value_fn` |
| the payoff-smoothing width | the model's `terminal_dispersion`, times `payoff.smooth_fraction` |
| terminal or path-dependent | the payoff spec |

A model with a multi-dimensional state that exposes neither `basket_value` nor
a `value_fn` raises, rather than silently pricing the first component.

---

## basket_mc.py

### Purpose

This module implements Monte Carlo pricing and Greeks for basket options. It is designed for multi-asset underlyings where the payoff depends on a weighted sum of asset values.

### Functions

- `basket_price(...)`: Monte Carlo price for a basket option
- `basket_greeks(...)`: computes pathwise Greeks using autodiff

### Use Case

Basket options are common in portfolio-valued settings and are one of the main examples in which the underlying state is multi-dimensional. This module therefore bridges the generic model layer with a realistic payoff structure.

---

## param_validation.py

### Purpose

Sanity checks on model parameters before anything is priced with them: a
volatility that is zero or negative, a correlation outside [-1, 1], basket
weights that do not sum to one, a correlation matrix that is not symmetric.

### Design

Rules are keyed by **field name**, not by parameter class. The six parameter
types share their fields - `sigma`, `kappa`, `weights`, `corr` and the rest
recur across models - so one rule per field covers every model that uses it,
and a seventh model is validated the moment it reuses a field name.

- `FIELD_RULES`: the field-name -> rule mapping
- `validate_params(params)`: applies every rule that matches a field
- `ValidationError`: raised on a violation

### Where it runs

At calibration entry and exit (`Calibrator.calibrate`). It compares concrete
numbers with Python `if`, so it **cannot run inside `jit`, `grad` or `vmap`**;
called on a traced value it says so explicitly rather than failing later with
a `ConcretizationError`.

Two things are deliberately not checked:

- **`r`** — negative interest rates are real, and a calibration that finds one
  is reporting the market, not failing.
- **The Feller condition** (`2*kappa*theta >= xi^2`) — violating it is
  legitimate, the variance simply reaches zero and the smooth positive-part
  scheme handles it. Roughly half of the shipped Heston sampling domain sits
  below it by design. The ratio is reported at calibration instead.

---
