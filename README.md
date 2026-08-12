# Introduction

This repository contains a **JAX/Equinox library for Higher-Order Sobolev Training of
option-pricing surrogates**, developed as software practicum SWP Gruppe 7.

A surrogate is a neural network trained to replace a Monte Carlo pricer. What makes the
training *Sobolev* is that the network is fitted not only to prices but to their
derivatives as well — first order (the Greeks) and second order (curvature, via
Hessian-vector products) — so that a single forward pass returns a price and a risk
profile that are consistent with each other.

The pipeline runs end to end:

| Stage | Role |
|---|---|
| **Calibration** | Fits a market model to a real option chain (AAPL, via Yahoo Finance) |
| **Label generation** | Prices a sampled domain by Monte Carlo, with derivatives by automatic differentiation |
| **Sobolev training** | Fits price + gradient + Hessian-vector products in one objective |
| **Validation** | Re-prices against an independent benchmark; checks arbitrage bounds, shape constraints and symmetry |
| **Risk** | Exposure profiles, CVA/DVA/XVA from the trained surrogate |

## Objective

Monte Carlo pricing is accurate and slow. A risk system that has to re-price a book
thousands of times per day cannot afford it. The usual answer is a surrogate — but a
surrogate fitted to prices alone produces derivatives that are noise, and derivatives are
exactly what a risk system needs.

Sobolev training addresses this by putting the derivatives into the loss. Following
Savine's *Differential Machine Learning*, the objective is a convex combination

```
L = α · L_price + β · L_gradient + γ · L_hvp,     α + β + γ = 1
```

The library exists to make that claim **testable rather than asserted**: every model is
measured against an independent reference, and where a closed form exists, against exact
mathematics.

## Audience

Supervisors and developers picking the project up without having written it. Familiarity
with option pricing helps but the documentation defines its terms; familiarity with JAX
is assumed only in `marktsimulation` and `surrogate_modeling`.

## Supported models

Six pricing problems are registered. `black_scholes` is the one to read first.

| `data.pricing_model` | Features | Closed form | Independent reference |
|---|---|---|---|
| `black_scholes` | S, K, T, σ, r | yes | analytic |
| `basket_black_scholes` | S1..Sn, K, T | no | Monte Carlo + comonotonic bound |
| `bachelier` | S, K, T, σ, r | yes | analytic |
| `basket_bachelier` | S1..Sn, K, T | **yes** (a normal basket stays normal) | analytic |
| `heston` | S, K, T, v0, κ, θ, ξ, ρ | Fourier | Fourier inversion |
| `basket_heston` | S1..Sn, K, T | no | its own Monte Carlo only |

Payoffs: `european_call`, `european_put`, `asian_call`, `asian_put`.

---

# Getting Started

## 1. Installation process

Prerequisites: **Python 3.12 or newer** (pinned to 3.12 in `.python-version`, developed
and tested with 3.13.2).

The project uses [uv](https://docs.astral.sh/uv/) and ships a `uv.lock`:

```bash
uv sync
```

Or with plain pip:

```bash
python -m pip install -e . && python -m pip install black pytest
```

Then run the pipeline:

```bash
python main.py
```

That is the whole entry point. It reads `ExperimentConfig`, runs every stage and writes a
timestamped directory under `results/`.

## 2. Software dependencies

| Package | Version used | Purpose |
|---|---|---|
| `jax` | 0.11.0 | Arrays, automatic differentiation, JIT |
| `equinox` | 0.13.8 | Neural network layers as JAX PyTrees |
| `optax` | 0.2.8 | Adam, gradient clipping, learning-rate schedules |
| `optimistix` | 0.1.0 | Levenberg–Marquardt, used for calibration |
| `matplotlib` | 3.10.1 | Diagnostic plots |
| `yfinance` | 0.2.66 | Option-chain download (cached, see below) |
| `mkdocs` | 1.6.1 | Documentation site |
| `black` | 26.5.1 | Formatting (dev) |
| `pytest` | 9.1.1 | Tests (dev) |

> **Double precision is mandatory.** `main.py` sets `jax_enable_x64` before anything else,
> and `tests/conftest.py` does the same. The Hessian-vector labels are not usable in
> float32 — and because the flag must be set *before* the first array is created, a module
> that builds arrays at import time will silently get float32 if it is imported first.

> **The option chain is cached.** `data/aapl_chain.json` is a snapshot (1472 instruments,
> fetched 2026-08-07) so runs are reproducible and work outside US market hours. Delete
> the file or set `market.use_cache = False` to refetch.

## 3. Latest release

| Version | Date | Content |
|---|---|---|
| **0.2** | 12/08/2026 | Bachelier, Basket Bachelier, Heston and Basket Heston added behind the `PricingProblem` abstraction. Three parallel Monte Carlo paths consolidated into one. Reproducibility guard added. Calibration now reports fit residuals, not just convergence. See `CHANGELOG.md` for the breaking artifact-schema change. |
| 0.1 | — | Black-Scholes and Basket Black-Scholes, Sobolev training, XVA |

## 4. Reference documentation

- Documentation site: `documentation/` — `cd documentation && python -m mkdocs serve`
- **[Adding a pricing model](documentation/docs/code_structure/adding_a_pricing_model.md)** — start here to extend the library
- **[Pipeline](documentation/docs/code_structure/pipeline.md)** — the stage order and every configuration knob
- `CHANGELOG.md` — breaking changes and their reasoning
- `Report/report.tex` — the written report (German)
- `diff-ml-main/` — the supervisor's reference implementation. **Read-only. Do not modify.**
- Savine & Huge, *Differential Machine Learning* (2020), the method this builds on

---

# Build and Test

## Repository layout

```
.
├── main.py                        the entire pipeline, read top to bottom
├── pyproject.toml                 dependencies, black config
├── CHANGELOG.md
│
├── pipeline/                      orchestration and ALL configuration
│   ├── config.py                  every knob - the file you edit to change a run
│   └── market.py  data.py  model.py  training.py
│       evaluation.py  reporting.py  risk.py
│
├── marktsimulation/               the mathematics: models, payoffs, Monte Carlo
│   ├── pricing_model.py           dynamics for all six models
│   ├── mc_pricing.py              THE Monte Carlo pricer - all six go through it
│   ├── black_scholes.py  bachelier.py  heston.py      closed forms
│   └── payoff.py  timesteppingscheme.py  basket_mc.py
│
├── surrogate_modeling/
│   ├── pricing_problem.py         the contract: PricingProblem, MonteCarloProblem
│   ├── problems/                  one module per model family
│   │   ├── black_scholes.py  bachelier.py  heston.py
│   ├── sobolev_trainer.py         the training loop
│   ├── losses.py  metrics.py  validation.py  domain.py
│
├── kalibrierung/                  market data and calibration
├── risk_visualisierung/           exposure, XVA, plots
├── tests/                         278 tests
├── documentation/                 mkdocs site (docs/ is source, site/ is generated)
├── results/                       GENERATED - one directory per run
└── diff-ml-main/                  supervisor's repository - DO NOT MODIFY
```

## Build

There is no build step — it is a library plus an entry point. A run is:

```bash
python main.py
```

Runtime depends on the model: Black-Scholes and Bachelier draw their terminal value
exactly and finish in minutes; the Heston models step through time and take roughly 30–35
minutes at the shipped budget.

Each run writes a timestamped directory under `results/` containing `config.json`
(everything needed to reproduce it), `metrics.json`, `calibration.json`, `xva.json`,
`best_model.eqx`, `report.txt`, `console_output.txt` and the diagnostic plots.

To change what is run, edit `pipeline/config.py` — normally just:

```python
data.pricing_model = "heston"      # one of the six
payoff.name        = "european_call"
data.sobolev_order = 2             # 0 price, 1 +gradient, 2 +curvature
```

## Test

```bash
python -m pytest tests/ -q
```

**278 tests, about 6 minutes.** Before that, run the guard — it is the informative one and
takes under a minute:

```bash
python -m pytest tests/test_reproducibility.py -q
```

`tests/test_reproducibility.py` hashes the sampled domain, the Monte Carlo labels, their
gradients, their Hessian-vector products and the trained weights, for **all six problems**,
and compares against recorded values. It is the only check that catches a change in the
*numbers* rather than in behaviour.

| Check | How |
|---|---|
| Numbers unchanged | `pytest tests/test_reproducibility.py -q` — all six hashes match |
| Tests pass | `pytest tests/ -q` — 278 passed |
| Formatting | `python -m black --check .` — no diff |
| Full run works | `python main.py` — check the console lists the diagnostics you expect |

CI (`.github/workflows/ci.yml`) runs all four on push and pull request.

> **A changed hash is not a test failure to be "fixed".** It means the numerical output
> moved. Either the change was not behaviour-preserving, or it was intended — in which
> case regenerate the baseline by running the file as a script and **write down why** in
> the comment next to the changed entry, as was done for the one float64-rounding change
> already recorded there.

---

# Contribute

## Where to make changes

| To change… | Edit |
|---|---|
| what a run does | `pipeline/config.py` |
| add a pricing model | a new module in `surrogate_modeling/problems/` + its import in `__init__.py` |
| add a payoff | `marktsimulation/payoff.py`, then `register_payoff` |
| add a network architecture | `surrogate_modeling/architectures.py`, then `register_architecture` |
| the training objective | `surrogate_modeling/losses.py`, `sobolev_trainer.py` |
| documentation | `documentation/docs/` — never `documentation/site/`, which is generated |

**Never edit `diff-ml-main/`.** It is the supervisor's repository, included for reference.

Adding a model is deliberately a **two-edit** change and requires no modification to
`pipeline/`, `risk_visualisierung/` or `main.py`. If a change you are making needs edits
in those places, the abstraction is probably being bypassed — see
[Adding a pricing model](documentation/docs/code_structure/adding_a_pricing_model.md).

## Content rules

These exist because the project's output is a scientific claim. A wrong number here
becomes a wrong statement in the report.

1. **Never claim a model works until it has been run end to end** — training, evaluation,
   diagnostics, plots and reporting. A passing test suite is not a working model.
2. **Separate fitted from assumed.** Anything the market data does not determine goes in
   `CalibrationResult.assumptions`, where it is archived into `config.json`. The basket
   correlation is assumed, not calibrated; the report must not imply otherwise.
3. **Report the noise floor next to the error.** A surrogate error is meaningless without
   the label noise and the reference noise beside it. All three are printed by the
   validation stage.
4. **Do not present a violation count without its bias.** Arbitrage and diversification
   counts track the *sign of the price bias*, not economic soundness — a positive bias
   suppresses one and inflates the other.
5. **Prefer a slower verified implementation to a faster unverified one.**
6. **State what was not done.** A model without an independent reference must say so in
   its output; `basket_heston` does.

## Formatting conventions

**Formatting is not a matter of taste here — `black` decides**, and CI enforces it:

```bash
python -m black .
```

The configuration lives in `pyproject.toml`: line length 88, and
`skip-magic-trailing-comma = true`. That second setting matters — without it, a trailing
comma pins a call to one-argument-per-line forever, which is how ~2000 lines of the
codebase came to hold nothing but a closing bracket.

**Other conventions:**

- **Comments are sparse by design.** The rationale lives in `documentation/`. Comments are
  reserved for cases where a plausible edit would silently break something — see the
  handful left in `pipeline/config.py` for the intended density.
- **Docstrings** are one paragraph: what it does and why it is not the obvious
  alternative. Not a parameter list.
- **Naming** is English throughout the code, including inside the German-named packages
  (`kalibrierung`, `marktsimulation`, `risk_visualisierung`), whose names match the German
  report.
- **One convention per concept.** If you find two names for the same thing, unify them.
- **Randomness is an argument, never a closure.** Label pricers take `(x, key)`; each
  sample gets its own key. A shared key biases every label the same way and the error does
  not average out.
- **Registries over `if/else`.** Models, payoffs and architectures are registered by name.
  No stage should ever ask *which* model it has.

## Before submitting a change

1. `python -m pytest tests/test_reproducibility.py -q` — six hashes match, or the change
   is explained in the file.
2. `python -m pytest tests/ -q` — 278 passed.
3. `python -m black --check .` — clean.
4. New behaviour has a test; new numerical behaviour has a hash.
5. For anything touching labels, training or pricing: a real `python main.py` run, with
   the console output read.
6. Documentation updated if the architecture changed.

## Open points

Carried over from the last review, in priority order:

- **Seed replication has not been run.** Every cross-model comparison in the report is a
  single draw with no error bars. Five network seeds × three label seeds is the smallest
  credible version. **This gates all comparative claims.**
- **Heston's second-order labels for the model parameters are Monte Carlo noise.**
  Measured against exact Fourier derivatives, the label HVPs for `v0, κ, θ, ξ, ρ` score a
  relative L2 of 1.13 — worse than predicting zero. The trained surrogate scores 1.05, so
  it is *better than its labels*. The reported `HVP_Relative_L2 = 0.876` therefore measures
  the label pipeline, not the surrogate. Fixing it by brute force needs ~30× the paths;
  the cheap route is to use the exact Fourier price as the label for single-asset Heston.
- **`basket_heston` has no independent reference.** It is validated only against its own
  Monte Carlo, so no accuracy claim can be drawn from its runs. A comonotonic upper bound
  is derivable (perfect correlation collapses the basket onto a single name) and would be
  the cheapest improvement.
- **Two of the six models have no production run.** `bachelier` has never been run;
  `black_scholes` only pre-dates the validation stage. Both are fast — this is the easiest
  gap to close.
- **The surrogate price bias is the dominant error term**, not the noise — 76 % of the
  squared error for `basket_heston`. It is a regression-to-the-mean pattern from unweighted
  MSE over a 0–700 price range, not a constant offset, so it cannot be fixed by subtracting
  a number.
- `risk_visualisierung/visualizer.py` is a 426-line class with no test coverage, so the
  reproducibility guard cannot protect a refactor there.
