"""
Shared test setup.

Double precision is switched on here rather than in each test module. The
pipeline runs in x64 (see `main.py`), and a test that ran in x32 would be
testing different arithmetic from the one that produces the results.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from marktsimulation.mc_pricing import make_feature_price, mc_price
from marktsimulation.pricing_model import (
    BasketBlackScholesModel,
    BasketBlackScholesParams,
    BlackScholesModel,
    BlackScholesParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama


def bs_mc_feature_price(
    x,
    key,
    payoff="european_call",
    num_paths=50_000,
    num_steps=50,
    smooth_fraction=0.05,
    antithetic=True,
):
    """
    Black-Scholes Monte Carlo price for the feature row [S, K, T, sigma, r].

    The same call `BlackScholesProblem._price` makes, so these tests exercise
    the code the pipeline actually runs.
    """

    return mc_price(
        BlackScholesModel(scheme=EulerMaruyama()),
        BlackScholesParams(r=x[4], sigma=x[3]),
        jnp.array([x[0]]),
        x[1],
        x[2],
        key,
        payoff=payoff,
        num_paths=num_paths,
        num_steps=num_steps,
        smooth_fraction=smooth_fraction,
        antithetic=antithetic,
    )


def basket_bs_feature_price(
    weights,
    corr,
    sigmas,
    r,
    payoff="european_call",
    num_paths=50_000,
    num_steps=50,
    smooth_fraction=0.05,
    symmetrize=False,
    antithetic=True,
):
    """
    `f(x, key) -> price` for a Black-Scholes basket, feature row
    [S1..Sn, K, T]. Mirrors `BasketBlackScholesProblem._pricer`.
    """

    return make_feature_price(
        BasketBlackScholesModel(scheme=EulerMaruyama()),
        BasketBlackScholesParams(r=r, sigmas=sigmas, weights=weights, corr=corr),
        n_assets=len(weights),
        payoff=payoff,
        num_paths=num_paths,
        num_steps=num_steps,
        smooth_fraction=smooth_fraction,
        symmetrize=symmetrize,
        antithetic=antithetic,
    )
