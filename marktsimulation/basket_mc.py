import jax
import jax.numpy as jnp

from marktsimulation.payoff import EuropeanCall, AsianCall, sigmoid_smooth
from marktsimulation.monte_carlo_pricer import MonteCarloPricer
from marktsimulation.pricing_model import (
    BasketBlackScholesModel,
    BasketBlackScholesParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama


def basket_price(
    model,
    params,
    s0,
    strike,
    maturity,
    key,
    num_paths: int = 50_000,
    num_steps: int = 50,
    asian: bool = False,
    smooth_w: float = 1.0,
): 

    payoff_fn = AsianCall if asian else EuropeanCall

    payoff = payoff_fn(
        strike=strike,
        smooth_fn=sigmoid_smooth,
        smooth_w=smooth_w,
    )

    pricer = MonteCarloPricer(
        model,
        payoff,
        value_fn=lambda s: model.basket_value(s, params),
        payoff_on_path=asian,
    )

    undiscounted = pricer.price(
        s0=s0,
        params=params,
        maturity=maturity,
        num_paths=num_paths,
        num_steps=num_steps,
        key=key,
    )

    return undiscounted * jnp.exp(-params.r * maturity)


def uniform_correlation(n_assets: int, rho: float) -> jnp.ndarray:
    """Correlation matrix with rho off the diagonal and ones on it."""

    return (
        jnp.full((n_assets, n_assets), rho)
        .at[jnp.diag_indices(n_assets)]
        .set(1.0)
    )


def make_basket_feature_price(
    weights: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    num_paths: int = 50_000,
    num_steps: int = 50,
    seed: int = 0,
    smooth_fraction: float = 0.05,
):
    """
    Build `f(x) -> price` for a basket call, with the feature layout

        x = [S_1, ..., S_n, K, T]

    The basket structure (weights, correlation, per-asset vols, rate) is
    fixed here rather than carried in x, mirroring how diff-ml's Bachelier
    example uses the spot vector alone as its input.

    The key is fixed so every label in a dataset shares its random
    numbers, exactly as bs_mc_price does.
    """

    n_assets = len(weights)

    model = BasketBlackScholesModel(scheme=EulerMaruyama())

    params = BasketBlackScholesParams(
        r=r,
        sigmas=sigmas,
        weights=weights,
        corr=corr,
    )

    key = jax.random.PRNGKey(seed)

    sigma_mean = jnp.mean(sigmas)

    def price_fn(x: jnp.ndarray) -> jnp.ndarray:
        s0 = x[:n_assets]
        strike = x[n_assets]
        maturity = x[n_assets + 1]

        # same construction as _payoff_smoothing_width, with the basket
        # value standing in for the single-asset spot
        basket0 = jnp.sum(weights * s0)
        dispersion = basket0 * sigma_mean * jnp.sqrt(jnp.maximum(maturity, 1e-6))
        smooth_w = jnp.maximum(smooth_fraction * dispersion, 1e-3)

        return basket_price(
            model,
            params,
            s0,
            strike,
            maturity,
            key,
            num_paths=num_paths,
            num_steps=num_steps,
            smooth_w=smooth_w,
        )

    return price_fn


def basket_greeks(
    model,
    params,
    s0,
    strike,
    maturity,
    key,
    num_paths: int = 50_000,
    num_steps: int = 50,
    asian: bool = False,
):
    """
    Price and per-asset deltas/gammas of a basket option.

    Pathwise AD through the whole Monte Carlo simulation; the key is
    fixed within one call, so all evaluations share common random
    numbers.
    """

    def price_fn(s0_):
        return basket_price(
            model, params, s0_, strike, maturity, key,
            num_paths=num_paths, num_steps=num_steps, asian=asian,
        )

    price, delta = jax.value_and_grad(price_fn)(s0)
    gamma = jax.hessian(price_fn)(s0)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
    }