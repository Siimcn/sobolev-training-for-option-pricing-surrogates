import jax
import jax.numpy as jnp

from jax.scipy.stats.norm import cdf, pdf

from marktsimulation.pricing_model import (
    BlackScholesParams,
)


def black_scholes_price(
    params: BlackScholesParams,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    is_call: jnp.ndarray,
    spot: float,
) -> jnp.ndarray:
    """Vectorized Black-Scholes pricing."""

    r = params.r

    sigma = jnp.maximum(
        params.sigma,
        1e-8,
    )

    strikes = jnp.maximum(
        strikes,
        1e-8,
    )

    T = jnp.maximum(
        maturities,
        1e-8,
    )

    sqrt_T = jnp.sqrt(T)

    d1 = (
        jnp.log(spot / strikes)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_T)

    d2 = d1 - sigma * sqrt_T

    discount = jnp.exp(-r * T)

    call_prices = (
        spot * cdf(d1)
        - strikes * discount * cdf(d2)
    )

    put_prices = (
        strikes * discount * cdf(-d2)
        - spot * cdf(-d1)
    )

    return jnp.where(
        is_call,
        call_prices,
        put_prices,
    )


def black_scholes_price_single(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    is_call: bool = True,
):
    """Single European option price."""

    params = BlackScholesParams(
        r=r,
        sigma=sigma,
    )

    return black_scholes_price(
        params=params,
        strikes=jnp.array([strike]),
        maturities=jnp.array([maturity]),
        is_call=jnp.array([is_call]),
        spot=spot,
    )[0]


def delta(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    is_call: bool = True,
):

    sqrt_T = jnp.sqrt(
        jnp.maximum(maturity, 1e-8)
    )

    d1 = (
        jnp.log(spot / strike)
        + (r + 0.5 * sigma**2)
        * maturity
    ) / (sigma * sqrt_T)

    return jnp.where(
        is_call,
        cdf(d1),
        cdf(d1) - 1.0,
    )


def gamma(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
):

    sqrt_T = jnp.sqrt(
        jnp.maximum(maturity, 1e-8)
    )

    d1 = (
        jnp.log(spot / strike)
        + (r + 0.5 * sigma**2)
        * maturity
    ) / (sigma * sqrt_T)

    return (
        pdf(d1)
        / (
            spot
            * sigma
            * sqrt_T
        )
    )


def vega(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
):

    sqrt_T = jnp.sqrt(
        jnp.maximum(maturity, 1e-8)
    )

    d1 = (
        jnp.log(spot / strike)
        + (r + 0.5 * sigma**2)
        * maturity
    ) / (sigma * sqrt_T)

    return (
        spot
        * pdf(d1)
        * sqrt_T
    )


def bs_feature_price(
    x: jnp.ndarray,
):
    """x = [S, K, T, sigma, r]"""

    return black_scholes_price_single(
        spot=x[0],
        strike=x[1],
        maturity=x[2],
        sigma=x[3],
        r=x[4],
        is_call=True,
    )


def bs_feature_gradient(
    x: jnp.ndarray,
):

    return jax.grad(
        bs_feature_price
    )(x)


def bs_feature_hessian(
    x: jnp.ndarray,
):

    return jax.hessian(
        bs_feature_price
    )(x)


def create_bs_dataset(
    X: jnp.ndarray,
):
    """Prices, gradients and Hessians from analytic Black-Scholes."""

    prices = jax.vmap(
        bs_feature_price
    )(X)

    gradients = jax.vmap(
        bs_feature_gradient
    )(X)

    hessians = jax.vmap(
        bs_feature_hessian
    )(X)

    return (
        prices,
        gradients,
        hessians,
    )


def compare_to_mc(
    analytic_price,
    mc_price,
):

    abs_error = jnp.abs(
        analytic_price
        - mc_price
    )

    rel_error = (
        abs_error
        / (
            jnp.abs(analytic_price)
            + 1e-12
        )
    )

    return {
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
    }


if __name__ == "__main__":

    price = black_scholes_price_single(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        sigma=0.2,
        r=0.05,
        is_call=True,
    )

    print(
        f"Price : {price:.6f}"
    )

    print(
        f"Delta : {delta(100,100,1,0.2,0.05):.6f}"
    )

    print(
        f"Gamma : {gamma(100,100,1,0.2,0.05):.6f}"
    )

    print(
        f"Vega  : {vega(100,100,1,0.2,0.05):.6f}"
    )
