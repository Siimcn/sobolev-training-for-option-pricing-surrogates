import jax
import jax.numpy as jnp

from jax.scipy.stats import norm

"""
Bachelier (normal) model: closed-form prices and Greeks.
"""


def _floor_maturity(maturity):
    """Keep sqrt(T) and the division by it finite at expiry."""

    return jnp.maximum(maturity, 1e-12)


def bachelier_price(
    params, strikes: jnp.ndarray, maturities: jnp.ndarray, is_call: jnp.ndarray, spot
) -> jnp.ndarray:
    """Vectorised Bachelier price from the spot, in the Calibrator's argument order."""

    forward = spot * jnp.exp(params.r * maturities)

    sqrt_t = jnp.sqrt(_floor_maturity(maturities))

    scale = params.sigma * sqrt_t

    omega = jnp.where(is_call, 1.0, -1.0)

    moneyness = omega * (forward - strikes)

    d = moneyness / scale

    undiscounted = moneyness * norm.cdf(d) + scale * norm.pdf(d)

    return jnp.exp(-params.r * maturities) * undiscounted


def bachelier_price_single(
    forward, strike, maturity, sigma, r=0.0, is_call: bool = True
):
    """Scalar Bachelier price."""

    sqrt_t = jnp.sqrt(_floor_maturity(maturity))

    scale = sigma * sqrt_t

    omega = jnp.where(is_call, 1.0, -1.0)

    moneyness = omega * (forward - strike)

    d = moneyness / scale

    return jnp.exp(-r * maturity) * (moneyness * norm.cdf(d) + scale * norm.pdf(d))


def bachelier_forward(spot, maturity, r):
    """The drift the normal model carries in its state, S exp(rT)."""

    return spot * jnp.exp(r * maturity)


def bachelier_spot_price(spot, strike, maturity, sigma, r=0.0, is_call: bool = True):
    """Bachelier price quoted from the spot rather than the forward."""

    return bachelier_price_single(
        bachelier_forward(spot, maturity, r), strike, maturity, sigma, r, is_call
    )


def basket_bachelier_spot_price(
    spots, strike, maturity, weights, sigmas, corr, r=0.0, is_call: bool = True
):
    """Exact basket price quoted from the spots."""

    return bachelier_price_single(
        forward=bachelier_forward(jnp.sum(weights * spots), maturity, r),
        strike=strike,
        maturity=maturity,
        sigma=basket_normal_volatility(weights, sigmas, corr),
        r=r,
        is_call=is_call,
    )


def bachelier_delta(forward, strike, maturity, sigma, r=0.0, is_call: bool = True):
    """omega exp(-rT) Phi(d); bounded by 1 in absolute value."""

    omega = jnp.where(is_call, 1.0, -1.0)

    d = omega * (forward - strike) / (sigma * jnp.sqrt(_floor_maturity(maturity)))

    return omega * jnp.exp(-r * maturity) * norm.cdf(d)


def bachelier_gamma(forward, strike, maturity, sigma, r=0.0):
    """exp(-rT) phi(d) / (sigma sqrt(T)); identical for calls and puts."""

    sqrt_t = jnp.sqrt(_floor_maturity(maturity))

    scale = sigma * sqrt_t

    d = (forward - strike) / scale

    return jnp.exp(-r * maturity) * norm.pdf(d) / scale


def bachelier_vega(forward, strike, maturity, sigma, r=0.0):
    """exp(-rT) sqrt(T) phi(d); identical for calls and puts."""

    sqrt_t = jnp.sqrt(_floor_maturity(maturity))

    d = (forward - strike) / (sigma * sqrt_t)

    return jnp.exp(-r * maturity) * sqrt_t * norm.pdf(d)


def basket_normal_volatility(
    weights: jnp.ndarray, sigmas: jnp.ndarray, corr: jnp.ndarray
):
    """Volatility of the weighted basket, sqrt(w' diag(sigma) C diag(sigma) w)."""

    scaled = weights * sigmas

    return jnp.sqrt(scaled @ corr @ scaled)


def basket_bachelier_price(
    spots: jnp.ndarray,
    strike,
    maturity,
    weights: jnp.ndarray,
    sigmas: jnp.ndarray,
    corr: jnp.ndarray,
    r=0.0,
    is_call: bool = True,
):
    """Exact basket price: a vanilla Bachelier option on the weighted basket."""

    return bachelier_price_single(
        forward=jnp.sum(weights * spots),
        strike=strike,
        maturity=maturity,
        sigma=basket_normal_volatility(weights, sigmas, corr),
        r=r,
        is_call=is_call,
    )


def basket_bachelier_greeks(
    spots: jnp.ndarray,
    strike,
    maturity,
    weights: jnp.ndarray,
    sigmas: jnp.ndarray,
    corr: jnp.ndarray,
    r=0.0,
    is_call: bool = True,
):
    """Closed-form price, per-asset deltas and the full spot Hessian."""

    basket = jnp.sum(weights * spots)

    sigma_b = basket_normal_volatility(weights, sigmas, corr)

    delta = bachelier_delta(basket, strike, maturity, sigma_b, r, is_call)
    gamma = bachelier_gamma(basket, strike, maturity, sigma_b, r)

    return {
        "price": bachelier_price_single(basket, strike, maturity, sigma_b, r, is_call),
        "delta": delta * weights,
        "gamma": gamma * jnp.outer(weights, weights),
        "basket": basket,
        "basket_volatility": sigma_b,
    }
