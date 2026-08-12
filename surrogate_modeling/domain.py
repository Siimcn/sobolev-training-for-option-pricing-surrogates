import jax.numpy as jnp

from typing import Dict


"""
Shared helpers for mapping a uniform block onto a sampling domain.
"""


def uniform(u, low, high):
    return low + (high - low) * u


def lognormal_spot_range(market_data, sigma, horizon, n_sigma):
    """Multiplicative range, for a model whose volatility is a percentage."""

    log_spread = n_sigma * sigma * jnp.sqrt(horizon)

    return (
        market_data.spot * jnp.exp(-log_spread),
        market_data.spot * jnp.exp(log_spread),
    )


def normal_spot_range(market_data, sigma, horizon, n_sigma):
    """Additive range, for a model whose volatility carries price units."""

    spread = n_sigma * sigma * jnp.sqrt(horizon)

    return market_data.spot - spread, market_data.spot + spread


def maturity_range(market_data, min_maturity):
    """
    Market maturities, optionally floored. The shortest expiries carry a near-
    discontinuous payoff, so their curvature labels are orders of magnitude
    larger than the rest of the domain.
    """

    low = float(jnp.min(market_data.maturities))
    high = float(jnp.max(market_data.maturities))

    if min_maturity is not None:
        low = max(low, float(min_maturity))

    if low >= high:
        raise ValueError(
            f"min_maturity {min_maturity} leaves no maturity range below "
            f"the longest market expiry {high}."
        )

    return low, high


def strike_range(market_data):
    return (
        float(jnp.min(market_data.strikes)),
        float(jnp.max(market_data.strikes)),
    )


def moneyness_strikes(spot: float) -> Dict[str, float]:
    """A deep ITM call is near-linear in S, so ATM catches Greeks errors it hides."""

    return {"ITM": 0.85 * spot, "ATM": float(spot), "OTM": 1.15 * spot}


def exposure_time_grid(horizon, num_steps, min_maturity):
    """Stop the exposure profile at the training floor."""

    time_grid = jnp.linspace(0.0, horizon, num_steps + 1)

    floor = max(float(min_maturity), 1e-8)

    return time_grid, jnp.maximum(horizon - time_grid, floor)
