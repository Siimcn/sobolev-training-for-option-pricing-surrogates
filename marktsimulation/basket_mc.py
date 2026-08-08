import jax
import jax.numpy as jnp

from marktsimulation.payoff import EuropeanCall, AsianCall, sigmoid_smooth
from marktsimulation.monte_carlo_pricer import MonteCarloPricer


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