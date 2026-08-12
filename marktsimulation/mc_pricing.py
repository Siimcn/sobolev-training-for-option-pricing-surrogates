import jax
import jax.numpy as jnp

from typing import Optional

from marktsimulation.monte_carlo_pricer import MonteCarloPricer
from marktsimulation.payoff import payoff_spec, sigmoid_smooth


def mc_price(
    model,
    params,
    s0: jnp.ndarray,
    strike,
    maturity,
    key: jnp.ndarray,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
    smooth_fraction: float = 0.05,
    antithetic: bool = True,
    value_fn=None,
):
    """One Monte Carlo price, for any `PricingModel`."""

    spec = payoff_spec(payoff)

    dispersion = jnp.squeeze(model.terminal_dispersion(s0, params, maturity))

    payoff_obj = spec.build(
        strike=strike,
        smooth_fn=sigmoid_smooth,
        smooth_w=jnp.maximum(smooth_fraction * dispersion, 1e-3),
    )

    if value_fn is None:
        value_fn = getattr(model, "basket_value", None)

        if value_fn is not None:
            value_fn = lambda s, _fn=value_fn: _fn(s, params)

        elif jnp.size(s0) > 1:
            raise ValueError(
                f"{type(model).__name__} has a state of width "
                f"{jnp.size(s0)} but exposes no `basket_value`, and no "
                f"`value_fn` was given. Pass `value_fn` to say which part "
                f"of the state the payoff is written on."
            )

    blocks = (
        None
        if spec.path_dependent
        else model.terminal_state(
            s0, params, maturity, num_paths, key, antithetic=antithetic
        )
    )

    if blocks is None:
        pricer = MonteCarloPricer(
            model, payoff_obj, value_fn=value_fn, payoff_on_path=spec.path_dependent
        )

        undiscounted = pricer.price(
            s0=s0,
            params=params,
            maturity=maturity,
            num_paths=num_paths,
            num_steps=num_steps,
            key=key,
        )

    else:
        undiscounted = jnp.mean(
            jnp.stack(
                [
                    jnp.mean(
                        jax.vmap(payoff_obj)(
                            jax.vmap(value_fn)(block)
                            if value_fn is not None
                            else block[:, 0]
                        )
                    )
                    for block in blocks
                ]
            )
        )

    return undiscounted * jnp.exp(-_rate(params) * maturity)


def _rate(params) -> float:
    """Models without a rate field are undiscounted."""

    return getattr(params, "r", 0.0)


def make_feature_price(
    model,
    params,
    n_assets: int,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
    smooth_fraction: float = 0.05,
    symmetrize: bool = False,
    antithetic: bool = True,
    state_fn=None,
):
    """Build `f(x, key) -> price` for the feature layout"""

    def price_fn(x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        s0 = jnp.sort(x[:n_assets]) if symmetrize else x[:n_assets]

        if state_fn is not None:
            s0 = state_fn(s0, x[n_assets + 1])

        return mc_price(
            model,
            params,
            s0,
            x[n_assets],
            x[n_assets + 1],
            key,
            payoff=payoff,
            num_paths=num_paths,
            num_steps=num_steps,
            smooth_fraction=smooth_fraction,
            antithetic=antithetic,
        )

    return price_fn
