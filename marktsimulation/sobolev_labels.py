import jax
import jax.numpy as jnp

from typing import Callable, Optional, Tuple


def create_sobolev_labels(
    price_fn: Callable,
    X: jnp.ndarray,
    V: jnp.ndarray,
    sobolev_order: int = 2,
    message: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Price, gradient and (for order 2) HVP of `price_fn` at every row of X.

    The loop is sequential on purpose: vmapping it would hold every
    sample's Monte Carlo paths in memory at once.
    """

    # value_and_grad so the simulation runs once per sample, not twice
    price_and_grad_fn = jax.jit(jax.value_and_grad(price_fn))

    if sobolev_order >= 2:
        hvp_fn = jax.jit(
            lambda x, v: jax.jvp(jax.grad(price_fn), (x,), (v,))[1]
        )

    prices = []
    gradients = []
    hvps = []

    if message is not None:
        print(message)

    for i in range(len(X)):
        price, grad = price_and_grad_fn(X[i])

        prices.append(price)
        gradients.append(grad)

        if sobolev_order >= 2:
            hvps.append(hvp_fn(X[i], V[i]))

    return (
        jnp.stack(prices),
        jnp.stack(gradients),
        jnp.stack(hvps) if sobolev_order >= 2 else None,
    )
