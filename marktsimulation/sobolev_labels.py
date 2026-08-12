import jax
import jax.numpy as jnp

from typing import Callable, Optional, Tuple


def label_keys(seed: int, n_samples: int, shared: bool = False) -> jnp.ndarray:
    """One PRNG key per sample."""

    base = jax.random.PRNGKey(seed)

    if shared:
        return jnp.broadcast_to(base, (n_samples, len(base)))

    return jax.random.split(base, n_samples)


def create_sobolev_labels(
    price_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    X: jnp.ndarray,
    V: jnp.ndarray,
    keys: jnp.ndarray,
    sobolev_order: int = 2,
    message: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Price, gradient and (for order 2) HVP of `price_fn` at every row of X."""

    if len(keys) != len(X):
        raise ValueError(
            f"expected one key per sample, got {len(keys)} keys for {len(X)} samples."
        )

    price_and_grad_fn = jax.jit(jax.value_and_grad(price_fn))

    if sobolev_order >= 2:
        hvp_fn = jax.jit(
            lambda x, v, key: jax.jvp(
                lambda z: jax.grad(price_fn)(z, key), (x,), (v,)
            )[1]
        )

    prices = []
    gradients = []
    hvps = []

    if message is not None:
        print(message)

    for i in range(len(X)):
        price, grad = price_and_grad_fn(X[i], keys[i])

        prices.append(price)
        gradients.append(grad)

        if sobolev_order >= 2:
            hvps.append(hvp_fn(X[i], V[i], keys[i]))

    return (
        jnp.stack(prices),
        jnp.stack(gradients),
        jnp.stack(hvps) if sobolev_order >= 2 else None,
    )
