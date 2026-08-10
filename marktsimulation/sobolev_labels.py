import jax
import jax.numpy as jnp

from typing import Callable, Optional, Tuple


def label_keys(seed: int, n_samples: int, shared: bool = False) -> jnp.ndarray:
    """
    One PRNG key per sample.

    `shared=True` reproduces the historical behaviour of binding a single
    key for the whole dataset. It exists so the bias it causes can be
    demonstrated and regression-tested, not because it is ever the better
    choice - see `create_sobolev_labels`.
    """

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
    """
    Price, gradient and (for order 2) HVP of `price_fn` at every row of X.

    `price_fn(x, key) -> price`. Randomness is an argument rather than a
    closure, and every sample gets its own key.

    Where common random numbers help, and where they hurt
    -----------------------------------------------------
    Sharing one key *within* a sample is essential and is kept: the price,
    the gradient and the HVP are all differentiated through the same path
    bundle, so the reported gradient really is the gradient of the
    reported price. Common random numbers are also kept inside the
    calibration residual, where re-drawing paths at each solver step would
    make the residual discontinuous in the parameters.

    Sharing one key *across* samples is a different matter, and was the
    previous behaviour. Derivatives here come from automatic
    differentiation within a single sample, never from differencing two
    samples, so a common stream across samples buys nothing. What it costs
    is measurable: the per-sample Monte Carlo errors become strongly
    correlated (measured pairwise correlation +0.80 against +0.001 for
    independent keys), so the dataset-wide mean error does not shrink with
    the sample count. At PRNGKey(0) that left every label low by about
    -0.71 in price units, with 63 of 64 probe points below a
    high-accuracy reference. The network then faithfully learns the
    shifted surface.

    Independent keys give the same per-label error but let it average out
    as 1/sqrt(n), which is what a regression needs.

    The loop is sequential on purpose: vmapping it would hold every
    sample's Monte Carlo paths in memory at once.
    """

    if len(keys) != len(X):
        raise ValueError(
            f"expected one key per sample, got {len(keys)} keys for {len(X)} samples."
        )

    # value_and_grad so the simulation runs once per sample, not twice
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
