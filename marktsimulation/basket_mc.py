import jax
import jax.numpy as jnp

from marktsimulation.payoff import payoff_spec, sigmoid_smooth
from marktsimulation.monte_carlo_pricer import MonteCarloPricer
from marktsimulation.pricing_model import (
    BasketBlackScholesModel,
    BasketBlackScholesParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama


def simulate_basket_terminal(
    s0: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    maturity,
    num_paths: int,
    key: jnp.ndarray,
    antithetic: bool = True,
):
    """
    Exact terminal draws of correlated geometric Brownian assets.

    A GBM with constant coefficients has a known transition law, so the
    terminal state is drawn in a single step

        S_i(T) = S_i(0) exp((r - sigma_i^2 / 2) T + sigma_i sqrt(T) Z_i)

    with Z correlated by the Cholesky factor of `corr`. For a payoff that
    reads only S(T) this carries no discretisation error at all, and costs
    one normal per asset per path instead of `num_steps` of them.

    With `antithetic` every draw Z is paired with -Z. Both have the same
    law, so the estimator stays unbiased, and the two payoffs are
    negatively correlated, which removes most of the variance that comes
    from the level of the basket. Adapted from diff-ml's
    `Bachelier.antithetic_payoff`.

    Returns a tuple of terminal-state blocks to be averaged over: one
    entry without antithetics, two with.

    Not valid for a path-dependent payoff - an Asian option needs the
    whole trajectory - which is why the stepping scheme stays.
    """

    z = jax.random.normal(key, (num_paths, len(sigmas)))

    correlated = z @ jnp.linalg.cholesky(corr).T

    drift = (r - 0.5 * sigmas**2) * maturity
    scale = sigmas * jnp.sqrt(maturity)

    forward = s0 * jnp.exp(drift + scale * correlated)

    if not antithetic:
        return (forward,)

    return forward, s0 * jnp.exp(drift - scale * correlated)


def basket_price(
    model,
    params,
    s0,
    strike,
    maturity,
    key,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
    smooth_w: float = 1.0,
    antithetic: bool = True,
):
    """
    One basket price under an explicit `key`.

    The scheme follows the payoff: a terminal-only payoff is priced from
    an exact one-step draw, a path-dependent one from the stepping scheme.
    Nothing here decides that - `payoff_spec(...).path_dependent` does.
    """

    spec = payoff_spec(payoff)

    payoff_obj = spec.build(
        strike=strike,
        smooth_fn=sigmoid_smooth,
        smooth_w=smooth_w,
    )

    if spec.path_dependent:

        pricer = MonteCarloPricer(
            model,
            payoff_obj,
            value_fn=lambda s: model.basket_value(s, params),
            payoff_on_path=True,
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

        blocks = simulate_basket_terminal(
            s0,
            params.corr,
            params.sigmas,
            params.r,
            maturity,
            num_paths,
            key,
            antithetic=antithetic,
        )

        undiscounted = jnp.mean(
            jnp.stack(
                [
                    jnp.mean(
                        jax.vmap(payoff_obj)(
                            jax.vmap(lambda s: model.basket_value(s, params))(block)
                        )
                    )
                    for block in blocks
                ]
            )
        )

    return undiscounted * jnp.exp(-params.r * maturity)


def uniform_correlation(n_assets: int, rho: float) -> jnp.ndarray:
    """Correlation matrix with rho off the diagonal and ones on it."""

    return (
        jnp.full((n_assets, n_assets), rho)
        .at[jnp.diag_indices(n_assets)]
        .set(1.0)
    )


def is_exchangeable(weights: jnp.ndarray, sigmas: jnp.ndarray, corr: jnp.ndarray) -> bool:
    """
    True when permuting the spots leaves the true price unchanged: equal
    weights, equal vols and a correlation matrix with one off-diagonal
    value. Only then may the price function be symmetrized.
    """

    off_diagonal = corr[~jnp.eye(len(corr), dtype=bool)]

    return bool(
        jnp.allclose(weights, weights[0])
        and jnp.allclose(sigmas, sigmas[0])
        and (len(off_diagonal) == 0 or jnp.allclose(off_diagonal, off_diagonal[0]))
    )


def make_basket_feature_price(
    weights: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
    smooth_fraction: float = 0.05,
    symmetrize: bool = False,
    antithetic: bool = True,
):
    """
    Build `f(x, key) -> price` for a basket option, with the feature layout

        x = [S_1, ..., S_n, K, T]

    The basket structure (weights, correlation, per-asset vols, rate) is
    fixed here rather than carried in x, mirroring how diff-ml's Bachelier
    example uses the spot vector alone as its input.

    The key is an argument, not a closure. Binding one key for a whole
    dataset makes every label share a single realisation of the Monte
    Carlo error, which then does not average out over samples - see
    `create_sobolev_labels` for the measurement and the reasoning.

    With `symmetrize` the spots are sorted before pricing. Asset i draws
    Brownian column i, so the raw estimator is not invariant under
    permuting the spots even though the true price is - it teaches the
    network an asymmetry that does not exist. Sorting picks one
    representative per orbit and restores the invariance exactly, at no
    extra cost; the gradient permutes back correctly.
    """

    if symmetrize and not is_exchangeable(weights, sigmas, corr):
        raise ValueError(
            "symmetrize requires an exchangeable basket: equal weights, "
            "equal volatilities and a uniform correlation."
        )

    def price_fn(x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        return basket_feature_price(
            x,
            weights=weights,
            corr=corr,
            sigmas=sigmas,
            r=r,
            key=key,
            payoff=payoff,
            num_paths=num_paths,
            num_steps=num_steps,
            smooth_fraction=smooth_fraction,
            symmetrize=symmetrize,
            antithetic=antithetic,
        )

    return price_fn


def basket_feature_price(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    key: jnp.ndarray,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
    smooth_fraction: float = 0.05,
    symmetrize: bool = False,
    antithetic: bool = True,
) -> jnp.ndarray:
    """One basket price for x = [S_1, ..., S_n, K, T] under an explicit `key`."""

    n_assets = len(weights)

    s0 = jnp.sort(x[:n_assets]) if symmetrize else x[:n_assets]
    strike = x[n_assets]
    maturity = x[n_assets + 1]

    model = BasketBlackScholesModel(scheme=EulerMaruyama())

    params = BasketBlackScholesParams(
        r=r,
        sigmas=sigmas,
        weights=weights,
        corr=corr,
    )

    # same construction as _payoff_smoothing_width, with the basket
    # value standing in for the single-asset spot
    basket0 = jnp.sum(weights * s0)
    dispersion = basket0 * jnp.mean(sigmas) * jnp.sqrt(jnp.maximum(maturity, 1e-6))
    smooth_w = jnp.maximum(smooth_fraction * dispersion, 1e-3)

    return basket_price(
        model,
        params,
        s0,
        strike,
        maturity,
        key,
        payoff=payoff,
        num_paths=num_paths,
        num_steps=num_steps,
        smooth_w=smooth_w,
        antithetic=antithetic,
    )


def generate_basket_training_paths(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    num_paths: int = 100,
    num_steps: int = 50,
    seed: int = 0,
):
    """
    Basket value paths behind one training label, with x = [S_1, ..., S_n, K, T].

    Returns the weighted basket, which is what the option is written on,
    rather than the individual assets.
    """

    n_assets = len(weights)

    time_grid, paths = simulate_basket_assets(
        s0=x[:n_assets],
        weights=weights,
        corr=corr,
        sigmas=sigmas,
        r=r,
        horizon=x[n_assets + 1],
        num_paths=num_paths,
        num_steps=num_steps,
        key=jax.random.PRNGKey(seed),
    )

    return time_grid, jnp.sum(paths * weights, axis=-1)


def simulate_basket_assets(
    s0: jnp.ndarray,
    weights: jnp.ndarray,
    corr: jnp.ndarray,
    sigmas: jnp.ndarray,
    r: float,
    horizon: float,
    num_paths: int,
    num_steps: int,
    key: jnp.ndarray,
):
    """
    Correlated per-asset paths, shaped `(num_paths, num_steps + 1, n_assets)`.

    The individual assets rather than the weighted basket, because the
    surrogate's features are the spots themselves.
    """

    model = BasketBlackScholesModel(scheme=EulerMaruyama())

    params = BasketBlackScholesParams(
        r=r,
        sigmas=sigmas,
        weights=weights,
        corr=corr,
    )

    paths = model.scheme.generate_paths(
        s0=s0,
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=key,
        num_paths=num_paths,
        num_steps=num_steps,
        dt=horizon / num_steps,
        corr=model.noise_correlation(params),
    )

    return jnp.linspace(0.0, horizon, num_steps + 1), paths


def basket_greeks(
    model,
    params,
    s0,
    strike,
    maturity,
    key,
    payoff: str = "european_call",
    num_paths: int = 50_000,
    num_steps: int = 50,
):
    """
    Price and per-asset deltas/gammas of a basket option.

    Pathwise AD through the whole Monte Carlo simulation. The key is fixed
    within one call, which is exactly where common random numbers belong:
    all derivatives are taken at the same point on the same paths.
    """

    def price_fn(s0_):
        return basket_price(
            model, params, s0_, strike, maturity, key,
            payoff=payoff, num_paths=num_paths, num_steps=num_steps,
        )

    price, delta = jax.value_and_grad(price_fn)(s0)
    gamma = jax.hessian(price_fn)(s0)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
    }