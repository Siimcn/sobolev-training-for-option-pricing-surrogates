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
    """Exact terminal draws of correlated geometric Brownian assets."""

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
    """One basket price under an explicit `key`."""

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
    weights, equal vols and a correlation matrix with one off-diagonal value.
    Only then may the price function be symmetrized.
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
    """Build `f(x, key) -> price` for a basket option, with the feature layout"""

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
    """Basket value paths behind one training label, with x = [S_1, ..., S_n, K, T]."""

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
    """Correlated per-asset paths, shaped `(num_paths, num_steps + 1, n_assets)`."""

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
    """Price and per-asset deltas/gammas of a basket option."""

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
