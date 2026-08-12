import jax
import jax.numpy as jnp
import numpy as np

from marktsimulation.pricing_model import (
    BlackScholesParams,
    BlackScholesModel,
)

from marktsimulation.timesteppingscheme import (
    EulerMaruyama,
)

from marktsimulation.payoff import (
    payoff_spec,
    sigmoid_smooth,
)

from marktsimulation.monte_carlo_pricer import (
    MonteCarloPricer,
)

from marktsimulation.sobolev_labels import (
    create_sobolev_labels,
)


MC_NUM_PATHS = 50_000
MC_NUM_STEPS = 50


def _payoff_smoothing_width(spot, maturity, sigma):
    """
    Width of the payoff-smoothing kernel, scaled to the terminal price's
    dispersion: a fixed dollar width is too narrow to catch enough paths for a
    stable HVP estimate once S_T's spread grows.
    """
    dispersion = spot * sigma * jnp.sqrt(jnp.maximum(maturity, 1e-6))
    return jnp.maximum(0.05 * dispersion, 1e-3)


def simulate_terminal(spot, sigma, r, maturity, num_paths, key, antithetic=True):
    """Exact terminal draws of a geometric Brownian motion."""

    z = jax.random.normal(key, (num_paths,))

    drift = (r - 0.5 * sigma**2) * maturity
    scale = sigma * jnp.sqrt(maturity)

    forward = spot * jnp.exp(drift + scale * z)

    if not antithetic:
        return (forward,)

    return forward, spot * jnp.exp(drift - scale * z)


def bs_mc_price(
    x: jnp.ndarray,
    key: jnp.ndarray,
    payoff: str = "european_call",
    num_paths: int = MC_NUM_PATHS,
    num_steps: int = MC_NUM_STEPS,
    antithetic: bool = True,
):
    """x = [S, K, T, sigma, r]"""

    spot = x[0]
    strike = x[1]
    maturity = x[2]
    sigma = x[3]
    r = x[4]

    params = BlackScholesParams(
        r=r,
        sigma=sigma,
    )

    smooth_w = _payoff_smoothing_width(spot, maturity, sigma)

    spec = payoff_spec(payoff)

    payoff_obj = spec.build(
        strike=strike,
        smooth_fn=sigmoid_smooth,
        smooth_w=smooth_w,
    )

    if spec.path_dependent:

        pricer = MonteCarloPricer(
            model=BlackScholesModel(scheme=EulerMaruyama()),
            payoff=payoff_obj,
            payoff_on_path=True,
        )

        undiscounted = pricer.price(
            s0=jnp.array([spot]),
            params=params,
            maturity=maturity,
            num_paths=num_paths,
            num_steps=num_steps,
            key=key,
        )

    else:

        blocks = simulate_terminal(
            spot, sigma, r, maturity, num_paths, key, antithetic=antithetic
        )

        undiscounted = jnp.mean(
            jnp.stack([jnp.mean(jax.vmap(payoff_obj)(block)) for block in blocks])
        )

    return jnp.exp(-r * maturity) * undiscounted


def make_mc_calibration_pricer(
    spot: float,
    maturities: jnp.ndarray,
    seed: int = 0,
    num_paths: int = MC_NUM_PATHS,
    num_steps: int = MC_NUM_STEPS,
):
    """
    Build a `pricing_fn` for the Calibrator that prices the whole instrument
    set by Monte Carlo instead of the analytic formula.
    """

    maturities = np.asarray(maturities)

    num_instruments = len(maturities)

    groups = [
        (
            float(T),
            jnp.asarray(
                np.where(maturities == T)[0]
            ),
        )
        for T in np.unique(maturities)
    ]

    base_key = jax.random.PRNGKey(seed)

    scheme = EulerMaruyama()

    model = BlackScholesModel(
        scheme=scheme
    )

    def mc_pricing_fn(
        params,
        strikes,
        maturities,
        is_call,
    ):

        prices = jnp.zeros(
            num_instruments
        )

        for group_index, (T, indices) in enumerate(groups):

            key = jax.random.fold_in(
                base_key,
                group_index,
            )

            paths = scheme.generate_paths(
                s0=jnp.array([spot]),
                drift_fn=model.drift,
                diffusion_fn=model.diffusion,
                params=params,
                key=key,
                num_paths=num_paths,
                num_steps=num_steps,
                dt=T / num_steps,
                corr=model.noise_correlation(params),
            )

            terminal = paths[:, -1, 0]

            omega = jnp.where(
                is_call[indices],
                1.0,
                -1.0,
            )

            intrinsic = (
                omega[None, :]
                * (
                    terminal[:, None]
                    - strikes[indices][None, :]
                )
            )

            payoffs = sigmoid_smooth(
                intrinsic,
                _payoff_smoothing_width(
                    spot,
                    T,
                    params.sigma,
                ),
            )

            discount = jnp.exp(
                -params.r * T
            )

            prices = prices.at[indices].set(
                discount
                * jnp.mean(
                    payoffs,
                    axis=0,
                )
            )

        return prices

    return mc_pricing_fn


def generate_training_paths(
    x: jnp.ndarray,
    num_paths: int = 100,
):

    spot = x[0]
    maturity = x[2]
    sigma = x[3]
    r = x[4]

    params = BlackScholesParams(
        r=r,
        sigma=sigma,
    )

    scheme = EulerMaruyama()

    model = BlackScholesModel(
        scheme=scheme
    )

    num_steps = MC_NUM_STEPS

    dt = maturity / num_steps

    paths = scheme.generate_paths(
        s0=jnp.array([spot]),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(0),
        num_paths=num_paths,
        num_steps=num_steps,
        dt=dt,
    )

    time_grid = jnp.linspace(
        0.0,
        maturity,
        num_steps + 1,
    )

    return time_grid, paths
