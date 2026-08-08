import jax
import jax.numpy as jnp

from marktsimulation.black_scholes_mc import (
    create_bs_mc_dataset_with_hvps,
)

from surrogate_modeling.dataset import (
    SobolevDataset,
)


def create_sobolev_dataset(
    market_data,
    fitted_params,
    sobolev_order,
    n_samples: int = 600,
    r_spread: float = 0.02,
    seed: int = 0,
    xva_horizon: float = 1.0,
    domain_n_sigma: float = 3.0,
):
    """
    Monte-Carlo-labelled Sobolev training set.

    Prices, gradients and HVPs all come from differentiating through the
    Monte Carlo pricer; analytic Black-Scholes is only used for
    calibration and the XVA benchmark, never for training labels.

    The sampling domain is centered on the calibrated `fitted_params`,
    with a spot range covering a `domain_n_sigma` lognormal move over
    `xva_horizon` years to match the exposure simulation in
    xva_analysis.py.
    """

    key = jax.random.PRNGKey(seed)
    key, x_key, v_key = jax.random.split(key, 3)

    X = jax.random.uniform(
        x_key,
        shape=(n_samples, 5),
    )

    spot = market_data.spot
    market_sigma = fitted_params.sigma
    market_r = fitted_params.r

    # S: cover a `domain_n_sigma`-sigma lognormal move over `xva_horizon` years
    log_spread = domain_n_sigma * market_sigma * jnp.sqrt(xva_horizon)
    S_min = spot * jnp.exp(-log_spread)
    S_max = spot * jnp.exp(log_spread)

    X = X.at[:, 0].set(
        S_min +
        (S_max - S_min) * X[:, 0]
    )

    K_min = float(jnp.min(market_data.strikes))
    K_max = float(jnp.max(market_data.strikes))

    X = X.at[:, 1].set(
        K_min +
        (K_max - K_min) * X[:, 1]
    )

    T_min = float(jnp.min(market_data.maturities))
    T_max = float(jnp.max(market_data.maturities))

    X = X.at[:, 2].set(
        T_min +
        (T_max - T_min) * X[:, 2]
    )

    sigma_min = 0.8 * market_sigma
    sigma_max = 1.2 * market_sigma

    X = X.at[:, 3].set(
        sigma_min
        + (sigma_max - sigma_min)
        * X[:, 3]
    )

    # r: vary around the calibrated market rate, not a fixed constant,
    # so the network actually sees a training signal for d(price)/dr
    r_min = market_r - r_spread
    r_max = market_r + r_spread

    X = X.at[:, 4].set(
        r_min
        + (r_max - r_min)
        * X[:, 4]
    )

    # one random unit probe direction per sample (Hutchinson-style HVP
    # probing), rather than one fixed direction reused for every sample
    V_raw = jax.random.normal(v_key, shape=(n_samples, X.shape[1]))
    V = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    print("Generating Monte Carlo BS dataset with HVPs...")

    prices, gradients, hvps = (
        create_bs_mc_dataset_with_hvps(X, V, sobolev_order)
    )

    dataset = SobolevDataset(
        X=X,
        y=prices,
        gradients=gradients,
        hvps=hvps,
        V=V,
    )

    return dataset
