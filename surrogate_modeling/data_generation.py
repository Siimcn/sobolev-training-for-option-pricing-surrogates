import jax
import jax.numpy as jnp

from typing import Optional

from marktsimulation.basket_mc import (
    make_basket_feature_price,
    uniform_correlation,
)

from marktsimulation.black_scholes_mc import (
    bs_mc_feature_price,
)

from marktsimulation.sobolev_labels import (
    create_sobolev_labels,
)

from surrogate_modeling.dataset import (
    SobolevDataset,
)


BLACK_SCHOLES = "black_scholes"
BASKET_BLACK_SCHOLES = "basket_black_scholes"

PRICING_MODELS = (BLACK_SCHOLES, BASKET_BLACK_SCHOLES)


def _uniform(u, low, high):
    return low + (high - low) * u


def _spot_range(market_data, fitted_params, xva_horizon, domain_n_sigma):
    """
    Spot range covering a `domain_n_sigma` lognormal move over
    `xva_horizon` years, matching the exposure simulation in
    xva_analysis.py.
    """

    log_spread = domain_n_sigma * fitted_params.sigma * jnp.sqrt(xva_horizon)

    return (
        market_data.spot * jnp.exp(-log_spread),
        market_data.spot * jnp.exp(log_spread),
    )


def _black_scholes_domain(
    u, market_data, fitted_params, r_spread, xva_horizon, domain_n_sigma
):
    """x = [S, K, T, sigma, r]"""

    S_min, S_max = _spot_range(
        market_data, fitted_params, xva_horizon, domain_n_sigma
    )

    X = u.at[:, 0].set(_uniform(u[:, 0], S_min, S_max))

    X = X.at[:, 1].set(
        _uniform(
            u[:, 1],
            float(jnp.min(market_data.strikes)),
            float(jnp.max(market_data.strikes)),
        )
    )

    X = X.at[:, 2].set(
        _uniform(
            u[:, 2],
            float(jnp.min(market_data.maturities)),
            float(jnp.max(market_data.maturities)),
        )
    )

    X = X.at[:, 3].set(
        _uniform(u[:, 3], 0.8 * fitted_params.sigma, 1.2 * fitted_params.sigma)
    )

    # r: vary around the calibrated market rate, not a fixed constant,
    # so the network actually sees a training signal for d(price)/dr
    X = X.at[:, 4].set(
        _uniform(
            u[:, 4],
            fitted_params.r - r_spread,
            fitted_params.r + r_spread,
        )
    )

    return X


def _basket_domain(
    u, market_data, fitted_params, n_assets, xva_horizon, domain_n_sigma
):
    """x = [S_1, ..., S_n, K, T]"""

    S_min, S_max = _spot_range(
        market_data, fitted_params, xva_horizon, domain_n_sigma
    )

    X = u

    for i in range(n_assets):
        X = X.at[:, i].set(_uniform(u[:, i], S_min, S_max))

    X = X.at[:, n_assets].set(
        _uniform(
            u[:, n_assets],
            float(jnp.min(market_data.strikes)),
            float(jnp.max(market_data.strikes)),
        )
    )

    X = X.at[:, n_assets + 1].set(
        _uniform(
            u[:, n_assets + 1],
            float(jnp.min(market_data.maturities)),
            float(jnp.max(market_data.maturities)),
        )
    )

    return X


def create_sobolev_dataset(
    market_data,
    fitted_params,
    sobolev_order,
    n_samples: int = 600,
    r_spread: float = 0.02,
    seed: int = 0,
    xva_horizon: float = 1.0,
    domain_n_sigma: float = 3.0,
    pricing_model: str = BLACK_SCHOLES,
    basket=None,
) -> SobolevDataset:
    """
    Monte-Carlo-labelled Sobolev training set.

    Prices, gradients and HVPs all come from differentiating through the
    Monte Carlo pricer; analytic Black-Scholes is only used for
    calibration and the XVA benchmark, never for training labels.

    `pricing_model` selects what the surrogate learns to price. The
    sampling domain is centered on the calibrated `fitted_params` either
    way; `basket` is the BasketConfig, required for the basket model.
    """

    if pricing_model not in PRICING_MODELS:
        raise ValueError(
            f"Unknown pricing_model '{pricing_model}'. "
            f"Expected one of: {', '.join(PRICING_MODELS)}."
        )

    key = jax.random.PRNGKey(seed)
    key, x_key, v_key = jax.random.split(key, 3)

    n_features = (
        5 if pricing_model == BLACK_SCHOLES else basket.n_assets + 2
    )

    u = jax.random.uniform(x_key, shape=(n_samples, n_features))

    if pricing_model == BLACK_SCHOLES:
        X = _black_scholes_domain(
            u, market_data, fitted_params, r_spread, xva_horizon, domain_n_sigma
        )
        price_fn = bs_mc_feature_price
        label = "BS"
    else:
        X = _basket_domain(
            u, market_data, fitted_params, basket.n_assets,
            xva_horizon, domain_n_sigma,
        )
        price_fn = _basket_price_fn(basket, fitted_params)
        label = f"basket BS ({basket.n_assets} assets)"

    # one random unit probe direction per sample (Hutchinson-style HVP
    # probing), rather than one fixed direction reused for every sample
    V_raw = jax.random.normal(v_key, shape=(n_samples, n_features))
    V = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    print(f"Generating Monte Carlo {label} dataset with HVPs...")

    prices, gradients, hvps = create_sobolev_labels(
        price_fn,
        X,
        V,
        sobolev_order=sobolev_order,
        message=f"Generating data with HVPs for {n_samples} samples...",
    )

    return SobolevDataset(
        X=X,
        y=prices,
        gradients=gradients,
        hvps=hvps,
        V=V,
    )


def _basket_price_fn(basket, fitted_params):
    """All assets share the calibrated vol and rate; only spots vary in x."""

    weights = (
        jnp.asarray(basket.weights)
        if basket.weights is not None
        else jnp.full(basket.n_assets, 1.0 / basket.n_assets)
    )

    return make_basket_feature_price(
        weights=weights,
        corr=uniform_correlation(basket.n_assets, basket.correlation),
        sigmas=jnp.full(basket.n_assets, fitted_params.sigma),
        r=fitted_params.r,
        num_paths=basket.num_paths,
        num_steps=basket.num_steps,
        seed=basket.label_seed,
    )
