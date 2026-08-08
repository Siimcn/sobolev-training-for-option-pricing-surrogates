import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.basket_mc import (
    make_basket_feature_price,
    uniform_correlation,
)
from marktsimulation.black_scholes import black_scholes_price_single
from marktsimulation.pricing_model import BlackScholesParams
from pipeline.config import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
    BasketConfig,
    DataConfig,
    ExperimentConfig,
)
from surrogate_modeling.data_generation import create_sobolev_dataset


SPOT, SIGMA, R = 100.0, 0.2, 0.05


def _market_data():
    strikes = jnp.array([80.0, 100.0, 120.0])
    maturities = jnp.array([0.25, 0.5, 1.0])

    return MarketData(
        spot=SPOT,
        strikes=strikes,
        maturities=maturities,
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _params():
    return BlackScholesParams(r=R, sigma=SIGMA)


def _dataset(pricing_model, basket=None, n_samples=3, sobolev_order=2):
    return create_sobolev_dataset(
        _market_data(),
        _params(),
        sobolev_order,
        n_samples=n_samples,
        pricing_model=pricing_model,
        basket=basket,
    )


# ------------------------------------------------------------------ config

def test_default_is_black_scholes():
    config = ExperimentConfig()

    assert config.data.pricing_model == BLACK_SCHOLES
    assert not config.is_basket
    assert config.feature_names == ("S", "K", "T", "sigma", "r")


def test_basket_selection_changes_the_feature_names():
    config = ExperimentConfig(
        data=DataConfig(pricing_model=BASKET_BLACK_SCHOLES),
        basket=BasketConfig(n_assets=3),
    )

    assert config.is_basket
    assert config.feature_names == ("S1", "S2", "S3", "K", "T")

    wider = ExperimentConfig(
        data=DataConfig(pricing_model=BASKET_BLACK_SCHOLES),
        basket=BasketConfig(n_assets=5),
    )

    assert wider.feature_names == ("S1", "S2", "S3", "S4", "S5", "K", "T")


def test_unknown_pricing_model_is_rejected():
    try:
        ExperimentConfig(data=DataConfig(pricing_model="heston"))
    except ValueError as e:
        assert "heston" in str(e)
    else:
        assert False, "an unknown pricing model should raise"


def test_weight_length_must_match_asset_count():
    try:
        ExperimentConfig(basket=BasketConfig(n_assets=3, weights=(0.5, 0.5)))
    except ValueError:
        pass
    else:
        assert False, "mismatched weights should raise"


def test_network_input_width_is_derived_by_default():
    assert ExperimentConfig().network.in_size is None


# ------------------------------------------------------------ basket pricer

def test_single_asset_basket_matches_the_analytic_call():
    # a one-asset basket is a vanilla call, so the closed form applies
    price_fn = make_basket_feature_price(
        weights=jnp.array([1.0]),
        corr=uniform_correlation(1, 0.0),
        sigmas=jnp.array([SIGMA]),
        r=R,
        num_paths=60_000,
        num_steps=50,
    )

    price = float(price_fn(jnp.array([SPOT, 100.0, 1.0])))
    analytic = float(black_scholes_price_single(SPOT, 100.0, 1.0, SIGMA, R))

    assert abs(price - analytic) < 0.5


def test_basket_price_is_differentiable_twice():
    price_fn = make_basket_feature_price(
        weights=jnp.full(3, 1.0 / 3.0),
        corr=uniform_correlation(3, 0.5),
        sigmas=jnp.full(3, SIGMA),
        r=R,
        num_paths=8_000,
        num_steps=20,
    )

    x = jnp.array([100.0, 100.0, 100.0, 100.0, 1.0])

    gradient = jax.grad(price_fn)(x)
    hvp = jax.jvp(jax.grad(price_fn), (x,), (jnp.ones(5),))[1]

    assert gradient.shape == (5,)
    assert hvp.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(hvp)))

    # every asset delta is positive and bounded by its weight
    assert bool(jnp.all(gradient[:3] > 0.0))
    assert bool(jnp.all(gradient[:3] < 1.0 / 3.0))

    # dPrice/dK is negative for a call
    assert float(gradient[3]) < 0.0


def test_uniform_correlation_matrix():
    corr = uniform_correlation(3, 0.5)

    assert corr.shape == (3, 3)
    assert bool(jnp.all(jnp.diag(corr) == 1.0))
    assert float(corr[0, 1]) == 0.5
    assert bool(jnp.all(corr == corr.T))


# ----------------------------------------------------------------- dataset

def test_black_scholes_dataset_shapes():
    dataset = _dataset(BLACK_SCHOLES, n_samples=3)

    assert dataset.X.shape == (3, 5)
    assert dataset.y.shape == (3,)
    assert dataset.gradients.shape == (3, 5)
    assert dataset.hvps.shape == (3, 5)
    assert dataset.V.shape == (3, 5)


def test_basket_dataset_shapes_follow_the_asset_count():
    basket = BasketConfig(n_assets=3, num_paths=4_000, num_steps=20)

    dataset = _dataset(BASKET_BLACK_SCHOLES, basket=basket, n_samples=3)

    assert dataset.X.shape == (3, 5)
    assert dataset.gradients.shape == (3, 5)
    assert dataset.input_dim == 5

    wider = _dataset(
        BASKET_BLACK_SCHOLES,
        basket=BasketConfig(n_assets=4, num_paths=4_000, num_steps=20),
        n_samples=2,
    )

    assert wider.X.shape == (2, 6)
    assert wider.input_dim == 6


def test_basket_domain_respects_the_market_ranges():
    basket = BasketConfig(n_assets=3, num_paths=4_000, num_steps=20)

    dataset = _dataset(BASKET_BLACK_SCHOLES, basket=basket, n_samples=8)

    strikes, maturities = dataset.X[:, 3], dataset.X[:, 4]

    assert bool(jnp.all(strikes >= 80.0)) and bool(jnp.all(strikes <= 120.0))
    assert bool(jnp.all(maturities >= 0.25)) and bool(jnp.all(maturities <= 1.0))

    # all three spot columns are drawn from the same range
    assert bool(jnp.all(dataset.X[:, :3] > 0.0))


def test_sobolev_order_one_skips_the_hvps():
    dataset = _dataset(BLACK_SCHOLES, n_samples=2, sobolev_order=1)

    assert dataset.hvps is None
    assert dataset.gradients is not None


def test_unknown_pricing_model_rejected_by_the_generator():
    try:
        _dataset("heston", n_samples=1)
    except ValueError as e:
        assert "heston" in str(e)
    else:
        assert False, "an unknown pricing model should raise"


if __name__ == "__main__":
    for check in [
        test_default_is_black_scholes,
        test_basket_selection_changes_the_feature_names,
        test_unknown_pricing_model_is_rejected,
        test_weight_length_must_match_asset_count,
        test_network_input_width_is_derived_by_default,
        test_single_asset_basket_matches_the_analytic_call,
        test_basket_price_is_differentiable_twice,
        test_uniform_correlation_matrix,
        test_black_scholes_dataset_shapes,
        test_basket_dataset_shapes_follow_the_asset_count,
        test_basket_domain_respects_the_market_ranges,
        test_sobolev_order_one_skips_the_hvps,
        test_unknown_pricing_model_rejected_by_the_generator,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
