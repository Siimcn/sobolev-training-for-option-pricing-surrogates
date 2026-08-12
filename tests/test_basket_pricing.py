import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from market_simulation.pricing_model import (
    BasketBlackScholesModel,
    BasketBlackScholesParams,
    BasketHestonModel,
    BasketHestonParams,
)
from market_simulation.timesteppingscheme import EulerMaruyama
from market_simulation.payoff import EuropeanCall, AsianCall, AsianPut, relu
from market_simulation.monte_carlo_pricer import MonteCarloPricer
from market_simulation.basket_mc import basket_price, basket_greeks
from market_simulation.black_scholes import black_scholes_price_single

R = 0.05
T = 1.0
K = 100.0


def _bs_model():
    return BasketBlackScholesModel(scheme=EulerMaruyama())


def _bs_params(sigmas, rho=0.5):
    sigmas = jnp.asarray(sigmas)
    n = sigmas.shape[0]

    corr = jnp.full((n, n), rho).at[jnp.diag_indices(n)].set(1.0)

    return BasketBlackScholesParams(
        r=R, sigmas=sigmas, weights=jnp.full(n, 1.0 / n), corr=corr
    )


def _heston_model_and_params():
    model = BasketHestonModel(scheme=EulerMaruyama())

    params = BasketHestonParams(
        r=R,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        nu0=0.04,
        weights=jnp.array([0.5, 0.5]),
        corr=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    return model, params


def _price(
    model, params, payoff, s0, key, on_path=False, num_paths=80_000, num_steps=50
):
    """
    Discounted basket price. relu keeps the payoff unbiased for the analytic
    comparisons; basket_mc uses a smooth one for the greeks.
    """

    pricer = MonteCarloPricer(
        model,
        payoff,
        value_fn=lambda s: model.basket_value(s, params),
        payoff_on_path=on_path,
    )

    undiscounted = pricer.price(
        s0=s0,
        params=params,
        maturity=T,
        num_paths=num_paths,
        num_steps=num_steps,
        key=key,
    )

    return float(undiscounted * jnp.exp(-params.r * T))


def _discounted_mean_basket(model, params, s0, key, num_paths, num_steps):
    paths = model.scheme.generate_paths(
        s0=s0,
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=key,
        num_paths=num_paths,
        num_steps=num_steps,
        dt=T / num_steps,
        corr=model.noise_correlation(params),
    )

    basket_T = jax.vmap(lambda s: model.basket_value(s, params))(paths[:, -1])

    return float(jnp.mean(basket_T) * jnp.exp(-params.r * T))


def test_single_asset_basket_matches_analytic():
    model = _bs_model()
    params = _bs_params([0.2])

    price = _price(
        model,
        params,
        EuropeanCall(strike=K, smooth_fn=relu),
        jnp.array([100.0]),
        jax.random.PRNGKey(0),
    )

    analytic = float(black_scholes_price_single(100.0, K, T, 0.2, R))

    assert abs(price - analytic) < 0.3


def test_perfectly_correlated_basket_matches_vanilla():
    model = _bs_model()
    params = _bs_params([0.2, 0.2], rho=0.999999)

    price = _price(
        model,
        params,
        EuropeanCall(strike=K, smooth_fn=relu),
        jnp.array([100.0, 100.0]),
        jax.random.PRNGKey(0),
    )

    analytic = float(black_scholes_price_single(100.0, K, T, 0.2, R))

    assert abs(price - analytic) < 0.3


def test_correlation_increases_basket_call():
    model = _bs_model()
    key = jax.random.PRNGKey(7)

    prices = [
        _price(
            model,
            _bs_params([0.2, 0.2], rho=rho),
            EuropeanCall(strike=K, smooth_fn=relu),
            jnp.array([100.0, 100.0]),
            key,
        )
        for rho in [0.0, 0.5, 0.9]
    ]

    assert prices[0] < prices[1] < prices[2]


def test_basket_martingale():
    model = _bs_model()
    params = _bs_params([0.2, 0.3])
    s0 = jnp.array([100.0, 100.0])

    disc_mean = _discounted_mean_basket(
        model, params, s0, jax.random.PRNGKey(1), 80_000, 50
    )

    assert abs(disc_mean - float(model.basket_value(s0, params))) < 0.5


def test_heston_basket_runs():
    model, params = _heston_model_and_params()

    s0 = jnp.array([100.0, 100.0, 0.04, 0.04])

    price = _price(
        model,
        params,
        EuropeanCall(strike=K, smooth_fn=relu),
        s0,
        jax.random.PRNGKey(2),
        num_paths=50_000,
        num_steps=100,
    )

    assert jnp.isfinite(price)
    assert price > 0.0


def test_heston_basket_martingale():
    model, params = _heston_model_and_params()
    s0 = jnp.array([100.0, 100.0, 0.04, 0.04])

    disc_mean = _discounted_mean_basket(
        model, params, s0, jax.random.PRNGKey(3), 50_000, 100
    )

    assert abs(disc_mean - float(model.basket_value(s0, params))) < 0.7


def test_asian_basket_below_european():
    model = _bs_model()
    params = _bs_params([0.2, 0.3])
    s0 = jnp.array([100.0, 100.0])
    key = jax.random.PRNGKey(3)

    euro = _price(model, params, EuropeanCall(strike=K, smooth_fn=relu), s0, key)
    asian = _price(
        model, params, AsianCall(strike=K, smooth_fn=relu), s0, key, on_path=True
    )

    assert jnp.isfinite(asian)
    assert asian > 0.0
    assert asian < euro


def test_asian_basket_put_call_parity():
    model = _bs_model()
    params = _bs_params([0.2, 0.3])
    s0 = jnp.array([100.0, 100.0])
    key = jax.random.PRNGKey(3)
    n = 50

    call = _price(
        model,
        params,
        AsianCall(strike=K, smooth_fn=relu),
        s0,
        key,
        on_path=True,
        num_steps=n,
    )
    put = _price(
        model,
        params,
        AsianPut(strike=K, smooth_fn=relu),
        s0,
        key,
        on_path=True,
        num_steps=n,
    )

    forwards = float(jnp.sum(jnp.exp(params.r * (T / n) * jnp.arange(1, n + 1))))
    expected_average = float(jnp.sum(params.weights * s0)) / n * forwards
    parity = float(jnp.exp(-params.r * T) * (expected_average - K))

    assert abs((call - put) - parity) < 0.1


def test_basket_greeks_match_finite_differences():
    model = _bs_model()
    params = _bs_params([0.2, 0.3])
    s0 = jnp.array([100.0, 100.0])
    key = jax.random.PRNGKey(5)

    greeks = basket_greeks(model, params, s0, K, T, key)

    h = 0.5
    bumped = jnp.array(
        [
            (
                basket_price(model, params, s0.at[i].add(h), K, T, key)
                - basket_price(model, params, s0.at[i].add(-h), K, T, key)
            )
            / (2 * h)
            for i in range(2)
        ]
    )

    assert float(jnp.max(jnp.abs(greeks["delta"] - bumped))) < 1e-3
    assert greeks["gamma"].shape == (2, 2)


def test_basket_delta_bounds():
    model = _bs_model()
    params = _bs_params([0.2, 0.3])

    delta = basket_greeks(
        model, params, jnp.array([100.0, 100.0]), K, T, jax.random.PRNGKey(5)
    )["delta"]

    assert bool(jnp.all(delta > 0.0))
    assert bool(jnp.all(delta < params.weights))


if __name__ == "__main__":
    for check in [
        test_single_asset_basket_matches_analytic,
        test_perfectly_correlated_basket_matches_vanilla,
        test_correlation_increases_basket_call,
        test_basket_martingale,
        test_heston_basket_runs,
        test_heston_basket_martingale,
        test_asian_basket_below_european,
        test_asian_basket_put_call_parity,
        test_basket_greeks_match_finite_differences,
        test_basket_delta_bounds,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
