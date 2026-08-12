import os
import sys
import tempfile

from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

import jax

jax.config.update("jax_enable_x64", True)

from calibration.calibrator import Calibrator
from calibration.market_data import MarketData
from calibration.market_data_loader import MarketDataLoader, OptionChainError
from market_simulation.black_scholes import black_scholes_price
from market_simulation.pricing_model import BlackScholesParams

SPOT = 100.0
TRUE_PARAMS = BlackScholesParams(r=0.03, sigma=0.25)


def _grid():
    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0] * 4)
    maturities = jnp.repeat(jnp.array([0.25, 0.5, 1.0, 2.0]), 5)
    is_call = jnp.array([True, True, False, False, True] * 4)

    return strikes, maturities, is_call


def _synthetic_market(params=TRUE_PARAMS):
    strikes, maturities, is_call = _grid()

    prices = black_scholes_price(
        params=params,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )

    return MarketData(
        spot=SPOT,
        strikes=strikes,
        maturities=maturities,
        market_prices=prices,
        is_call=is_call,
    )


def _pricer(params, strikes, maturities, is_call):
    return black_scholes_price(
        params=params,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )


def test_market_data_defaults_weights_to_ones():
    market = _synthetic_market()

    assert market.weights.shape == market.market_prices.shape
    assert bool(jnp.all(market.weights == 1.0))


def test_market_data_rejects_inconsistent_lengths():
    strikes, maturities, is_call = _grid()

    for field in ["strikes", "maturities", "is_call", "weights"]:
        kwargs = dict(
            spot=SPOT,
            strikes=strikes,
            maturities=maturities,
            market_prices=jnp.ones_like(strikes),
            is_call=is_call,
        )

        if field == "weights":
            kwargs["weights"] = jnp.ones(len(strikes) - 1)
        else:
            kwargs[field] = kwargs[field][:-1]

        try:
            MarketData(**kwargs)
        except ValueError:
            pass
        else:
            assert False, f"inconsistent {field} should raise"


def test_market_data_call_put_split():
    market = _synthetic_market()

    n_calls = int(jnp.sum(market.is_call))

    assert len(market.calls) == n_calls
    assert len(market.puts) == len(market) - n_calls
    assert bool(jnp.all(market.calls.is_call))
    assert not bool(jnp.any(market.puts.is_call))
    assert market.num_instruments == len(market)


def test_calibration_tuple_order():
    market = _synthetic_market()

    strikes, maturities, prices, is_call, weights = market.calibration_tuple()

    assert bool(jnp.all(strikes == market.strikes))
    assert bool(jnp.all(maturities == market.maturities))
    assert bool(jnp.all(prices == market.market_prices))
    assert bool(jnp.all(is_call == market.is_call))
    assert bool(jnp.all(weights == market.weights))


def test_calibrator_recovers_known_parameters():
    market = _synthetic_market()

    fitted, _ = Calibrator(pricing_fn=_pricer).calibrate(
        BlackScholesParams(r=0.06, sigma=0.15), market
    )

    assert abs(float(fitted.sigma) - TRUE_PARAMS.sigma) < 1e-4
    assert abs(float(fitted.r) - TRUE_PARAMS.r) < 1e-4


def test_pricing_error_vanishes_at_true_parameters():
    market = _synthetic_market()

    errors = Calibrator(pricing_fn=_pricer).pricing_error(TRUE_PARAMS, market)

    assert errors["RMSE"] < 1e-10
    assert errors["MAE"] < 1e-10


def test_calibrator_applies_parameter_transform():
    market = _synthetic_market()

    calibrator = Calibrator(
        pricing_fn=_pricer,
        transform_fn=lambda p: BlackScholesParams(r=p.r, sigma=jnp.exp(p.sigma)),
        inv_transform_fn=lambda p: BlackScholesParams(r=p.r, sigma=jnp.log(p.sigma)),
    )

    fitted, _ = calibrator.calibrate(BlackScholesParams(r=0.06, sigma=0.15), market)

    assert abs(float(fitted.sigma) - TRUE_PARAMS.sigma) < 1e-4
    assert float(fitted.sigma) > 0.0


def test_zero_weights_remove_instruments_from_the_fit():
    strikes, maturities, is_call = _grid()

    prices = black_scholes_price(
        params=TRUE_PARAMS,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )

    corrupted = prices.at[:5].set(prices[:5] + 50.0)
    weights = jnp.concatenate([jnp.zeros(5), jnp.ones(len(strikes) - 5)])

    market = MarketData(
        spot=SPOT,
        strikes=strikes,
        maturities=maturities,
        market_prices=corrupted,
        is_call=is_call,
        weights=weights,
    )

    fitted, _ = Calibrator(pricing_fn=_pricer).calibrate(
        BlackScholesParams(r=0.06, sigma=0.15), market
    )

    assert abs(float(fitted.sigma) - TRUE_PARAMS.sigma) < 1e-4


def test_market_data_loader_cache_round_trip():
    strikes, maturities, is_call = _grid()

    prices = black_scholes_price(
        params=TRUE_PARAMS,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "chain.json")

        MarketDataLoader.save_cache(
            path, "TEST", SPOT, strikes, maturities, prices, is_call
        )

        assert os.path.exists(path)

        loaded = MarketDataLoader.load_cache(path)

        assert abs(loaded[0] - SPOT) < 1e-12
        assert bool(jnp.allclose(loaded[1], strikes))
        assert bool(jnp.allclose(loaded[2], maturities))
        assert bool(jnp.allclose(loaded[3], prices))
        assert bool(jnp.all(loaded[4] == is_call))


def test_fetch_uses_cache_without_network():
    strikes, maturities, is_call = _grid()

    prices = black_scholes_price(
        params=TRUE_PARAMS,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "chain.json")

        MarketDataLoader.save_cache(
            path, "TEST", SPOT, strikes, maturities, prices, is_call
        )

        spot, k, t, p, c = MarketDataLoader.fetch_yahoo_options(
            ticker_symbol="TEST", cache_path=path, use_cache=True
        )

        assert abs(spot - SPOT) < 1e-12
        assert len(k) == len(strikes)


@contextmanager
def _yahoo_failing_with(error: Exception):
    """Swap in a fetch that fails the way a dead connection does."""

    original = MarketDataLoader.__dict__["_fetch_from_yahoo"]

    def fail(**kwargs):
        raise error

    MarketDataLoader._fetch_from_yahoo = staticmethod(fail)

    try:
        yield
    finally:
        MarketDataLoader._fetch_from_yahoo = original


def _cached_chain(path):
    strikes, maturities, is_call = _grid()

    prices = black_scholes_price(
        params=TRUE_PARAMS,
        strikes=strikes,
        maturities=maturities,
        is_call=is_call,
        spot=SPOT,
    )

    MarketDataLoader.save_cache(
        path, "TEST", SPOT, strikes, maturities, prices, is_call
    )

    return strikes


def test_failed_fetch_falls_back_to_the_cache():
    """use_cache=False asks for a fresh chain; a stale one still beats none."""

    dead_connection = IndexError("single positional indexer is out-of-bounds")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "chain.json")

        strikes = _cached_chain(path)

        with _yahoo_failing_with(dead_connection):
            spot, k, _, _, _ = MarketDataLoader.fetch_yahoo_options(
                ticker_symbol="TEST", cache_path=path, use_cache=False
            )

        assert abs(spot - SPOT) < 1e-12
        assert len(k) == len(strikes)


def test_failed_fetch_without_a_cache_names_the_connection():
    dead_connection = IndexError("single positional indexer is out-of-bounds")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "missing.json")

        with _yahoo_failing_with(dead_connection):
            try:
                MarketDataLoader.fetch_yahoo_options(
                    ticker_symbol="TEST", cache_path=path, use_cache=True
                )
            except ConnectionError as error:
                message = str(error)
            else:
                raise AssertionError("expected a ConnectionError")

        assert "TEST" in message
        assert path in message


def test_an_empty_chain_is_not_disguised_as_a_connection_failure():
    """The fetch reached Yahoo; the filters emptied it. Keep the diagnosis."""

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "missing.json")

        with _yahoo_failing_with(OptionChainError("no liquid options remained")):
            try:
                MarketDataLoader.fetch_yahoo_options(
                    ticker_symbol="TEST", cache_path=path, use_cache=True
                )
            except OptionChainError as error:
                assert "no liquid options" in str(error)
            else:
                raise AssertionError("expected an OptionChainError")


if __name__ == "__main__":
    for check in [
        test_market_data_defaults_weights_to_ones,
        test_market_data_rejects_inconsistent_lengths,
        test_market_data_call_put_split,
        test_calibration_tuple_order,
        test_calibrator_recovers_known_parameters,
        test_pricing_error_vanishes_at_true_parameters,
        test_calibrator_applies_parameter_transform,
        test_zero_weights_remove_instruments_from_the_fit,
        test_market_data_loader_cache_round_trip,
        test_fetch_uses_cache_without_network,
        test_failed_fetch_falls_back_to_the_cache,
        test_failed_fetch_without_a_cache_names_the_connection,
        test_an_empty_chain_is_not_disguised_as_a_connection_failure,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
