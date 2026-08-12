import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.black_scholes import black_scholes_price_single
from conftest import bs_mc_feature_price
from marktsimulation.pricing_model import BlackScholesParams
from marktsimulation.sobolev_labels import label_keys
from pipeline.config import (
    BLACK_SCHOLES,
    DataConfig,
    ExperimentConfig,
    SimulationConfig,
)
from surrogate_modeling.pricing_problem import CalibrationResult, build_problem

SPOT, SIGMA, R = 100.0, 0.2, 0.05

N_POINTS, N_SEEDS, N_PATHS = 32, 8, 4_000


def _problem():
    market_data = MarketData(
        spot=SPOT,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )

    return build_problem(
        BLACK_SCHOLES,
        config=ExperimentConfig(
            simulation=SimulationConfig(num_paths=N_PATHS),
            data=DataConfig(pricing_model=BLACK_SCHOLES),
        ),
        market_data=market_data,
        calibration=CalibrationResult(params=BlackScholesParams(r=R, sigma=SIGMA)),
    )


def _error_matrix(points, reference, shared):
    """(n_seeds, n_points) label error, one row per base seed."""

    rows = []

    for seed in range(N_SEEDS):
        keys = label_keys(1000 + seed, len(points), shared=shared)

        labels = jnp.array(
            [
                float(bs_mc_feature_price(x, k, num_paths=N_PATHS))
                for x, k in zip(points, keys)
            ]
        )

        rows.append(labels - reference)

    return jnp.stack(rows)


def _setup():
    problem = _problem()

    points = problem.reference_points(n_points=N_POINTS, seed=7)

    reference = jnp.array([float(problem.analytic_price(x)) for x in points])

    return points, reference


def test_shared_keys_make_the_label_errors_coherent():
    """Documents the failure mode the default avoids."""

    points, reference = _setup()

    def mean_correlation(shared):
        errors = _error_matrix(points, reference, shared=shared)

        standardised = (errors - jnp.mean(errors, axis=0)) / (
            jnp.std(errors, axis=0) + 1e-12
        )

        correlation = (standardised.T @ standardised) / errors.shape[0]

        return float(jnp.mean(correlation[~jnp.eye(len(points), dtype=bool)]))

    assert mean_correlation(shared=True) > 0.4
    assert abs(mean_correlation(shared=False)) < 0.2


def test_independent_keys_let_the_dataset_wide_error_average_out():
    """
    The property that matters for training: across base seeds, the mean label
    error of an independently keyed dataset is far tighter than a shared-key
    one, because the individual errors cancel instead of adding.
    """

    points, reference = _setup()

    def spread(shared):
        return float(
            jnp.std(jnp.mean(_error_matrix(points, reference, shared=shared), axis=1))
        )

    shared_spread = spread(shared=True)
    independent_spread = spread(shared=False)

    assert shared_spread > 2.0 * independent_spread


def test_per_label_accuracy_is_the_same_in_both_modes():
    """
    Independent keys are not "more accurate per label" - one label is one
    label. Only the way the errors combine changes, which is what makes this a
    bias question rather than a variance one.
    """

    points, reference = _setup()

    shared = float(jnp.mean(jnp.abs(_error_matrix(points, reference, shared=True))))
    independent = float(
        jnp.mean(jnp.abs(_error_matrix(points, reference, shared=False)))
    )

    assert abs(shared - independent) < 0.5 * max(shared, independent)


def test_default_configuration_uses_independent_keys():
    assert ExperimentConfig().simulation.shared_label_keys is False


def test_a_default_dataset_is_unbiased_against_the_closed_form():
    """
    End to end: the labels a default run produces sit on the analytic price up
    to the payoff-smoothing offset, which is small and negative.
    """

    points, reference = _setup()

    keys = label_keys(0, len(points))

    labels = jnp.array(
        [
            float(bs_mc_feature_price(x, k, num_paths=20_000))
            for x, k in zip(points, keys)
        ]
    )

    relative_bias = float(jnp.mean(labels - reference) / jnp.mean(reference))

    assert -0.01 < relative_bias < 0.005, f"label bias {relative_bias:+.4%}"


if __name__ == "__main__":
    for check in [
        test_shared_keys_make_the_label_errors_coherent,
        test_independent_keys_let_the_dataset_wide_error_average_out,
        test_per_label_accuracy_is_the_same_in_both_modes,
        test_default_configuration_uses_independent_keys,
        test_a_default_dataset_is_unbiased_against_the_closed_form,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
