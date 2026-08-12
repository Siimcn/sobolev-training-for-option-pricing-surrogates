"""Parameter validation: what it accepts, what it rejects, and where it may run."""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_simulation.basket_mc import uniform_correlation
from market_simulation.param_validation import (
    FIELD_RULES,
    ValidationError,
    validate_params,
)
from market_simulation.pricing_model import (
    BachelierParams,
    BasketBachelierParams,
    BasketBlackScholesParams,
    BasketHestonParams,
    BlackScholesParams,
    HestonParams,
)

N = 3
WEIGHTS = jnp.full(N, 1.0 / N)
SIGMAS = jnp.full(N, 0.2)
CORR = uniform_correlation(N, 0.5)


def _valid():
    """One admissible instance of every parameter type."""

    return [
        BlackScholesParams(r=0.03, sigma=0.2),
        BachelierParams(sigma=20.0, r=0.03),
        HestonParams(r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, nu0=0.04),
        BasketBlackScholesParams(r=0.03, sigmas=SIGMAS, weights=WEIGHTS, corr=CORR),
        BasketBachelierParams(r=0.03, sigmas=SIGMAS, weights=WEIGHTS, corr=CORR),
        BasketHestonParams(
            r=0.03,
            kappa=2.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            nu0=0.04,
            weights=WEIGHTS,
            corr=CORR,
        ),
    ]


@pytest.mark.parametrize("params", _valid(), ids=lambda p: type(p).__name__)
def test_admissible_parameters_pass(params):
    validate_params(params)


@pytest.mark.parametrize("bad_sigma", [0.0, -0.2])
def test_non_positive_volatility_is_rejected(bad_sigma):
    with pytest.raises(ValidationError, match="sigma must be > 0"):
        validate_params(BlackScholesParams(r=0.03, sigma=bad_sigma))

    with pytest.raises(ValidationError, match="sigma must be > 0"):
        validate_params(BachelierParams(sigma=bad_sigma, r=0.03))


@pytest.mark.parametrize("field", ["kappa", "theta", "xi", "nu0"])
def test_non_positive_heston_parameters_are_rejected(field):
    good = dict(r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, nu0=0.04)

    with pytest.raises(ValidationError, match=f"{field} must be > 0"):
        validate_params(HestonParams(**{**good, field: -1.0}))


@pytest.mark.parametrize("bad_rho", [-1.5, 1.5])
def test_correlation_outside_the_unit_interval_is_rejected(bad_rho):
    with pytest.raises(ValidationError, match=r"rho must be in \[-1, 1\]"):
        validate_params(
            HestonParams(r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=bad_rho, nu0=0.04)
        )


def test_rho_at_the_boundary_is_accepted():
    """+-1 is degenerate but admissible; the comonotonic limit relies on it."""

    for rho in (-1.0, 1.0):
        validate_params(
            HestonParams(r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=rho, nu0=0.04)
        )


def test_one_bad_entry_in_a_vector_is_enough():
    with pytest.raises(ValidationError, match="sigmas must be > 0"):
        validate_params(
            BasketBlackScholesParams(
                r=0.03, sigmas=SIGMAS.at[1].set(-0.1), weights=WEIGHTS, corr=CORR
            )
        )


def test_negative_weights_are_rejected():
    with pytest.raises(ValidationError, match="weights must be >= 0"):
        validate_params(
            BasketBlackScholesParams(
                r=0.03, sigmas=SIGMAS, weights=jnp.array([-0.2, 0.6, 0.6]), corr=CORR
            )
        )


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="weights must sum to 1"):
        validate_params(
            BasketBlackScholesParams(
                r=0.03, sigmas=SIGMAS, weights=jnp.full(N, 0.2), corr=CORR
            )
        )


def test_asymmetric_correlation_is_rejected():
    with pytest.raises(ValidationError, match="corr must be symmetric"):
        validate_params(
            BasketBlackScholesParams(
                r=0.03, sigmas=SIGMAS, weights=WEIGHTS, corr=CORR.at[0, 1].set(0.9)
            )
        )


def test_correlation_diagonal_must_be_one():
    with pytest.raises(ValidationError, match="corr must have 1s on the diagonal"):
        validate_params(
            BasketBlackScholesParams(
                r=0.03, sigmas=SIGMAS, weights=WEIGHTS, corr=CORR.at[0, 0].set(0.5)
            )
        )


def test_a_negative_rate_is_allowed():
    """
    Negative rates are real. A calibration that finds one is reporting the
    market, not failing, so `r` deliberately carries no rule.
    """

    validate_params(BlackScholesParams(r=-0.005, sigma=0.2))

    assert "r" not in FIELD_RULES


def test_the_feller_condition_is_not_enforced():
    """
    Violating it is legitimate - the variance reaches zero and the smooth
    positive-part scheme handles it. Roughly half the shipped Heston sampling
    domain sits below it by design, so raising here would reject valid runs.
    """

    # 2 * kappa * theta = 0.008, xi^2 = 4.0
    validate_params(
        HestonParams(r=0.03, kappa=0.1, theta=0.04, xi=2.0, rho=-0.7, nu0=0.04)
    )


def test_a_traced_value_is_reported_clearly():
    """
    Validation compares concrete numbers. Called under `jit` it must say so,
    rather than failing later with a ConcretizationError from somewhere less
    obvious.
    """

    @jax.jit
    def run(sigma):
        validate_params(BlackScholesParams(r=0.03, sigma=sigma))
        return sigma

    with pytest.raises(TypeError, match="traced"):
        run(jnp.array(0.2))


def test_a_non_namedtuple_is_rejected():
    with pytest.raises(TypeError, match="NamedTuple"):
        validate_params({"sigma": 0.2})


def test_every_rule_is_reachable_from_some_parameter_type():
    """
    Guards the field-name coupling: a rule keyed to a field that no model has
    is dead, and a renamed field would silently stop being validated.
    """

    fields = set()
    for params in _valid():
        fields.update(params._fields)

    unreachable = set(FIELD_RULES) - fields

    assert not unreachable, f"rules that match no parameter field: {unreachable}"
