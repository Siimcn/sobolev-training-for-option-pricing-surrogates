"""
Sanity checks on model parameters, before anything is priced with them.

Rules are keyed by **field name**, not by parameter class. The six parameter
types share their fields - `sigma`, `kappa`, `weights`, `corr` and the rest
recur across models - so a per-class validator would state the same rule up to
six times, and a seventh model would need a seventh copy. Keyed by field, each
rule is written once and a new model is covered the moment it reuses a field
name.

Not every field has a rule. `r` deliberately has none: negative interest rates
are real, and a calibration that finds one is reporting the market, not a bug.
"""

import jax
import jax.numpy as jnp

__all__ = ["FIELD_RULES", "ValidationError", "validate_params"]


class ValidationError(ValueError):
    """A model parameter is outside the range its model is defined on."""


def _positive(name, value, owner):
    if jnp.any(jnp.asarray(value) <= 0):
        raise ValidationError(f"{owner}: {name} must be > 0, got {name}={value}")


def _unit_interval(name, value, owner):
    v = jnp.asarray(value)

    if jnp.any(v < -1.0) or jnp.any(v > 1.0):
        raise ValidationError(f"{owner}: {name} must be in [-1, 1], got {name}={value}")


def _weights(name, value, owner):
    v = jnp.asarray(value)

    if jnp.any(v < 0):
        raise ValidationError(f"{owner}: all {name} must be >= 0, got {name}={value}")

    if not jnp.isclose(jnp.sum(v), 1.0):
        raise ValidationError(
            f"{owner}: {name} must sum to 1, got sum({name})={float(jnp.sum(v))}"
        )


def _correlation_matrix(name, value, owner):
    v = jnp.asarray(value)

    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValidationError(
            f"{owner}: {name} must be a square matrix, got shape {v.shape}"
        )

    if not jnp.allclose(v, v.T):
        raise ValidationError(f"{owner}: {name} must be symmetric, got {name}={value}")

    if not jnp.allclose(jnp.diag(v), 1.0):
        raise ValidationError(
            f"{owner}: {name} must have 1s on the diagonal, got {name}={value}"
        )


# field name -> the rule that field must satisfy, whichever model it belongs to
FIELD_RULES = {
    "sigma": _positive,
    "sigmas": _positive,
    "kappa": _positive,
    "theta": _positive,
    "xi": _positive,
    "nu0": _positive,
    "rho": _unit_interval,
    "weights": _weights,
    "corr": _correlation_matrix,
}


def validate_params(params) -> None:
    """
    Raise `ValidationError` if any field of `params` breaks its rule.

    Call this on concrete values only - at calibration entry and exit, or
    before a run starts. It compares numbers with Python `if`, which cannot
    work on a traced value, so it must not be called inside `jit`, `grad` or
    `vmap`; a tracer is reported as such rather than left to fail later with a
    `ConcretizationError` from somewhere less obvious.
    """

    fields = getattr(params, "_asdict", None)

    if fields is None:
        raise TypeError(
            f"validate_params expects a parameter NamedTuple, got "
            f"{type(params).__name__}."
        )

    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(params)
    ):
        raise TypeError(
            f"validate_params was called on traced {type(params).__name__}. "
            f"Validation compares concrete numbers, so it belongs outside "
            f"jit/grad/vmap - validate before entering the traced region."
        )

    owner = type(params).__name__

    for name, value in fields().items():
        rule = FIELD_RULES.get(name)

        if rule is not None:
            rule(name, value, owner)


# The Feller condition (2 kappa theta >= xi^2) is deliberately not checked here.
# Violating it is legitimate - the variance simply reaches zero, which the
# smooth positive-part scheme is built to handle, and roughly half of the
# shipped Heston sampling domain sits below it by design. The ratio is reported
# at calibration instead, where a human can weigh it.
