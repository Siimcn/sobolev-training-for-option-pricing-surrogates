import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from dataclasses import dataclass

from typing import Callable, Dict, Optional, Tuple


def relu(
    x: jnp.ndarray,
    w: float = 0.0,
) -> jnp.ndarray:

    return jnp.maximum(
        x,
        0.0,
    )


def sigmoid_smooth(
    x: jnp.ndarray,
    w: float = 0.01,
) -> jnp.ndarray:

    return x * jnn.sigmoid(
        x / w
    )


def cubic_spline_smooth(
    x: jnp.ndarray,
    w: float = 0.01,
) -> jnp.ndarray:

    inside = (
        -x**4 / (16 * w**3)
        + 3 * x**2 / (8 * w)
        + 0.5 * x
        + 3 * w / 16
    )

    return jnp.where(
        x < -w,
        0.0,
        jnp.where(
            x > w,
            x,
            inside,
        ),
    )


class Payoff(eqx.Module):

    strike: float
    omega: float

    smooth_fn: Callable = eqx.field(
        static=True
    )

    smooth_w: float

    def __call__(self, x):
        raise NotImplementedError
    

class EuropeanPayoff(
    Payoff
):

    def __call__(
        self,
        spot: jnp.ndarray,
    ) -> jnp.ndarray:

        intrinsic = (
            self.omega
            * (spot - self.strike)
        )

        return self.smooth_fn(
            intrinsic,
            self.smooth_w,
        )
    

class AsianPayoff(
    Payoff
):

    def __call__(
        self,
        path: jnp.ndarray,
    ) -> jnp.ndarray:

        average_price = jnp.mean(
            path[1:]
        )

        intrinsic = (
            self.omega
            * (
                average_price
                - self.strike
            )
        )

        return self.smooth_fn(
            intrinsic,
            self.smooth_w,
        )
    

def EuropeanCall(
    strike: float,
    smooth_fn=sigmoid_smooth,
    smooth_w: float = 0.05,
):

    return EuropeanPayoff(
        strike=strike,
        omega=1.0,
        smooth_fn=smooth_fn,
        smooth_w=smooth_w,
    )


def EuropeanPut(
    strike: float,
    smooth_fn=sigmoid_smooth,
    smooth_w: float = 0.05,
):

    return EuropeanPayoff(
        strike=strike,
        omega=-1.0,
        smooth_fn=smooth_fn,
        smooth_w=smooth_w,
    )


def AsianCall(
    strike: float,
    smooth_fn=sigmoid_smooth,
    smooth_w: float = 0.01,
):

    return AsianPayoff(
        strike=strike,
        omega=1.0,
        smooth_fn=smooth_fn,
        smooth_w=smooth_w,
    )


def AsianPut(
    strike: float,
    smooth_fn=sigmoid_smooth,
    smooth_w: float = 0.01,
):

    return AsianPayoff(
        strike=strike,
        omega=-1.0,
        smooth_fn=smooth_fn,
        smooth_w=smooth_w,
    )


# --------------------------------------------------------------- registry


@dataclass(frozen=True)
class PayoffSpec:
    """
    Everything the pricing and validation layers need to know about a payoff.

    `path_dependent` decides whether the terminal state is enough: a
    European payoff reads only S(T) and can therefore be priced by an
    exact one-step draw, an Asian payoff needs the whole trajectory and
    forces a stepping scheme.

    `bounds` returns model-free (lower, upper) price bounds, or None when
    no simple bound is known. The validation stage checks whatever is
    offered and skips the rest rather than inventing one.
    """

    name: str
    build: Callable[..., Payoff]
    path_dependent: bool
    omega: float = 1.0
    bounds: Optional[Callable[[float, float, float], Tuple[float, float]]] = None

    @property
    def is_call(self) -> bool:
        return self.omega > 0.0


def _call_bounds(underlying, strike, discount):
    """A call is worth at least its discounted intrinsic, never more than
    the underlying."""

    return max(float(underlying - strike * discount), 0.0), float(underlying)


def _put_bounds(underlying, strike, discount):
    """Mirror image: never worth more than the discounted strike."""

    discounted_strike = float(strike * discount)

    return max(discounted_strike - float(underlying), 0.0), discounted_strike


_PAYOFFS: Dict[str, PayoffSpec] = {}


def register_payoff(spec: PayoffSpec, overwrite: bool = False) -> None:
    """Make `spec.name` selectable as `payoff.name` in the configuration."""

    key = spec.name.lower()

    if key in _PAYOFFS and not overwrite:
        raise ValueError(
            f"A payoff named '{spec.name}' is already registered. "
            f"Pass overwrite=True to replace it."
        )

    _PAYOFFS[key] = spec


def available_payoffs() -> Tuple[str, ...]:
    return tuple(sorted(_PAYOFFS))


def payoff_spec(name: str) -> PayoffSpec:
    key = name.lower()

    if key not in _PAYOFFS:
        raise ValueError(
            f"Unknown payoff '{name}'. "
            f"Expected one of: {', '.join(available_payoffs())}."
        )

    return _PAYOFFS[key]


def build_payoff(
    name: str,
    strike,
    smooth_w,
    smooth_fn=sigmoid_smooth,
) -> Payoff:
    """Construct the registered payoff `name`."""

    return payoff_spec(name).build(
        strike=strike,
        smooth_fn=smooth_fn,
        smooth_w=smooth_w,
    )


register_payoff(
    PayoffSpec("european_call", EuropeanCall, False, omega=1.0, bounds=_call_bounds)
)
register_payoff(
    PayoffSpec("european_put", EuropeanPut, False, omega=-1.0, bounds=_put_bounds)
)

# an arithmetic average is less volatile than the terminal value, so the
# European bounds do not carry over; none is declared rather than a wrong one
register_payoff(PayoffSpec("asian_call", AsianCall, True, omega=1.0))
register_payoff(PayoffSpec("asian_put", AsianPut, True, omega=-1.0))