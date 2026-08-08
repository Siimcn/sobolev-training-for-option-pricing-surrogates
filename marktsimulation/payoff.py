import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from typing import Callable


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