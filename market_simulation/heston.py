import jax
import jax.numpy as jnp
import numpy as np

"""
Heston (stochastic volatility) model: semi-analytic prices.
"""


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(160)


def _quadrature(u_max):
    """Map the Gauss-Legendre rule from [-1, 1] onto (0, u_max)."""

    half = 0.5 * u_max

    return (jnp.asarray(half * (_GL_NODES + 1.0)), jnp.asarray(half * _GL_WEIGHTS))


def heston_characteristic(u, spot, maturity, params):
    """E[exp(i u ln S_T)] under the risk-neutral measure."""

    xi2 = params.xi**2

    rho_xi_iu = params.rho * params.xi * 1j * u

    d = jnp.sqrt((rho_xi_iu - params.kappa) ** 2 + xi2 * (1j * u + u**2))

    minus = params.kappa - rho_xi_iu - d
    plus = params.kappa - rho_xi_iu + d

    g = minus / plus

    decay = jnp.exp(-d * maturity)

    drift_term = (params.kappa * params.theta / xi2) * (
        minus * maturity - 2.0 * jnp.log((1.0 - g * decay) / (1.0 - g))
    )

    variance_term = (minus / xi2) * ((1.0 - decay) / (1.0 - g * decay))

    forward = jnp.log(spot) + params.r * maturity

    return jnp.exp(1j * u * forward + drift_term + variance_term * params.nu0)


def _probability(spot, strike, maturity, params, shift, u_max):
    """Gil-Pelaez inversion, P_1 with `shift = 1` and P_2 with `shift = 0`."""

    nodes, weights = _quadrature(u_max)

    def integrand(u):
        numerator = heston_characteristic(u - 1j * shift, spot, maturity, params)

        if shift:
            numerator = numerator / heston_characteristic(-1j, spot, maturity, params)

        return jnp.real(jnp.exp(-1j * u * jnp.log(strike)) * numerator / (1j * u))

    return 0.5 + jnp.sum(weights * jax.vmap(integrand)(nodes)) / jnp.pi


def heston_price(
    spot, strike, maturity, params, is_call: bool = True, u_max: float = 200.0
):
    """European price by Fourier inversion."""

    maturity = jnp.maximum(maturity, 1e-8)

    p1 = _probability(spot, strike, maturity, params, 1, u_max)
    p2 = _probability(spot, strike, maturity, params, 0, u_max)

    discount = jnp.exp(-params.r * maturity)

    call = spot * p1 - strike * discount * p2

    return jnp.where(is_call, call, call - spot + strike * discount)


def heston_price_vector(params, strikes, maturities, is_call, spot):
    """Vectorised over instruments, in the Calibrator's argument order."""

    return jax.vmap(lambda k, t, c: heston_price(spot, k, t, params, is_call=c))(
        strikes, maturities, is_call
    )


def feller_ratio(params) -> float:
    """
    2 kappa theta / xi^2. At or above 1 the variance cannot reach zero; below
    it the process touches zero and every discretisation has to decide what to
    do there.
    """

    return float(2.0 * params.kappa * params.theta / params.xi**2)
