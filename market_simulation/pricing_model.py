import jax
import jax.numpy as jnp
import equinox as eqx

from typing import NamedTuple, Optional, Tuple

from market_simulation.timesteppingscheme import TimeSteppingScheme


class BachelierParams(NamedTuple):
    sigma: float
    r: float = 0.0


class BlackScholesParams(NamedTuple):
    r: float
    sigma: float


class HestonParams(NamedTuple):
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    nu0: float


class BasketBlackScholesParams(NamedTuple):
    r: float
    sigmas: jnp.ndarray
    weights: jnp.ndarray
    corr: jnp.ndarray


class BasketBachelierParams(NamedTuple):
    r: float
    sigmas: jnp.ndarray
    weights: jnp.ndarray
    corr: jnp.ndarray


class BasketHestonParams(NamedTuple):
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    nu0: float

    weights: jnp.ndarray
    corr: jnp.ndarray


class PricingModel(eqx.Module):

    scheme: TimeSteppingScheme

    def __init__(self, scheme: TimeSteppingScheme):
        self.scheme = scheme

    def drift(self, state: jnp.ndarray, params, t: float) -> jnp.ndarray:
        raise NotImplementedError

    def diffusion(self, state: jnp.ndarray, params, t: float) -> jnp.ndarray:
        raise NotImplementedError

    def noise_correlation(self, params):
        return None

    def terminal_state(
        self,
        s0: jnp.ndarray,
        params,
        maturity,
        num_paths: int,
        key: jnp.ndarray,
        antithetic: bool = True,
    ) -> Optional[Tuple[jnp.ndarray, ...]]:
        """
        Exact draws of the terminal state, or None when the model has no closed
        transition law.
        """

        return None

    def terminal_dispersion(self, s0: jnp.ndarray, params, maturity):
        """Standard deviation of whatever the payoff is written on at T."""

        raise NotImplementedError


VARIANCE_SMOOTHING = 0.01


def smooth_positive(v, width):
    """0.5 (v + sqrt(v^2 + w^2)): a positive part bounded away from zero."""

    return 0.5 * (v + jnp.sqrt(v * v + width * width))


def _correlated_normals(key, num_paths, n_assets, corr):
    """Cholesky-correlated standard normals, or plain ones when corr is None."""

    z = jax.random.normal(key, (num_paths, n_assets))

    if corr is None:
        return z

    return z @ jnp.linalg.cholesky(corr).T


class BachelierModel(PricingModel):

    def drift(self, state, params: BachelierParams, t):
        return jnp.zeros_like(state)

    def diffusion(self, state, params: BachelierParams, t):
        return jnp.full_like(state, params.sigma)

    def terminal_state(self, s0, params, maturity, num_paths, key, antithetic=True):
        """F_T = F_0 + sigma sqrt(T) Z, exactly."""

        z = _correlated_normals(key, num_paths, jnp.size(s0), None)

        move = params.sigma * jnp.sqrt(maturity) * z

        if not antithetic:
            return (s0 + move,)

        return s0 + move, s0 - move

    def terminal_dispersion(self, s0, params, maturity):
        return params.sigma * jnp.sqrt(jnp.maximum(maturity, 1e-12))


class BlackScholesModel(PricingModel):

    def drift(self, state, params: BlackScholesParams, t):
        return params.r * state

    def diffusion(self, state, params: BlackScholesParams, t):
        return params.sigma * state

    def terminal_state(self, s0, params, maturity, num_paths, key, antithetic=True):
        """S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z), exactly."""

        z = _correlated_normals(key, num_paths, jnp.size(s0), None)

        drift = (params.r - 0.5 * params.sigma**2) * maturity
        scale = params.sigma * jnp.sqrt(maturity)

        if not antithetic:
            return (s0 * jnp.exp(drift + scale * z),)

        return s0 * jnp.exp(drift + scale * z), s0 * jnp.exp(drift - scale * z)

    def terminal_dispersion(self, s0, params, maturity):
        return s0 * params.sigma * jnp.sqrt(jnp.maximum(maturity, 1e-12))


class HestonModel(PricingModel):
    """dS = r S dt + sqrt(v) S dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2."""

    def _variance(self, v, params):
        return smooth_positive(v, VARIANCE_SMOOTHING * params.theta)

    def drift(self, state, params: HestonParams, t):

        S = state[0]

        nu = self._variance(state[1], params)

        return jnp.array([params.r * S, params.kappa * (params.theta - nu)])

    def diffusion(self, state, params: HestonParams, t):

        S = state[0]

        nu = self._variance(state[1], params)

        sqrt_nu = jnp.sqrt(nu)

        return jnp.array([sqrt_nu * S, params.xi * sqrt_nu])

    def noise_correlation(self, params: HestonParams):
        return jnp.array([[1.0, params.rho], [params.rho, 1.0]])

    def terminal_dispersion(self, s0, params, maturity):
        """Spot scale times the square root of the expected integrated variance."""

        maturity = jnp.maximum(maturity, 1e-12)

        decay = jnp.exp(-params.kappa * maturity)

        integrated = (
            params.theta * maturity
            + (params.nu0 - params.theta) * (1.0 - decay) / params.kappa
        )

        return s0[0] * jnp.sqrt(jnp.maximum(integrated, 1e-12))


class BasketBlackScholesModel(PricingModel):

    def drift(self, state, params: BasketBlackScholesParams, t):
        return params.r * state

    def diffusion(self, state, params: BasketBlackScholesParams, t):
        return params.sigmas * state

    def basket_value(self, state, params: BasketBlackScholesParams):
        return jnp.sum(params.weights * state)

    def noise_correlation(self, params: BasketBlackScholesParams):
        return params.corr

    def terminal_state(self, s0, params, maturity, num_paths, key, antithetic=True):
        """Correlated lognormal terminal draws, one step, no discretisation."""

        z = _correlated_normals(key, num_paths, len(params.sigmas), params.corr)

        drift = (params.r - 0.5 * params.sigmas**2) * maturity
        scale = params.sigmas * jnp.sqrt(maturity)

        if not antithetic:
            return (s0 * jnp.exp(drift + scale * z),)

        return s0 * jnp.exp(drift + scale * z), s0 * jnp.exp(drift - scale * z)

    def terminal_dispersion(self, s0, params, maturity):
        basket = self.basket_value(s0, params)

        return basket * jnp.mean(params.sigmas) * jnp.sqrt(jnp.maximum(maturity, 1e-12))


class BasketBachelierModel(PricingModel):
    """Correlated driftless normal assets, dS_i = sigma_i dW_i."""

    def drift(self, state, params: BasketBachelierParams, t):
        return jnp.zeros_like(state)

    def diffusion(self, state, params: BasketBachelierParams, t):
        return jnp.broadcast_to(params.sigmas, jnp.shape(state))

    def basket_value(self, state, params: BasketBachelierParams):
        return jnp.sum(params.weights * state)

    def noise_correlation(self, params: BasketBachelierParams):
        return params.corr

    def terminal_state(self, s0, params, maturity, num_paths, key, antithetic=True):
        """S_i(T) = S_i(0) + sigma_i sqrt(T) Z_i with Z correlated; exact."""

        z = _correlated_normals(key, num_paths, len(params.sigmas), params.corr)

        move = params.sigmas * jnp.sqrt(maturity) * z

        if not antithetic:
            return (s0 + move,)

        return s0 + move, s0 - move

    def terminal_dispersion(self, s0, params, maturity):
        from market_simulation.bachelier import basket_normal_volatility

        return basket_normal_volatility(
            params.weights, params.sigmas, params.corr
        ) * jnp.sqrt(jnp.maximum(maturity, 1e-12))


class BasketHestonModel(PricingModel):

    def drift(self, state, params: BasketHestonParams, t):

        n_assets = len(params.weights)

        S = state[:n_assets]

        nu = smooth_positive(state[n_assets:], VARIANCE_SMOOTHING * params.theta)

        return jnp.concatenate([params.r * S, params.kappa * (params.theta - nu)])

    def diffusion(self, state, params: BasketHestonParams, t):

        n_assets = len(params.weights)

        S = state[:n_assets]

        nu = smooth_positive(state[n_assets:], VARIANCE_SMOOTHING * params.theta)

        sqrt_nu = jnp.sqrt(nu)

        return jnp.concatenate([sqrt_nu * S, params.xi * sqrt_nu])

    def basket_value(self, state, params: BasketHestonParams):

        n_assets = len(params.weights)

        S = state[:n_assets]

        return jnp.sum(params.weights * S)

    def noise_correlation(self, params: BasketHestonParams):
        return jnp.block(
            [
                [params.corr, params.rho * params.corr],
                [params.rho * params.corr, params.corr],
            ]
        )

    def terminal_dispersion(self, s0, params, maturity):
        n_assets = len(params.weights)

        maturity = jnp.maximum(maturity, 1e-12)

        decay = jnp.exp(-params.kappa * maturity)

        integrated = (
            params.theta * maturity
            + (params.nu0 - params.theta) * (1.0 - decay) / params.kappa
        )

        basket = jnp.sum(params.weights * s0[:n_assets])

        return basket * jnp.sqrt(jnp.maximum(integrated, 1e-12))
