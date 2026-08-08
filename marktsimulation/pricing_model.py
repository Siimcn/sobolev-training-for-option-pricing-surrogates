import jax.numpy as jnp
import equinox as eqx

from typing import NamedTuple

from marktsimulation.timesteppingscheme import (
    TimeSteppingScheme,
)


class BachelierParams(NamedTuple):
    sigma: float


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


class BasketHestonParams(NamedTuple):
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float

    weights: jnp.ndarray
    corr: jnp.ndarray


class PricingModel(eqx.Module):

    scheme: TimeSteppingScheme

    def __init__(
        self,
        scheme: TimeSteppingScheme,
    ):
        self.scheme = scheme

    def drift(
        self,
        state: jnp.ndarray,
        params,
        t: float,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def diffusion(
        self,
        state: jnp.ndarray,
        params,
        t: float,
    ) -> jnp.ndarray:
        raise NotImplementedError
    
    def noise_correlation(
        self,
        params,
    ):
        return None


class BachelierModel(PricingModel):

    def drift(
        self,
        state,
        params: BachelierParams,
        t,
    ):
        return jnp.zeros_like(state)

    def diffusion(
        self,
        state,
        params: BachelierParams,
        t,
    ):
        return jnp.full_like(
            state,
            params.sigma,
        )


class BlackScholesModel(PricingModel):

    def drift(
        self,
        state,
        params: BlackScholesParams,
        t,
    ):
        return params.r * state

    def diffusion(
        self,
        state,
        params: BlackScholesParams,
        t,
    ):
        return params.sigma * state


class HestonModel(PricingModel):

    def drift(
        self,
        state,
        params: HestonParams,
        t,
    ):

        S = state[0]

        nu = jnp.maximum(
            state[1],
            1e-8,
        )

        return jnp.array([
            params.r * S,
            params.kappa * (
                params.theta - nu
            ),
        ])

    def diffusion(
        self,
        state,
        params: HestonParams,
        t,
    ):

        S = state[0]

        nu = jnp.maximum(
            state[1],
            1e-8,
        )

        sqrt_nu = jnp.sqrt(nu)

        return jnp.array([
            sqrt_nu * S,
            params.xi * sqrt_nu,
        ])

    def noise_correlation(
        self,
        params: HestonParams,
    ):
        # spot and variance driver correlated by rho
        return jnp.array([
            [1.0, params.rho],
            [params.rho, 1.0],
        ])


class BasketBlackScholesModel(PricingModel):

    def drift(
        self,
        state,
        params: BasketBlackScholesParams,
        t,
    ):
        return params.r * state

    def diffusion(
        self,
        state,
        params: BasketBlackScholesParams,
        t,
    ):
        return params.sigmas * state

    def basket_value(
        self,
        state,
        params: BasketBlackScholesParams,
    ):
        return jnp.sum(
            params.weights * state
        )

    def noise_correlation(
        self,
        params: BasketBlackScholesParams,
    ):
        return params.corr


class BasketHestonModel(PricingModel):

    def drift(
        self,
        state,
        params: BasketHestonParams,
        t,
    ):

        n_assets = len(
            params.weights
        )

        S = state[:n_assets]

        nu = jnp.maximum(
            state[n_assets:],
            1e-8,
        )

        return jnp.concatenate([
            params.r * S,
            params.kappa * (
                params.theta - nu
            ),
        ])

    def diffusion(
        self,
        state,
        params: BasketHestonParams,
        t,
    ):

        n_assets = len(
            params.weights
        )

        S = state[:n_assets]

        nu = jnp.maximum(
            state[n_assets:],
            1e-8,
        )

        sqrt_nu = jnp.sqrt(nu)

        return jnp.concatenate([
            sqrt_nu * S,
            params.xi * sqrt_nu,
        ])

    def basket_value(
        self,
        state,
        params: BasketHestonParams,
    ):

        n_assets = len(
            params.weights
        )

        S = state[:n_assets]

        return jnp.sum(
            params.weights * S
        )
    
    def noise_correlation(
        self,
        params: BasketHestonParams,
    ):
        # assets correlated by corr, each
        # spot-vol pair correlated by rho
        return jnp.block([
            [params.corr, params.rho * params.corr],
            [params.rho * params.corr, params.corr],
        ])