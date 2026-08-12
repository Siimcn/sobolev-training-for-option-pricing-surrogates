import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Any, Callable

DriftFn = Callable[[jnp.ndarray, Any, float], jnp.ndarray]

DiffusionFn = Callable[[jnp.ndarray, Any, float], jnp.ndarray]


class TimeSteppingScheme(eqx.Module):
    """Basisklasse für SDE-Solver."""

    def step(
        self,
        state: jnp.ndarray,
        drift_fn: DriftFn,
        diffusion_fn: DiffusionFn,
        params: Any,
        t: float,
        dW: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def generate_paths(
        self,
        s0: jnp.ndarray,
        drift_fn: DriftFn,
        diffusion_fn: DiffusionFn,
        params: Any,
        key: jax.Array,
        num_paths: int,
        num_steps: int,
        dt: float,
        corr=None,
    ) -> jnp.ndarray:

        state_dim = s0.shape[0]

        dW = jax.random.normal(key, shape=(num_paths, num_steps, state_dim)) * jnp.sqrt(
            dt
        )
        if corr is not None:

            L = jnp.linalg.cholesky(jnp.asarray(corr))

            dW = dW @ L.T

        def simulate_path(dw_path):

            def body(state, data):

                step_idx, dw_t = data

                t = step_idx * dt

                state_next = self.step(
                    state, drift_fn, diffusion_fn, params, t, dw_t, dt
                )

                return state_next, state_next

            indices = jnp.arange(num_steps, dtype=jnp.int32)

            _, path_tail = jax.lax.scan(body, s0, (indices, dw_path))

            return jnp.vstack([s0[None, :], path_tail])

        return jax.vmap(simulate_path)(dW)


class EulerMaruyama(TimeSteppingScheme):

    def step(self, state, drift_fn, diffusion_fn, params, t, dW, dt):

        drift = drift_fn(state, params, t)

        diffusion = diffusion_fn(state, params, t)

        return state + drift * dt + diffusion * dW


class Milstein(TimeSteppingScheme):

    def step(self, state, drift_fn, diffusion_fn, params, t, dW, dt):

        drift = drift_fn(state, params, t)

        diffusion = diffusion_fn(state, params, t)

        jacobian = jax.jacobian(lambda x: diffusion_fn(x, params, t))(state)

        diffusion_derivative = jnp.diag(jacobian)

        correction = 0.5 * diffusion * diffusion_derivative * (dW**2 - dt)

        return state + drift * dt + diffusion * dW + correction
