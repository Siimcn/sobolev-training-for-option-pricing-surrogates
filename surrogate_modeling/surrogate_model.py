import jax
import jax.numpy as jnp
import equinox as eqx

from jax.typing import ArrayLike
from typing import Callable


class SurrogateModel(eqx.Module):
    """
    Normalisierender Wrapper um ein Preis-Netz.

    Ableitungen werden bezüglich der rohen Eingabe x zurückgegeben, da
    JAX die Kettenregel durch die affine Normalisierung automatisch zieht.

    `model` maps one sample of shape (d,) to a scalar. The statistics
    default to the identity, which turns normalization off.
    """

    model: Callable
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    def __init__(
        self,
        model: Callable,
        x_mean: ArrayLike = 0.0,
        x_std: ArrayLike = 1.0,
        y_mean: ArrayLike = 0.0,
        y_std: ArrayLike = 1.0,
    ):
        if not callable(model):
            raise TypeError(
                "model must be a callable JAX model "
                "(e.g. equinox.nn.MLP), got "
                f"{type(model).__name__}."
            )

        self.model = model
        self.x_mean = jnp.asarray(x_mean)
        # floor to avoid division by (near-)zero for a degenerate/constant feature column
        self.x_std = jnp.maximum(jnp.asarray(x_std), 1e-6)
        self.y_mean = jnp.asarray(y_mean)
        self.y_std = jnp.maximum(jnp.asarray(y_std), 1e-6)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # these are fixed dataset stats, not learned params - stop_gradient
        # keeps Adam from slowly drifting them away from their real values
        x_mean = jax.lax.stop_gradient(self.x_mean)
        x_std = jax.lax.stop_gradient(self.x_std)
        y_mean = jax.lax.stop_gradient(self.y_mean)
        y_std = jax.lax.stop_gradient(self.y_std)

        x_norm = (x - x_mean) / x_std
        y_norm = self.model(x_norm)
        return y_norm * y_std + y_mean

    def predict_price(self, x: jnp.ndarray) -> jnp.ndarray:
        """Preis in Originaleinheiten."""
        return self(x).squeeze()

    def predict_gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient des Preises bezüglich der rohen Eingangsgrößen."""
        return jax.grad(self.predict_price)(x)

    def predict_hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """Hessian des Preises bezüglich der rohen Eingangsgrößen."""
        return jax.hessian(self.predict_price)(x)

    def predict_prices(self, X: jnp.ndarray) -> jnp.ndarray:
        """Batch-Auswertung: (N, d) -> (N,)."""
        return jax.vmap(self.predict_price)(X)

    def predict_gradients(self, X: jnp.ndarray) -> jnp.ndarray:
        """Batch-Auswertung: (N, d) -> (N, d)."""
        return jax.vmap(self.predict_gradient)(X)

    def predict_hessians(self, X: jnp.ndarray) -> jnp.ndarray:
        """Batch-Auswertung: (N, d) -> (N, d, d)."""
        return jax.vmap(self.predict_hessian)(X)

    def predict_hvp(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """H @ v, ohne die volle Hessian zu instanziieren."""
        # forward-over-reverse: jvp of the gradient function is exactly H @ v
        _, hvp_val = jax.jvp(jax.grad(self.predict_price), (x,), (v,))
        return hvp_val

    def predict_hvps(self, X: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Batch-Auswertung: (N, d), (N, d) -> (N, d)."""
        return jax.vmap(self.predict_hvp)(X, V)
