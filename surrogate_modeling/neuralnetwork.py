import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Any, Callable, Optional

from surrogate_modeling.architectures import (
    build_network,
)


class NeuralNetwork(eqx.Module):
    """Generischer Adapter für beliebige JAX-Netze."""

    model: Callable
    architecture: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable,
        architecture: Optional[str] = None,
    ):
        if not callable(model):
            raise TypeError(
                "model must be a callable JAX model "
                "(e.g. equinox.nn.MLP), got "
                f"{type(model).__name__}."
            )

        self.model = model

        self.architecture = (
            architecture
            if architecture is not None
            else type(model).__name__
        )

    @classmethod
    def from_architecture(
        cls,
        key: jax.Array,
        in_size: int,
        out_size: int = 1,
        architecture: str = "MLP",
        **kwargs,
    ) -> "NeuralNetwork":
        """Baut eine registrierte Architektur und verpackt sie."""

        return cls(
            build_network(
                architecture,
                key=key,
                in_size=in_size,
                out_size=out_size,
                **kwargs,
            ),
            architecture=architecture.upper(),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.model(x)

    def predict_price(self, x: jnp.ndarray) -> float:
        return self(x).squeeze()


class FunctionalNetwork(eqx.Module):
    """Adapter für Netze mit getrennt gehaltenen Parametern (Flax, Haiku)."""

    params: Any
    apply_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        params: Any,
        apply_fn: Callable,
    ):
        if not callable(apply_fn):
            raise TypeError(
                "apply_fn must be callable as apply_fn(params, x), got "
                f"{type(apply_fn).__name__}."
            )

        self.params = params
        self.apply_fn = apply_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.apply_fn(self.params, x)
