import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Callable, Dict, Tuple


NetworkBuilder = Callable[..., Callable]


class ResidualMLP(eqx.Module):
    """MLP mit Skip-Connections."""

    input_layer: eqx.nn.Linear
    blocks: Tuple[Tuple[eqx.nn.Linear, eqx.nn.Linear], ...]
    output_layer: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int = 1,
        width_size: int = 64,
        depth: int = 4,
        activation: Callable = jax.nn.softplus,
        *,
        key: jax.Array,
    ):
        n_blocks = max(depth // 2, 1)

        keys = jax.random.split(
            key,
            2 * n_blocks + 2,
        )

        self.input_layer = eqx.nn.Linear(
            in_size,
            width_size,
            key=keys[0],
        )

        self.blocks = tuple(
            (
                eqx.nn.Linear(
                    width_size,
                    width_size,
                    key=keys[2 * i + 1],
                ),
                eqx.nn.Linear(
                    width_size,
                    width_size,
                    key=keys[2 * i + 2],
                ),
            )
            for i in range(n_blocks)
        )

        self.output_layer = eqx.nn.Linear(
            width_size,
            out_size,
            key=keys[-1],
        )

        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        h = self.input_layer(x)

        for first, second in self.blocks:
            h = h + second(
                self.activation(
                    first(self.activation(h))
                )
            )

        return self.output_layer(
            self.activation(h)
        )


def build_mlp(
    key: jax.Array,
    in_size: int,
    out_size: int = 1,
    width_size: int = 64,
    depth: int = 4,
    activation: Callable = jax.nn.softplus,
    **kwargs,
) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation=activation,
        key=key,
        **kwargs,
    )


def build_residual_mlp(
    key: jax.Array,
    in_size: int,
    out_size: int = 1,
    width_size: int = 64,
    depth: int = 4,
    activation: Callable = jax.nn.softplus,
    **kwargs,
) -> ResidualMLP:
    return ResidualMLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation=activation,
        key=key,
        **kwargs,
    )


_ARCHITECTURES: Dict[str, NetworkBuilder] = {}


def register_architecture(
    name: str,
    builder: NetworkBuilder,
    overwrite: bool = False,
) -> None:
    """Registriert eine Netzarchitektur unter `name` (case-insensitive)."""

    if not callable(builder):
        raise TypeError(
            f"Builder for architecture '{name}' must be callable."
        )

    key = name.upper()

    if key in _ARCHITECTURES and not overwrite:
        raise ValueError(
            f"Architecture '{key}' is already registered. "
            "Pass overwrite=True to replace it."
        )

    _ARCHITECTURES[key] = builder


def available_architectures() -> Tuple[str, ...]:
    return tuple(sorted(_ARCHITECTURES))


def build_network(
    architecture: str,
    key: jax.Array,
    in_size: int,
    out_size: int = 1,
    **kwargs,
) -> Callable:
    """Erzeugt ein Netz der registrierten Architektur `architecture`."""

    name = architecture.upper()

    if name not in _ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Registered: {', '.join(available_architectures())}. "
            "Add your own with register_architecture(), or hand an "
            "already-built JAX model straight to SurrogateModel."
        )

    model = _ARCHITECTURES[name](
        key=key,
        in_size=in_size,
        out_size=out_size,
        **kwargs,
    )

    if not callable(model):
        raise TypeError(
            f"Builder for architecture '{name}' returned a "
            "non-callable object."
        )

    return model


register_architecture("MLP", build_mlp)
register_architecture("RESMLP", build_residual_mlp)
