import jax
import jax.numpy as jnp

from typing import NamedTuple, Optional

from surrogate_modeling.architectures import build_network
from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.surrogate_model import SurrogateModel

from pipeline.config import ExperimentConfig


class Surrogate(NamedTuple):
    """The trainable model plus the Greek scales the trainer needs."""

    model: SurrogateModel
    grad_scale: Optional[jnp.ndarray]
    hvp_scale: Optional[jnp.ndarray]


def build_surrogate(
    train_dataset: SobolevDataset,
    config: ExperimentConfig,
) -> Surrogate:

    # nothing downstream depends on the architecture; SurrogateModel takes
    # any callable JAX model, the registry is only a convenience
    network = build_network(
        architecture=config.network.architecture,
        key=jax.random.PRNGKey(config.network.seed),
        in_size=config.network.in_size or train_dataset.input_dim,
        out_size=config.network.out_size,
        width_size=config.network.width_size,
        depth=config.network.depth,
    )

    # features and the price target span very different scales, which
    # leaves training ill-conditioned without normalization
    x_mean = jnp.mean(train_dataset.X, axis=0)
    x_std = jnp.std(train_dataset.X, axis=0)
    y_mean = jnp.mean(train_dataset.y)
    y_std = jnp.std(train_dataset.y)

    model = SurrogateModel(
        network,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    scale = x_std / y_std

    return Surrogate(
        model=model,
        grad_scale=_derivative_scale(train_dataset.gradients, scale),
        hvp_scale=_derivative_scale(train_dataset.hvps, scale),
    )


def _derivative_scale(values, scale):
    """
    Per-dimension std of a rescaled gradient/HVP, so the pooled MSE is not
    dominated by whichever input dimension has the largest x_std.
    """

    if values is None:
        return None

    return jnp.maximum(
        jnp.std(values * scale, axis=0), 1e-6
    )
