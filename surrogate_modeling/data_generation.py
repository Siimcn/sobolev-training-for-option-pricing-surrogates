import jax
import jax.numpy as jnp

from marktsimulation.sobolev_labels import create_sobolev_labels, label_keys

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.pricing_problem import PricingProblem


def create_sobolev_dataset(
    problem: PricingProblem,
    sobolev_order: int,
    n_samples: int = 600,
    seed: int = 0,
    label_seed: int = 0,
    shared_label_keys: bool = False,
) -> SobolevDataset:
    """Monte-Carlo-labelled Sobolev training set for any `PricingProblem`."""

    key = jax.random.PRNGKey(seed)
    x_key, v_key = jax.random.split(key)

    n_features = problem.n_features

    X = problem.sample_features(
        jax.random.uniform(x_key, shape=(n_samples, n_features))
    )

    V_raw = jax.random.normal(v_key, shape=(n_samples, n_features))
    V = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    keys = label_keys(label_seed, n_samples, shared=shared_label_keys)

    if shared_label_keys:
        print(
            "WARNING: shared_label_keys is on. Every label is priced with "
            "the same random stream, so their Monte Carlo errors do not "
            "average out over the dataset."
        )

    print(f"Generating Monte Carlo {problem.name} dataset with HVPs...")

    prices, gradients, hvps = create_sobolev_labels(
        problem.label_price_fn(),
        X,
        V,
        keys,
        sobolev_order=sobolev_order,
        message=f"Generating data with HVPs for {n_samples} samples...",
    )

    return SobolevDataset(X=X, y=prices, gradients=gradients, hvps=hvps, V=V)
