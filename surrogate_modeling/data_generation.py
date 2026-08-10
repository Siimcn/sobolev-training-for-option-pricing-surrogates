import jax
import jax.numpy as jnp

from marktsimulation.sobolev_labels import create_sobolev_labels

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.pricing_problem import PricingProblem


def create_sobolev_dataset(
    problem: PricingProblem,
    sobolev_order: int,
    n_samples: int = 600,
    seed: int = 0,
) -> SobolevDataset:
    """
    Monte-Carlo-labelled Sobolev training set for any `PricingProblem`.

    Prices, gradients and HVPs all come from differentiating through the
    problem's Monte Carlo pricer; a closed form, where one exists, is used
    only for calibration and validation, never for training labels.

    The domain, the feature layout and the pricer all come from `problem`,
    so this function contains nothing specific to any one model.
    """

    key = jax.random.PRNGKey(seed)
    x_key, v_key = jax.random.split(key)

    n_features = problem.n_features

    X = problem.sample_features(
        jax.random.uniform(x_key, shape=(n_samples, n_features))
    )

    # one random unit probe direction per sample (Hutchinson-style HVP
    # probing), rather than one fixed direction reused for every sample
    V_raw = jax.random.normal(v_key, shape=(n_samples, n_features))
    V = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    print(f"Generating Monte Carlo {problem.name} dataset with HVPs...")

    prices, gradients, hvps = create_sobolev_labels(
        problem.label_price_fn(),
        X,
        V,
        sobolev_order=sobolev_order,
        message=f"Generating data with HVPs for {n_samples} samples...",
    )

    return SobolevDataset(
        X=X,
        y=prices,
        gradients=gradients,
        hvps=hvps,
        V=V,
    )
