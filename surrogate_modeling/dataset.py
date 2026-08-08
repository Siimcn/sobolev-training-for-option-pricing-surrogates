import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Optional, Tuple


class SobolevDataset(eqx.Module):

    X: jnp.ndarray
    y: jnp.ndarray

    gradients: Optional[jnp.ndarray] = None
    hvps: Optional[jnp.ndarray] = None
    V: Optional[jnp.ndarray] = None  # HVP probe directions

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]

    def get_batch(
        self,
        indices: jnp.ndarray,
    ):
        X_batch = self.X[indices]
        y_batch = self.y[indices]

        grad_batch = self.gradients[indices] if self.gradients is not None else None
        hvp_batch = self.hvps[indices] if self.hvps is not None else None
        V_batch = self.V[indices] if self.V is not None else None

        return X_batch, y_batch, grad_batch, hvp_batch, V_batch
    

class DataLoader:

    def __init__(
        self,
        dataset: SobolevDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def batches(self, key):

        n = len(self.dataset)

        if self.shuffle:
            indices = jax.random.permutation(key, n)
        else:
            indices = jnp.arange(n)

        for start in range(0, n, self.batch_size):

            end = min(
                start + self.batch_size,
                n,
            )

            batch_indices = indices[start:end]

            yield self.dataset.get_batch(
                batch_indices
            )
    

def train_test_split(
    dataset,
    train_fraction=0.8,
):

    n = len(dataset)

    n_train = int(
        train_fraction * n
    )

    train_dataset = SobolevDataset(
        X=dataset.X[:n_train],
        y=dataset.y[:n_train],
        gradients=(
            dataset.gradients[:n_train]
            if dataset.gradients is not None
            else None
        ),
        hvps=(
            dataset.hvps[:n_train]
            if dataset.hvps is not None
            else None
        ),
        V=(
            dataset.V[:n_train]
            if getattr(dataset, "V", None) is not None
            else None
        ),
    )

    test_dataset = SobolevDataset(
        X=dataset.X[n_train:],
        y=dataset.y[n_train:],
        gradients=(
            dataset.gradients[n_train:]
            if dataset.gradients is not None
            else None
        ),
        hvps=(
            dataset.hvps[n_train:]
            if dataset.hvps is not None
            else None
        ),
        V=(
            dataset.V[n_train:]
            if getattr(dataset, "V", None) is not None
            else None
        ),
    )

    return (
        train_dataset,
        test_dataset,
    )