import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import copy
from dataclasses import dataclass
import math

from typing import Dict, List, Optional


from surrogate_modeling.dataset import SobolevDataset, DataLoader


from surrogate_modeling.losses import sobolev_loss

from surrogate_modeling.metrics import sobolev_metrics

from surrogate_modeling.training_config import (
    COSINE_LR,
    GRADIENT,
    HESSIAN,
    PRICE,
    PRICE_GRADIENT,
    TOTAL,
    TrainingConfig,
)

from surrogate_modeling.surrogate_model import SurrogateModel


@dataclass(frozen=True)
class _EpochStats:
    """One epoch's averaged losses, so `fit` carries five numbers, not five variables."""

    total: float
    price: float
    price_rmse: float
    gradient: float
    hessian: float

    @staticmethod
    def from_metrics(total, metrics) -> "_EpochStats":
        """Build from a single evaluation, e.g. the validation pass."""

        return _EpochStats(
            total=float(total),
            price=float(metrics["price_loss"]),
            price_rmse=float(metrics.get("price_mse_raw", metrics["price_loss"]))
            ** 0.5,
            gradient=float(metrics.get("gradient_loss", 0.0)),
            hessian=float(metrics.get("hessian_loss", 0.0)),
        )


class SobolevTrainer:

    def __init__(
        self,
        model: SurrogateModel,
        config: TrainingConfig,
        checkpoint_path=None,
        grad_scale: Optional[jnp.ndarray] = None,
        hvp_scale: Optional[jnp.ndarray] = None,
    ):
        self.model = model
        self.config = config
        self.checkpoint_path = checkpoint_path

        self.grad_scale = grad_scale
        self.hvp_scale = hvp_scale

        self.config.validate()

        self._build_optimizer(steps_per_epoch=1)

    def _build_optimizer(self, steps_per_epoch: int) -> None:

        self._steps_per_epoch = steps_per_epoch

        self.optimizer = optax.chain(
            *(
                []
                if self.config.gradient_clip is None
                else [optax.clip_by_global_norm(self.config.gradient_clip)]
            ),
            optax.adam(self._learning_rate(steps_per_epoch)),
        )

        self.opt_state = self.optimizer.init(
            eqx.filter(self.model, eqx.is_inexact_array)
        )

    def _current_learning_rate(self, epoch: int) -> float:
        """For the training history; a schedule is a function of the step."""

        rate = self._learning_rate(self._steps_per_epoch)

        if callable(rate):
            return float(rate(epoch * self._steps_per_epoch))

        return float(rate)

    def _learning_rate(self, steps_per_epoch: int):
        """Constant, or warmup then cosine decay towards a small floor."""

        if self.config.lr_schedule != COSINE_LR:
            return self.config.learning_rate

        total = max(self.config.epochs * steps_per_epoch, 2)

        warmup = int(
            min(self.config.warmup_epochs * steps_per_epoch, max(total // 5, 1))
        )

        return optax.warmup_cosine_decay_schedule(
            init_value=self.config.learning_rate * self.config.lr_final_fraction,
            peak_value=self.config.learning_rate,
            warmup_steps=warmup,
            decay_steps=total,
            end_value=self.config.learning_rate * self.config.lr_final_fraction,
        )

    def compute_loss(
        self,
        model: SurrogateModel,
        X: jnp.ndarray,
        y: jnp.ndarray,
        gradients: Optional[jnp.ndarray] = None,
        hvps: Optional[jnp.ndarray] = None,
        V: Optional[jnp.ndarray] = None,
    ):
        """Sobolev loss for one batch."""

        prices_pred = model.predict_prices(X)

        gradients_pred_n = None
        gradients_true_n = None
        hvps_pred_n = None
        hvps_true_n = None

        scale = jax.lax.stop_gradient(model.x_std / model.y_std)

        y_std_sg = jax.lax.stop_gradient(model.y_std)
        price_floor = jnp.maximum(0.05, 0.3 * y_std_sg)
        price_ceiling = jnp.maximum(price_floor * 2.0, 2.0 * y_std_sg)

        if gradients is not None and self.config.sobolev_order >= 1:
            gradients_pred = model.predict_gradients(X)
            gradients_pred_n = gradients_pred * scale
            gradients_true_n = gradients * scale

            if self.grad_scale is not None:
                gradients_pred_n = gradients_pred_n / self.grad_scale
                gradients_true_n = gradients_true_n / self.grad_scale

        if hvps is not None and V is not None and self.config.sobolev_order >= 2:
            hvps_pred = model.predict_hvps(X, V)
            hvps_pred_n = hvps_pred * scale
            hvps_true_n = hvps * scale

            if self.hvp_scale is not None:
                hvps_pred_n = hvps_pred_n / self.hvp_scale
                hvps_true_n = hvps_true_n / self.hvp_scale

        n_dims = X.shape[-1]

        return sobolev_loss(
            prices_pred=prices_pred,
            prices_true=y,
            gradients_pred=gradients_pred_n,
            gradients_true=gradients_true_n,
            hvps_pred=hvps_pred_n,
            hvps_true=hvps_true_n,
            n_dims=n_dims,
            lambda_grad=self.config.lambda_grad,
            lambda_hessian=self.config.lambda_hessian,
            price_scale_floor=price_floor,
            price_scale_ceiling=price_ceiling,
        )

    def _selection_loss(self, valid_loss, valid_metrics) -> float:
        """The quantity early stopping and checkpointing compare."""

        metric = self.config.selection_metric

        if metric == TOTAL:
            return float(valid_loss)

        alpha = float(valid_metrics["alpha"])
        beta = float(valid_metrics["beta"])
        gamma = float(valid_metrics.get("gamma", 0.0))

        terms = {
            PRICE: [(alpha, "price_loss")],
            GRADIENT: [(beta, "gradient_loss")],
            PRICE_GRADIENT: [(alpha, "price_loss"), (beta, "gradient_loss")],
            HESSIAN: [(gamma, "hessian_loss")],
        }[metric]

        weighted = 0.0
        total_weight = 0.0

        for weight, name in terms:
            if name not in valid_metrics:
                continue

            weighted += weight * float(valid_metrics[name])
            total_weight += weight

        if total_weight <= 0.0:
            return float(valid_loss)

        return weighted / total_weight

    def _improvement_threshold(self, best_loss: float) -> float:
        """Relative once a finite best exists, so the first epoch always counts."""

        if self.config.min_delta_relative > 0.0 and math.isfinite(best_loss):
            return self.config.min_delta_relative * abs(best_loss)

        return self.config.min_delta

    @eqx.filter_jit
    def train_step(self, model, opt_state, X, y, gradients, hvps, V):

        (loss_value, metrics), grads = eqx.filter_value_and_grad(
            self.compute_loss, has_aux=True
        )(model, X, y, gradients, hvps, V)

        updates, opt_state = self.optimizer.update(grads, opt_state, model)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state, loss_value, metrics)

    def _run_epoch(self, model, opt_state, dataset, key):
        """
        One pass over the training set.

        Returns the updated model and optimiser state alongside the epoch's
        averaged losses. Sums first and divides once, so the arithmetic is the
        same regardless of how the batches fall.
        """

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        total = price = price_mse_raw = gradient = hessian = 0.0
        n_batches = 0

        for X_batch, y_batch, grad_batch, hvp_batch, V_batch in loader.batches(key):

            model, opt_state, loss_value, metrics = self.train_step(
                model, opt_state, X_batch, y_batch, grad_batch, hvp_batch, V_batch
            )

            total += float(loss_value)
            price += float(metrics["price_loss"])
            price_mse_raw += float(metrics.get("price_mse_raw", metrics["price_loss"]))
            gradient += float(metrics.get("gradient_loss", 0.0))
            hessian += float(metrics.get("hessian_loss", 0.0))

            n_batches += 1

        n = max(n_batches, 1)

        stats = _EpochStats(
            total=total / n,
            price=price / n,
            price_rmse=(price_mse_raw / n) ** 0.5,
            gradient=gradient / n,
            hessian=hessian / n,
        )

        return model, opt_state, stats

    @staticmethod
    def _record(history, prefix, stats):
        """Append one epoch's numbers under `train_` or `valid_`."""

        history[f"{prefix}_loss"].append(stats.total)
        history[f"{prefix}_price_rmse"].append(stats.price_rmse)
        history[f"{prefix}_price_loss"].append(stats.price)
        history[f"{prefix}_gradient_loss"].append(stats.gradient)
        history[f"{prefix}_hessian_loss"].append(stats.hessian)

    def _print_progress(self, epoch, train, valid):
        if valid is None:
            print(f"Epoch {epoch:4d} | Train Loss: {train.total:.6e}")
            return

        print(
            f"Epoch {epoch:4d}"
            f" | Train: {train.total:.6e}"
            f" | Valid: {valid.total:.6e}"
            f" | TrainRMSE: {train.price_rmse:.6e}"
            f" | ValidRMSE: {valid.price_rmse:.6e}"
            f" | GradLoss: {train.gradient:.6e}"
            f" | HessLoss: {train.hessian:.6e}"
        )

    def fit(
        self,
        train_dataset: SobolevDataset,
        valid_dataset: Optional[SobolevDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train, tracking the best model by `config.selection_metric`.

        With a validation set the returned model is the best epoch's, not the
        last one's; without one it is simply the last.
        """

        key = jax.random.PRNGKey(self.config.seed)

        self._build_optimizer(
            steps_per_epoch=max(1, -(-len(train_dataset) // self.config.batch_size))
        )

        history = {
            f"{prefix}_{name}": []
            for prefix in ("train", "valid")
            for name in (
                "loss",
                "price_rmse",
                "price_loss",
                "gradient_loss",
                "hessian_loss",
            )
        }
        history["learning_rate"] = []

        model = self.model
        opt_state = self.opt_state

        best_loss = float("inf")
        patience_counter = 0
        best_model = copy.deepcopy(model)

        for epoch in range(self.config.epochs):

            key, loader_key = jax.random.split(key)

            model, opt_state, train_stats = self._run_epoch(
                model, opt_state, train_dataset, loader_key
            )

            self._record(history, "train", train_stats)
            history["learning_rate"].append(self._current_learning_rate(epoch))

            valid_stats = None

            if valid_dataset is not None:

                valid_loss, valid_metrics = self.compute_loss(
                    model,
                    valid_dataset.X,
                    valid_dataset.y,
                    valid_dataset.gradients,
                    valid_dataset.hvps,
                    valid_dataset.V,
                )

                valid_stats = _EpochStats.from_metrics(valid_loss, valid_metrics)
                self._record(history, "valid", valid_stats)

                selection_loss = self._selection_loss(valid_stats.total, valid_metrics)

                if best_loss - selection_loss > self._improvement_threshold(best_loss):
                    best_loss = selection_loss
                    best_model = copy.deepcopy(model)
                    patience_counter = 0

                    if self.checkpoint_path is not None:
                        eqx.tree_serialise_leaves(self.checkpoint_path, model)

                else:
                    patience_counter += 1

                if (
                    self.config.early_stopping
                    and patience_counter >= self.config.patience
                ):
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % self.config.print_every == 0:
                self._print_progress(epoch, train_stats, valid_stats)

        if valid_dataset is not None:
            model = best_model

        self.model = model
        self.opt_state = opt_state

        return history

    def evaluate(self, dataset: SobolevDataset) -> Dict[str, float]:

        prices_pred = self.model.predict_prices(dataset.X)

        gradients_pred = None
        hvps_pred = None

        if dataset.gradients is not None:
            gradients_pred = self.model.predict_gradients(dataset.X)

        if dataset.hvps is not None and dataset.V is not None:
            hvps_pred = self.model.predict_hvps(dataset.X, dataset.V)

        return sobolev_metrics(
            prices_pred,
            dataset.y,
            gradients_pred,
            dataset.gradients,
            hvps_pred,
            dataset.hvps,
        )
