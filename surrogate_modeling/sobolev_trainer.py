import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import copy
import math

from typing import Dict, List, Optional


from surrogate_modeling.dataset import (
    SobolevDataset,
    DataLoader,
)


from surrogate_modeling.losses import (
    sobolev_loss,
)

from surrogate_modeling.metrics import (
    sobolev_metrics,
)

from surrogate_modeling.training_config import (
    COSINE_LR,
    GRADIENT,
    HESSIAN,
    PRICE,
    PRICE_GRADIENT,
    TOTAL,
    TrainingConfig,
)

from surrogate_modeling.surrogate_model import (
    SurrogateModel,
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

        # per-dimension std of each chain-rule-rescaled Greek, computed once
        # from the training set - keeps one dimension from dominating the
        # pooled gradient/HVP MSE just because its raw x_std is larger
        self.grad_scale = grad_scale
        self.hvp_scale = hvp_scale

        self.config.validate()

        # one step per batch, so the schedule needs the batch count; it is
        # only known once fit() sees the training set
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

        # is_inexact_array, not is_array: filter_value_and_grad differentiates
        # only inexact leaves, so a bool mask would desync opt_state and grads
        self.opt_state = self.optimizer.init(
            eqx.filter(
                self.model,
                eqx.is_inexact_array,
            )
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

        # warmup has to leave room for the decay: optax measures the cosine
        # over (decay_steps - warmup_steps), which must stay positive even
        # for a very short run
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
        """
        Sobolev loss for one batch.

        Gradients/HVPs are rescaled by x_std/y_std (chain rule for the
        model's normalization) so the three residuals share a common
        scale, then optionally divided by `grad_scale`/`hvp_scale` so no
        single input dimension dominates the pooled MSE.
        """

        prices_pred = model.predict_prices(X)

        gradients_pred_n = None
        gradients_true_n = None
        hvps_pred_n = None
        hvps_true_n = None

        # d(y_norm)/d(x_norm) = (dy/dx) * (x_std / y_std).
        # stop_gradient again here (a separate read of x_std/y_std),
        # otherwise Adam would slowly update these fixed stats
        scale = jax.lax.stop_gradient(model.x_std / model.y_std)

        # price loss floor/ceiling, tied to the global y_std rather than this
        # batch's own std (a 32-sample batch would be too noisy)
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
        """
        The quantity early stopping and checkpointing compare.

        TOTAL is the Sobolev objective itself. The subset criteria drop
        terms and renormalize the remaining weights, which keeps their
        balance but means the dropped term is no longer being optimised
        for at selection time - measured to leave the validation HVP loss
        2.6x above its achievable value under PRICE_GRADIENT.
        """

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
    def train_step(
        self,
        model,
        opt_state,
        X,
        y,
        gradients,
        hvps,
        V,
    ):

        (loss_value, metrics), grads = (
            eqx.filter_value_and_grad(
                self.compute_loss,
                has_aux=True,
            )(
                model,
                X,
                y,
                gradients,
                hvps,
                V,
            )
        )

        updates, opt_state = self.optimizer.update(
            grads,
            opt_state,
            model,
        )

        model = eqx.apply_updates(
            model,
            updates,
        )

        return (
            model,
            opt_state,
            loss_value,
            metrics,
        )

    def fit(
        self,
        train_dataset: SobolevDataset,
        valid_dataset: Optional[SobolevDataset] = None,
    ) -> Dict[str, List[float]]:

        key = jax.random.PRNGKey(
            self.config.seed
        )

        # the schedule is defined in optimizer steps, which is batches, so
        # it can only be built once the training set size is known
        self._build_optimizer(
            steps_per_epoch=max(
                1,
                -(-len(train_dataset) // self.config.batch_size),
            )
        )

        history = {
            "train_loss": [],
            "valid_loss": [],
            "learning_rate": [],

            "train_price_rmse": [],
            "valid_price_rmse": [],

            "train_price_loss": [],
            "train_gradient_loss": [],
            "train_hessian_loss": [],

            "valid_price_loss": [],
            "valid_gradient_loss": [],
            "valid_hessian_loss": [],
        }

        model = self.model
        opt_state = self.opt_state

        best_loss = float("inf")
        patience_counter = 0

        best_model = copy.deepcopy(
            model
        )

        for epoch in range(
            self.config.epochs
        ):

            key, loader_key = jax.random.split(
                key
            )

            loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            epoch_loss = 0.0
            epoch_price_loss = 0.0
            epoch_price_mse_raw = 0.0
            epoch_gradient_loss = 0.0
            epoch_hessian_loss = 0.0
            n_batches = 0

            for (
                X_batch,
                y_batch,
                grad_batch,
                hvp_batch,
                V_batch,
            ) in loader.batches(loader_key):

                (
                    model,
                    opt_state,
                    loss_value,
                    metrics,
                ) = self.train_step(
                    model,
                    opt_state,
                    X_batch,
                    y_batch,
                    grad_batch,
                    hvp_batch,
                    V_batch,
                )

                epoch_loss += float(loss_value)

                epoch_price_loss += float(
                        metrics["price_loss"]
                    )

                # raw dollar MSE, for interpretable RMSE in plots/console
                epoch_price_mse_raw += float(
                    metrics.get("price_mse_raw", metrics["price_loss"])
                )

                epoch_gradient_loss += float(
                    metrics.get(
                        "gradient_loss",
                        0.0,
                    )
                )

                epoch_hessian_loss += float(
                    metrics.get(
                        "hessian_loss",
                        0.0,
                    )
                )

                n_batches += 1

            epoch_loss /= max(
                n_batches,
                1,
            )

            epoch_price_loss /= max(
                n_batches,
                1,
            )

            epoch_price_mse_raw /= max(
                n_batches,
                1,
            )

            epoch_gradient_loss /= max(
                n_batches,
                1,
            )

            epoch_hessian_loss /= max(
                n_batches,
                1,
            )


            epoch_price_rmse = (
                epoch_price_mse_raw
            ) ** 0.5


            history["train_loss"].append(
                epoch_loss
            )

            history["learning_rate"].append(
                self._current_learning_rate(epoch)
            )

            history["train_price_rmse"].append(
                epoch_price_rmse
            )

            history["train_price_loss"].append(
                epoch_price_loss
            )

            history["train_gradient_loss"].append(
                epoch_gradient_loss
            )

            history["train_hessian_loss"].append(
                epoch_hessian_loss
            )

            if valid_dataset is not None:

                valid_loss, valid_metrics = self.compute_loss(
                    model,
                    valid_dataset.X,
                    valid_dataset.y,
                    valid_dataset.gradients,
                    valid_dataset.hvps,
                    valid_dataset.V,
                )


                valid_price_rmse = (
                    float(
                        valid_metrics.get(
                            "price_mse_raw",
                            valid_metrics["price_loss"],
                        )
                    )
                ) ** 0.5


                history["valid_price_rmse"].append(
                    valid_price_rmse
                )

                valid_loss = float(valid_loss)

                history["valid_loss"].append(
                    valid_loss
                )

                history["valid_price_loss"].append(
                    float(valid_metrics["price_loss"])
                )

                history["valid_gradient_loss"].append(
                    float(valid_metrics.get("gradient_loss", 0.0))
                )

                history["valid_hessian_loss"].append(
                    float(valid_metrics.get("hessian_loss", 0.0))
                )

                selection_loss = self._selection_loss(
                    valid_loss, valid_metrics
                )

                # best-checkpoint tracking always runs, even if early_stopping
                # is off, since `model = best_model` runs unconditionally below
                improvement = (
                    best_loss - selection_loss
                )

                if (
                    improvement
                    > self._improvement_threshold(best_loss)
                ):

                    best_loss = selection_loss

                    best_model = copy.deepcopy(
                        model
                    )

                    if self.checkpoint_path is not None:

                        eqx.tree_serialise_leaves(
                            self.checkpoint_path,
                            model,
                        )

                    patience_counter = 0

                else:
                    patience_counter += 1

                if (
                    self.config.early_stopping
                    and patience_counter
                    >= self.config.patience
                ):
                    print(
                        f"Early stopping at "
                        f"epoch {epoch}"
                    )
                    break

            if (
                epoch
                % self.config.print_every
                == 0
            ):

                if valid_dataset is None:

                    print(
                        f"Epoch {epoch:4d}"
                        f" | Train Loss:"
                        f" {epoch_loss:.6e}"
                    )

                else:

                    print(
                        f"Epoch {epoch:4d}"
                        f" | Train: {epoch_loss:.6e}"
                        f" | Valid: {valid_loss:.6e}"
                        f" | TrainRMSE: {epoch_price_rmse:.6e}"
                        f" | ValidRMSE: {valid_price_rmse:.6e}"
                        f" | GradLoss: {epoch_gradient_loss:.6e}"
                        f" | HessLoss: {epoch_hessian_loss:.6e}"
                    )

        if valid_dataset is not None:

            model = best_model

        self.model = model
        self.opt_state = opt_state

        return history

    def evaluate(
        self,
        dataset: SobolevDataset,
    ) -> Dict[str, float]:

        prices_pred = self.model.predict_prices(
            dataset.X
        )

        gradients_pred = None
        hvps_pred = None

        if dataset.gradients is not None:
            gradients_pred = (
                self.model.predict_gradients(
                    dataset.X
                )
            )

        if dataset.hvps is not None and dataset.V is not None:
            hvps_pred = (
                self.model.predict_hvps(
                    dataset.X,
                    dataset.V
                )
            )

        return sobolev_metrics(
            prices_pred,
            dataset.y,
            gradients_pred,
            dataset.gradients,
            hvps_pred,
            dataset.hvps,
        )
