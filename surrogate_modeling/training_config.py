from dataclasses import dataclass


TOTAL = "total"
PRICE_GRADIENT = "price_gradient"

SELECTION_METRICS = (TOTAL, PRICE_GRADIENT)


@dataclass(frozen=True)
class TrainingConfig:

    learning_rate: float = 1e-4

    # Relative factors for the convex combination in
    # losses.sobolev_loss_weights, not raw multipliers. lambda_hessian
    # defaults lower since HVP labels, a doubly differentiated MC
    # estimator, are the noisiest of the three.

    lambda_grad: float = 1.0
    lambda_hessian: float = 0.1

    epochs: int = 500
    batch_size: int = 32

    early_stopping: bool = True
    patience: int = 50

    # Improvement threshold. min_delta is absolute; when
    # min_delta_relative is set it takes over from the second epoch on and
    # scales with the current best, so the criterion keeps its meaning as
    # the loss shrinks instead of degenerating into a noise detector.
    min_delta: float = 1e-6
    min_delta_relative: float = 0.0

    # Which validation quantity drives early stopping and best-model
    # selection. TOTAL is the Sobolev objective, whose magnitude the HVP
    # term dominates; PRICE_GRADIENT drops that term, so the checkpoint is
    # chosen on the parts the model can actually learn.
    selection_metric: str = "total"

    seed: int = 42

    print_every: int = 10

    save_best_model: bool = True
    checkpoint_path: str = "checkpoints/best_model.eqx"

    sobolev_order: int = 2

    def validate(self) -> None:

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if self.sobolev_order not in (0, 1, 2):
            raise ValueError(
                "sobolev_order must be 0, 1 or 2."
            )

        if self.lambda_grad < 0:
            raise ValueError(
                "lambda_grad must be non-negative."
            )

        if self.lambda_hessian < 0:
            raise ValueError(
                "lambda_hessian must be non-negative."
            )

        if self.min_delta_relative < 0:
            raise ValueError(
                "min_delta_relative must be non-negative."
            )

        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of: "
                f"{', '.join(SELECTION_METRICS)}."
            )