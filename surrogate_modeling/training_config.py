from dataclasses import dataclass


TOTAL = "total"
PRICE = "price"
GRADIENT = "gradient"
PRICE_GRADIENT = "price_gradient"
HESSIAN = "hessian"

SELECTION_METRICS = (TOTAL, PRICE, GRADIENT, PRICE_GRADIENT, HESSIAN)

CONSTANT_LR = "constant"
COSINE_LR = "cosine"

LR_SCHEDULES = (CONSTANT_LR, COSINE_LR)


@dataclass(frozen=True)
class TrainingConfig:

    learning_rate: float = 1e-3

    # Relative factors for the convex combination in
    # losses.sobolev_loss_weights, not raw multipliers. lambda_hessian
    # defaults lower since HVP labels, a doubly differentiated MC
    # estimator, are the noisiest of the three.

    lambda_grad: float = 1.0
    lambda_hessian: float = 0.1

    epochs: int = 1000
    batch_size: int = 32

    # Learning-rate schedule. At a constant rate the Sobolev loss
    # oscillates by a factor of three for hundreds of epochs, so which
    # epoch wins the checkpoint is largely luck. Cosine decay to
    # `lr_final_fraction` of the initial rate removes that.
    lr_schedule: str = COSINE_LR
    lr_final_fraction: float = 0.02
    warmup_epochs: int = 10

    # Global-norm clip on the update. None disables it.
    gradient_clip: float = 1.0

    early_stopping: bool = True
    patience: int = 200

    # Improvement threshold. min_delta is absolute; when
    # min_delta_relative is set it takes over from the second epoch on and
    # scales with the current best, so the criterion keeps its meaning as
    # the loss shrinks instead of degenerating into a noise detector.
    min_delta: float = 1e-6
    min_delta_relative: float = 1e-3

    # Which validation quantity drives early stopping and best-model
    # selection.
    #
    # TOTAL is the Sobolev objective itself and is the right default: it
    # is what the research question asks about. Selecting on
    # PRICE_GRADIENT was measured to stop while the curvature term was
    # still descending, leaving the validation HVP loss 2.6x above what
    # the same run reached later. TOTAL leaves 15%.
    selection_metric: str = TOTAL

    seed: int = 42

    print_every: int = 25

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

        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of: {', '.join(LR_SCHEDULES)}."
            )

        if not 0.0 < self.lr_final_fraction <= 1.0:
            raise ValueError(
                "lr_final_fraction must lie in (0, 1]."
            )

        if self.warmup_epochs < 0:
            raise ValueError(
                "warmup_epochs must be non-negative."
            )

        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError(
                "gradient_clip must be positive, or None to disable it."
            )
