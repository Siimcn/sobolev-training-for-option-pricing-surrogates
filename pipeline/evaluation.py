from typing import Dict, Optional, Sequence

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.metrics import per_dimension_relative_error
from surrogate_modeling.pricing_problem import PricingProblem
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.validation import run_reference_validation

from pipeline.config import ExperimentConfig


def evaluate_surrogate(
    trainer: SobolevTrainer,
    test_dataset: SobolevDataset,
    problem: PricingProblem,
) -> Dict[str, float]:

    metrics = trainer.evaluate(
        test_dataset
    )

    Visualizer.report_metrics(
        metrics
    )

    report_per_dimension_greeks(
        trainer.model, test_dataset, problem.feature_names
    )

    return metrics


def validate_surrogate(
    surrogate: SurrogateModel,
    problem: PricingProblem,
    config: ExperimentConfig,
) -> Optional[Dict[str, float]]:
    """
    Independent benchmark, run for every pricing model alike.

    Test-set metrics only say the surrogate reproduces its own labels; this
    re-prices it against Monte Carlo with an unrelated seed and checks the
    model-free properties the problem declares.
    """

    if not config.validation.enabled:
        return None

    return run_reference_validation(
        surrogate,
        problem,
        n_points=config.validation.n_points,
        seed=config.validation.seed,
    )


def report_per_dimension_greeks(
    surrogate: SurrogateModel,
    test_dataset: SobolevDataset,
    feature_names: Sequence[str],
) -> None:
    """A pooled Relative_L2 hides which Greek drives the error."""
    grad_rel_err = None
    if test_dataset.gradients is not None:
        grad_rel_err = per_dimension_relative_error(
            surrogate.predict_gradients(test_dataset.X), test_dataset.gradients
        )

    hvp_rel_err = None
    if test_dataset.hvps is not None:
        hvp_rel_err = per_dimension_relative_error(
            surrogate.predict_hvps(test_dataset.X, test_dataset.V), test_dataset.hvps
        )

    print("\n===== Per-dimension Greek accuracy (test set, Relative L2) =====")
    for i, name in enumerate(feature_names):
        ge_str = f"{float(grad_rel_err[i]):.4f}" if grad_rel_err is not None else "n/a"
        he_str = f"{float(hvp_rel_err[i]):.4f}" if hvp_rel_err is not None else "n/a"
        print(f"{name:6s} | Gradient: {ge_str} | HVP: {he_str}")


def print_example_prediction(
    surrogate: SurrogateModel,
    dataset: SobolevDataset,
) -> None:
    x = dataset.X[0]

    print("\n===== Example =====")

    print(
        "\nPrice:"
    )

    print(
        surrogate.predict_price(x)
    )

    print(
        "\nGradient:"
    )

    print(
        surrogate.predict_gradient(x)
    )

    print(
        "\nHessian:"
    )

    print(
        surrogate.predict_hessian(x)
    )
