from typing import Dict, List, Optional

from calibration.market_data import MarketData

from risk_visualization.visualizer import Visualizer

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.pricing_problem import CalibrationResult, PricingProblem
from surrogate_modeling.surrogate_model import SurrogateModel

from pipeline.config import ExperimentConfig


def print_header(enabled: bool = True) -> None:
    """Runs before the logger takes over stdout, so it checks the switch itself."""

    if not enabled:
        return

    print("\n====================================")
    print(" MARKET -> CALIBRATION -> SURROGATE ")
    print("====================================\n")


def plot_training_diagnostics(
    history: Dict[str, List[float]],
    surrogate: SurrogateModel,
    test_dataset: SobolevDataset,
    problem: PricingProblem,
    config: ExperimentConfig,
) -> None:

    Visualizer.plot_training_history(history)

    Visualizer.plot_price_comparison(
        test_dataset.y, surrogate.predict_prices(test_dataset.X)
    )

    print("\nGenerating surrogate surface...")

    _plot_price_surfaces(surrogate, problem, config)


def _plot_price_surfaces(
    surrogate: SurrogateModel, problem: PricingProblem, config: ExperimentConfig
) -> None:
    """Slices through the surrogate's input space, all of it from `problem`."""

    baseline = problem.baseline_features()

    names = problem.feature_names
    labels = problem.feature_labels

    for spec in problem.surface_specs():

        filename = (
            f"surrogate_surface_" f"{names[spec.x_index]}_{names[spec.y_index]}.png"
        )

        Visualizer.plot_surrogate_surface(
            surrogate=surrogate,
            fixed_input=baseline,
            x_idx=spec.x_index,
            y_idx=spec.y_index,
            x_range=spec.x_range,
            y_range=spec.y_range,
            feature_labels=labels,
            grid_points=config.surface_grid_points,
            filename=filename,
        )


def save_artifacts(
    logger,
    config: ExperimentConfig,
    problem: PricingProblem,
    market_data: MarketData,
    calibration: CalibrationResult,
    metrics: Dict[str, float],
    validation: Optional[Dict[str, float]],
    xva: Optional[Dict[str, float]],
) -> None:

    logger.save_report(_report_text(problem, calibration, metrics, validation, xva))

    logger.save_calibration(calibration.params, market_data.spot)

    logger.save_metrics(
        {
            **metrics,
            **(validation or {}),
            **_fitted_metrics(calibration),
            "Spot": float(market_data.spot),
        },
        filename="metrics.json",
    )

    if xva is not None:
        logger.save_xva(xva)

    logger.save_config(config.to_dict(problem))


def _fitted_fields(calibration) -> Dict[str, object]:
    """The fitted parameters by their own names, whatever the model calls them."""

    params = calibration.params

    if hasattr(params, "_asdict"):
        return dict(params._asdict())

    return {"params": params}


def _fitted_metrics(calibration) -> Dict[str, float]:
    """Prefixed so they cannot collide with a metric name."""

    out = {}

    for name, value in _fitted_fields(calibration).items():
        try:
            out[f"Calibrated_{name}"] = float(value)
        except (TypeError, ValueError):
            continue

    return out


def _report_text(
    problem: PricingProblem,
    calibration: CalibrationResult,
    metrics: Dict[str, float],
    validation: Optional[Dict[str, float]],
    xva: Optional[Dict[str, float]],
) -> str:

    xva_block = (
        f"""
    CVA      : {xva['CVA']}
    DVA      : {xva['DVA']}
    Net XVA  : {xva['NetXVA']}
"""
        if xva is not None
        else """
    XVA      : not run for this pricing model
"""
    )

    validation_block = (
        "".join(f"    {k:<34s}: {v}\n" for k, v in validation.items())
        if validation
        else "    not run\n"
    )

    fitted_block = "".join(
        f"    {k:<34s}: {v}\n" for k, v in _fitted_fields(calibration).items()
    )

    assumption_block = (
        "".join(f"    {k:<34s}: {v}\n" for k, v in calibration.assumptions.items())
        if calibration.assumptions
        else "    none - every parameter was fitted\n"
    )

    return f"""
    Experiment Report
    =================

    Pricing model    : {problem.name}
    Features         : {', '.join(problem.feature_names)}

    Converged        : {calibration.converged}

    Fitted parameters
    -----------------
{fitted_block}

    Assumed, not calibrated
    -----------------------
{assumption_block}

    RMSE     : {metrics['RMSE']}
    MAE      : {metrics['MAE']}
    R2       : {metrics['R2']}

    Independent validation
    ----------------------
{validation_block}{xva_block}"""
