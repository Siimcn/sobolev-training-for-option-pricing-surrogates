import jax.numpy as jnp

from typing import Dict, List, Optional

from kalibrierung.market_data import MarketData

from marktsimulation.pricing_model import BlackScholesParams

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import TrainingConfig

from pipeline.config import ExperimentConfig


def print_header(enabled: bool = True) -> None:
    """Runs before the logger takes over stdout, so it checks the switch itself."""

    if not enabled:
        return

    print("\n====================================")
    print(" MARKET -> BS -> SURROGATE -> XVA ")
    print("====================================\n")


def plot_training_diagnostics(
    history: Dict[str, List[float]],
    surrogate: SurrogateModel,
    test_dataset: SobolevDataset,
    market_data: MarketData,
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
) -> None:

    Visualizer.plot_training_history(
        history
    )

    Visualizer.plot_price_comparison(
        test_dataset.y,
        surrogate.predict_prices(test_dataset.X),
    )

    print(
        "\nGenerating surrogate surface..."
    )

    _plot_price_surfaces(surrogate, market_data, fitted_params, config)


def _plot_price_surfaces(
    surrogate: SurrogateModel,
    market_data: MarketData,
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
) -> None:

    fixed_input = jnp.array(
        [
            market_data.spot,
            float(jnp.median(market_data.strikes)),
            float(jnp.median(market_data.maturities)),
            fitted_params.sigma,
            fitted_params.r,
        ]
    )

    spot_range = (0.8 * market_data.spot, 1.2 * market_data.spot)

    # spot on the x axis; (y feature index, y range, filename)
    surfaces = [
        (
            1,
            (float(jnp.min(market_data.strikes)), float(jnp.max(market_data.strikes))),
            "surrogate_surface_S_K.png",
        ),
        (
            3,
            (0.8 * fitted_params.sigma, 1.2 * fitted_params.sigma),
            "surrogate_surface_S_sigma.png",
        ),
        (
            2,
            (float(jnp.min(market_data.maturities)), float(jnp.max(market_data.maturities))),
            "surrogate_surface_S_T.png",
        ),
    ]

    for y_idx, y_range, filename in surfaces:
        Visualizer.plot_surrogate_surface(
            surrogate=surrogate,
            fixed_input=fixed_input,
            x_idx=0,
            y_idx=y_idx,
            x_range=spot_range,
            y_range=y_range,
            grid_points=config.surface_grid_points,
            filename=filename,
        )


def save_artifacts(
    logger,
    training_config: TrainingConfig,
    market_data: MarketData,
    fitted_params: BlackScholesParams,
    metrics: Dict[str, float],
    xva: Optional[Dict[str, float]],
) -> None:

    logger.save_report(
        _report_text(fitted_params, metrics, xva)
    )

    logger.save_calibration(
        fitted_params,
        market_data.spot,
    )

    logger.save_metrics(
        {
            **metrics,
            "CalibratedSigma": float(
                fitted_params.sigma
            ),
            "CalibratedRate": float(
                fitted_params.r
            ),
            "Spot": float(
                market_data.spot
            ),
        },
        filename="metrics.json",
    )

    if xva is not None:
        logger.save_xva(
            xva
        )

    logger.save_config(
        training_config
    )


def _report_text(
    fitted_params: BlackScholesParams,
    metrics: Dict[str, float],
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

    return f"""
    Experiment Report
    =================

    Calibrated Sigma : {fitted_params.sigma}
    Calibrated Rate  : {fitted_params.r}

    RMSE     : {metrics['RMSE']}
    MAE      : {metrics['MAE']}
    R2       : {metrics['R2']}
{xva_block}"""
