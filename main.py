import jax

from risk_visualisierung.visualizer import Visualizer

from utils.experiment_logger import ExperimentLogger

from pipeline.config import ExperimentConfig
from pipeline.data import (
    build_dataset,
    plot_training_path_samples,
    split_dataset,
)
from pipeline.evaluation import (
    evaluate_surrogate,
    print_example_prediction,
)
from pipeline.market import load_and_calibrate
from pipeline.model import build_surrogate
from pipeline.reporting import (
    plot_training_diagnostics,
    print_header,
    save_artifacts,
)
from pipeline.risk import run_risk_analysis
from pipeline.training import train_surrogate


def main(config: ExperimentConfig = None):
    """Market data -> calibration -> Sobolev surrogate -> XVA."""

    config = config or ExperimentConfig()

    print_header(config.prints)

    logger = ExperimentLogger(echo=config.prints)
    logger.print_location()
    Visualizer.set_logger(logger)

    market = load_and_calibrate(config)

    if market is None:
        return

    market_data, fitted_params = market

    dataset = build_dataset(market_data, fitted_params, config)
    plot_training_path_samples(dataset, config)

    train_dataset, test_dataset = split_dataset(dataset, config)

    surrogate = build_surrogate(train_dataset, config)

    trainer, history = train_surrogate(
        surrogate,
        train_dataset,
        test_dataset,
        config,
        checkpoint_path=logger.path("best_model.eqx"),
    )

    metrics = evaluate_surrogate(trainer, test_dataset, config)

    plot_training_diagnostics(
        history,
        trainer.model,
        test_dataset,
        market_data,
        fitted_params,
        config,
    )

    print_example_prediction(trainer.model, dataset)

    xva = run_risk_analysis(trainer.model, market_data, fitted_params, config)

    save_artifacts(
        logger,
        config.training,
        market_data,
        fitted_params,
        metrics,
        xva,
    )

    print(
        "\nPipeline finished.\n"
    )


if __name__ == "__main__":

    jax.config.update(
        "jax_enable_x64",
        True,
    )

    main()
