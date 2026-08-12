import jax

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.pricing_problem import build_problem

from utils.experiment_logger import ExperimentLogger

from pipeline.config import ExperimentConfig
from pipeline.data import build_dataset, plot_training_path_samples, split_dataset
from pipeline.evaluation import (
    evaluate_surrogate,
    print_example_prediction,
    validate_surrogate,
)
from pipeline.market import load_and_calibrate
from pipeline.model import build_surrogate
from pipeline.reporting import plot_training_diagnostics, print_header, save_artifacts
from pipeline.risk import run_risk_analysis
from pipeline.training import train_surrogate


def main(config: ExperimentConfig = None):
    """Problem -> calibration -> dataset -> training -> validation -> risk."""

    config = config or ExperimentConfig()

    print_header(config.prints)

    logger = ExperimentLogger(echo=config.prints)
    logger.print_location()
    Visualizer.set_logger(logger)

    market = load_and_calibrate(config)

    if market is None:
        return

    market_data, calibration = market

    problem = build_problem(
        config.data.pricing_model,
        config=config,
        market_data=market_data,
        calibration=calibration,
    )

    dataset = build_dataset(problem, config)
    plot_training_path_samples(dataset, problem, config)

    train_dataset, test_dataset = split_dataset(dataset, config)

    surrogate = build_surrogate(train_dataset, config)

    trainer, history = train_surrogate(
        surrogate,
        train_dataset,
        test_dataset,
        config,
        checkpoint_path=logger.path("best_model.eqx"),
    )

    metrics = evaluate_surrogate(trainer, test_dataset, problem)

    plot_training_diagnostics(history, trainer.model, test_dataset, problem, config)

    print_example_prediction(trainer.model, dataset)

    validation = validate_surrogate(trainer.model, problem, config)

    xva = run_risk_analysis(trainer.model, problem, config)

    save_artifacts(
        logger, config, problem, market_data, calibration, metrics, validation, xva
    )

    print("\nPipeline finished.\n")


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    main()
