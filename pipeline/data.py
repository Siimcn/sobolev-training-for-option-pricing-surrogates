from typing import Tuple

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.data_generation import create_sobolev_dataset
from surrogate_modeling.dataset import SobolevDataset, train_test_split
from surrogate_modeling.pricing_problem import PricingProblem

from pipeline.config import ExperimentConfig


def build_dataset(
    problem: PricingProblem,
    config: ExperimentConfig,
) -> SobolevDataset:
    # prices/gradients/HVPs come from the problem's MC pricer, not a closed
    # form (that's only used for calibration and validation)
    return create_sobolev_dataset(
        problem,
        config.data.sobolev_order,
        n_samples=config.data.n_samples,
        seed=config.data.seed,
        label_seed=config.simulation.label_seed,
        shared_label_keys=config.simulation.shared_label_keys,
    )


def plot_training_path_samples(
    dataset: SobolevDataset,
    problem: PricingProblem,
    config: ExperimentConfig,
) -> None:

    indices = [
        i for i in config.data.preview_sample_indices if i < len(dataset)
    ]

    if not indices:
        return

    if problem.underlying_paths(dataset.X[indices[0]], num_paths=2) is None:
        print(
            f"\nSkipping training-path preview: '{problem.name}' does not "
            f"expose paths of its underlying."
        )
        return

    print(
        "\nGenerating training Monte Carlo paths..."
    )

    for i in indices:

        time_grid_train, train_paths = problem.underlying_paths(
            dataset.X[i],
            num_paths=config.data.preview_num_paths,
        )

        print(
            f"\nSample {i}:"
        )

        # the feature layout is the problem's, so label the values from it
        # rather than assuming [S, K, T, sigma, r]
        for name, value in zip(problem.feature_names, dataset.X[i]):
            print(f"{name:6s}= {float(value):.4f}")

        Visualizer.plot_mc_paths(
            time_grid_train,
            train_paths,
            num_paths=config.data.preview_num_paths,
            filename=f"training_paths_{i}.png",
        )


def split_dataset(
    dataset: SobolevDataset,
    config: ExperimentConfig,
) -> Tuple[SobolevDataset, SobolevDataset]:
    # split before computing normalization stats, to avoid test-set leakage
    train_dataset, test_dataset = (
        train_test_split(
            dataset,
            train_fraction=config.data.train_fraction,
        )
    )

    print(
        f"\nTrain size: {len(train_dataset)}"
    )

    print(
        f"Test size: {len(test_dataset)}"
    )

    return train_dataset, test_dataset
