from typing import Tuple

from kalibrierung.market_data import MarketData

from marktsimulation.black_scholes_mc import generate_training_paths
from marktsimulation.pricing_model import BlackScholesParams

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.data_generation import create_sobolev_dataset
from surrogate_modeling.dataset import SobolevDataset, train_test_split

from pipeline.config import ExperimentConfig


def build_dataset(
    market_data: MarketData,
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
) -> SobolevDataset:
    # prices/gradients/HVPs come from the MC pricer, not analytic BS
    # (that's only used for calibration and the XVA benchmark)
    return create_sobolev_dataset(
        market_data,
        fitted_params,
        config.data.sobolev_order,
        n_samples=config.data.n_samples,
        pricing_model=config.data.pricing_model,
        basket=config.basket,
    )


def plot_training_path_samples(
    dataset: SobolevDataset,
    config: ExperimentConfig,
) -> None:
    print(
        "\nGenerating training Monte Carlo paths..."
    )

    for i in config.data.preview_sample_indices:

        time_grid_train, train_paths = (
            generate_training_paths(
                dataset.X[i]
            )
        )

        print(
            f"\nSample {i}:"
        )

        print(
            f"S     = {dataset.X[i,0]:.2f}"
        )

        print(
            f"K     = {dataset.X[i,1]:.2f}"
        )

        print(
            f"T     = {dataset.X[i,2]:.2f}"
        )

        print(
            f"sigma = {dataset.X[i,3]:.4f}"
        )

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
