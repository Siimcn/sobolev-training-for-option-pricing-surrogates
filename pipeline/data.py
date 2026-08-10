import jax.numpy as jnp

from typing import Tuple

from kalibrierung.market_data import MarketData

from marktsimulation.basket_mc import (
    generate_basket_training_paths,
    uniform_correlation,
)
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
        min_maturity=config.data.min_maturity,
        pricing_model=config.data.pricing_model,
        basket=config.basket,
    )


def plot_training_path_samples(
    dataset: SobolevDataset,
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
) -> None:
    print(
        "\nGenerating training Monte Carlo paths..."
    )

    paths_for = _path_generator(fitted_params, config)

    for i in config.data.preview_sample_indices:

        time_grid_train, train_paths = paths_for(dataset.X[i])

        print(
            f"\nSample {i}:"
        )

        # the feature layout depends on the pricing model, so label the
        # values from the config rather than assuming [S, K, T, sigma, r]
        for name, value in zip(config.feature_names, dataset.X[i]):
            print(f"{name:6s}= {float(value):.4f}")

        Visualizer.plot_mc_paths(
            time_grid_train,
            train_paths,
            num_paths=config.data.preview_num_paths,
            filename=f"training_paths_{i}.png",
        )


def _path_generator(
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
):
    if not config.is_basket:
        return generate_training_paths

    return lambda x: generate_basket_training_paths(
        x,
        weights=jnp.asarray(config.basket.resolved_weights),
        corr=uniform_correlation(
            config.basket.n_assets, config.basket.correlation
        ),
        sigmas=jnp.full(config.basket.n_assets, fitted_params.sigma),
        r=fitted_params.r,
        num_paths=config.data.preview_num_paths,
        num_steps=config.basket.num_steps,
        seed=config.basket.label_seed,
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
