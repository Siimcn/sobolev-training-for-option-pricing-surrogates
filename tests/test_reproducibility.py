"""
Bitwise reproducibility guard for refactoring.

Each registered problem is driven through the real pipeline - sample the
domain, price the labels, train a few epochs - and the resulting arrays are
hashed. A refactor that is meant to preserve behaviour must leave every
hash untouched; a changed hash means the numbers moved, whatever the tests
say about behaviour.

The budgets here are deliberately tiny. This is a fingerprint, not an
experiment, and it has to be cheap enough to run after every change.
"""

import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from calibration.market_data import MarketData
from pipeline.config import (
    BasketConfig,
    DataConfig,
    ExperimentConfig,
    HestonConfig,
    NetworkConfig,
    SimulationConfig,
)
from pipeline.model import build_surrogate
from surrogate_modeling.data_generation import create_sobolev_dataset
from surrogate_modeling.dataset import train_test_split
from surrogate_modeling.pricing_problem import build_problem, calibrate_problem
from surrogate_modeling.training_config import TrainingConfig

# a fixed synthetic chain, so the guard never depends on cached market data
_STRIKES = jnp.linspace(80.0, 120.0, 12)
_MATURITIES = jnp.linspace(0.1, 1.5, 12)


def _market() -> MarketData:
    spot = 100.0
    intrinsic = jnp.maximum(spot - _STRIKES, 0.0)
    time_value = 8.0 * jnp.sqrt(_MATURITIES)

    return MarketData(
        spot=spot,
        strikes=_STRIKES,
        maturities=_MATURITIES,
        market_prices=intrinsic + time_value,
        is_call=jnp.ones_like(_STRIKES, dtype=bool),
    )


def _config(model: str) -> ExperimentConfig:
    return ExperimentConfig(
        simulation=SimulationConfig(num_paths=2_000, reference_paths=4_000),
        heston=HestonConfig(num_paths=2_000, num_steps=8, reference_paths=4_000),
        basket=BasketConfig(n_assets=2),
        network=NetworkConfig(width_size=16, depth=2),
        data=DataConfig(pricing_model=model, n_samples=16, min_maturity=0.1),
        prints=False,
    )


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(jnp.asarray(a).astype(jnp.float64).tobytes())
    return h.hexdigest()[:16]


def _fingerprint(model: str) -> dict:
    """Domain, labels and trained weights for one problem, as hashes."""

    config = _config(model)
    market_data = _market()

    calibration = calibrate_problem(model, config=config, market_data=market_data)
    problem = build_problem(
        model, config=config, market_data=market_data, calibration=calibration
    )

    dataset = create_sobolev_dataset(
        problem,
        config.data.sobolev_order,
        n_samples=config.data.n_samples,
        seed=config.data.seed,
        label_seed=config.simulation.label_seed,
    )

    train, _ = train_test_split(dataset, train_fraction=config.data.train_fraction)
    surrogate = build_surrogate(train, config)

    from surrogate_modeling.sobolev_trainer import SobolevTrainer

    trainer = SobolevTrainer(
        model=surrogate.model,
        config=TrainingConfig(
            epochs=3,
            batch_size=8,
            seed=42,
            early_stopping=False,
            sobolev_order=2,
            print_every=10_000,
        ),
        grad_scale=surrogate.grad_scale,
        hvp_scale=surrogate.hvp_scale,
    )
    trainer.fit(train)

    weights = [
        leaf
        for leaf in jax.tree_util.tree_leaves(
            eqx.filter(trainer.model, eqx.is_inexact_array)
        )
    ]

    return {
        "features": _digest(dataset.X),
        "labels": _digest(dataset.y),
        "gradients": _digest(dataset.gradients),
        "hvps": _digest(dataset.hvps),
        "weights": _digest(*weights),
    }


# Recorded before the 2026-08 refactor. Any change claiming to preserve
# behaviour must reproduce these exactly. Regenerate by running this file
# as a script - but only when a numerical change is intended and explained.
EXPECTED = {
    "bachelier": {
        "features": "2569ae87e5ceb54c",
        "labels": "20c96d3b9248d5a2",
        "gradients": "fa6a535d7dc4fa30",
        "hvps": "88f71ab3c42b6ba2",
        "weights": "f4b3aa2a7d121b44",
    },
    "basket_bachelier": {
        "features": "f8526824f58e4705",
        "labels": "60099d69923f78fe",
        "gradients": "cb034b42308bd100",
        "hvps": "4e0fde23aa8bdd87",
        "weights": "98effa49edfcca51",
    },
    "basket_black_scholes": {
        "features": "e7f42ccc3bdd4579",
        "labels": "dc95c39d283ca612",
        "gradients": "9811720f9311a7c7",
        "hvps": "44fecbfe8856e132",
        "weights": "4cb5f3bedbbe32e4",
    },
    "basket_heston": {
        "features": "60da7549570dd907",
        "labels": "f208dcacc3e0dc76",
        "gradients": "4cd7e517a247db28",
        "hvps": "8d84f73144dc2700",
        "weights": "aa1d437d8efc77bb",
    },
    "black_scholes": {
        "features": "e18658c0da758695",
        "labels": "9d204a41be499934",
        "gradients": "46df862521b8c54c",
        # Changed once, when this problem moved off `bs_mc_price` onto the
        # shared `mc_price`. Prices and gradients stayed bitwise identical;
        # the HVPs moved by 9.8e-17 relative - below float64 epsilon - because
        # a single-asset state is shaped (num_paths, 1) rather than
        # (num_paths,), which reduces in a different order. The trained
        # `weights` hash below did not move at all.
        "hvps": "130fca03c861cb56",
        "weights": "2c625cee5f126edc",
    },
    "heston": {
        "features": "4d5217a41007fbfa",
        "labels": "e4775330cc164629",
        "gradients": "6e7f289f0f3a39cf",
        "hvps": "df4e2622e81f8ecd",
        "weights": "2ff15aa526b3d906",
    },
}


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_pipeline_is_bitwise_reproducible(model):
    assert _fingerprint(model) == EXPECTED[model]


if __name__ == "__main__":
    print("EXPECTED = {")
    for name in sorted(EXPECTED):
        print(f'    "{name}": {_fingerprint(name)!r},')
    print("}")
