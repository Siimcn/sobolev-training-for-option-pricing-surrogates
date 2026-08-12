from typing import Dict, List, Optional, Tuple

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.sobolev_trainer import SobolevTrainer

from pipeline.config import ExperimentConfig
from pipeline.model import Surrogate


def train_surrogate(
    surrogate: Surrogate,
    train_dataset: SobolevDataset,
    test_dataset: SobolevDataset,
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
) -> Tuple[SobolevTrainer, Dict[str, List[float]]]:

    trainer = SobolevTrainer(
        surrogate.model,
        config.training,
        checkpoint_path=checkpoint_path,
        grad_scale=surrogate.grad_scale,
        hvp_scale=surrogate.hvp_scale,
    )

    history = trainer.fit(train_dataset=train_dataset, valid_dataset=test_dataset)

    print("\nTraining finished.\n")

    return trainer, history
