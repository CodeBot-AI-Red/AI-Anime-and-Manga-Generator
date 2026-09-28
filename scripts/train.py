"""Executa uma demonstração local do pipeline com dados sintéticos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import ProcessedDataLoader, Trainer, TrainingConfig
from src.training.mock import synthetic_samples


def main() -> None:
    config = TrainingConfig()
    samples = synthetic_samples()
    trainer = Trainer(config)
    history = trainer.fit(ProcessedDataLoader(samples, config.batch_size, config.seed), ProcessedDataLoader(samples, config.batch_size, config.seed))
    print(f"Treinamento local concluído em {trainer.current_epoch} épocas; loss final: {history.train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
