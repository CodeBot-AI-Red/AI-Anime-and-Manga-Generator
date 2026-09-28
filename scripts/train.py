"""Train the diffusion denoiser from images and annotations already stored locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DatasetPreprocessor, DatasetValidator, PreprocessingConfig, split_dataset
from src.models.config import ModelConfig
from src.training import DiffusionTrainingModel, ProcessedDataLoader, Trainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Diretório de imagens locais.")
    parser.add_argument("--annotations", type=Path, required=True, help="Diretório de JSON/TXT locais.")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    samples, errors = DatasetValidator().validate(args.images, args.annotations)
    if errors:
        print("Amostras ignoradas:\n" + "\n".join(errors))
    if len(samples) < 2:
        raise SystemExit("São necessárias ao menos duas imagens locais válidas para treino e validação.")
    split = split_dataset(samples, validation_fraction=0.2)
    processor = DatasetPreprocessor(PreprocessingConfig((args.resolution, args.resolution), args.batch_size))
    train_samples = [processor.process(sample) for sample in split.train]
    validation_source = split.validation or split.train[:1]
    validation_samples = [processor.process(sample) for sample in validation_source]
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir, learning_rate=0.001, device="cpu")
    model = DiffusionTrainingModel(ModelConfig(image_resolution=(args.resolution, args.resolution)), config.learning_rate, config.resolved_device, config.seed)
    trainer = Trainer(config, model)
    history = trainer.fit(ProcessedDataLoader(train_samples, args.batch_size, config.seed), ProcessedDataLoader(validation_samples, args.batch_size, config.seed))
    print(f"Treinamento concluído em {trainer.current_epoch} épocas; loss final: {history.train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
