"""Train the diffusion denoiser from images and annotations already stored locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DatasetPreprocessor, DatasetValidator, PreprocessingConfig, split_dataset
from src.models.config import ModelConfig
from src.models.diffusion import count_trainable_parameters
from src.training import DiffusionTrainingModel, ProcessedDataLoader, Trainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Diretório de imagens locais.")
    parser.add_argument("--annotations", type=Path, required=True, help="Diretório de JSON/TXT locais.")
    parser.add_argument("--preset", choices=["large", "xlarge", "xxlarge", "lite"], default="large", help="large = grande porte, xlarge = ultra grande porte (512x512), xxlarge = estúdio supremo (1024x1024).")
    parser.add_argument("--resolution", type=int, default=None, help="Padrão: 256 (large), 512 (xlarge), 1024 (xxlarge) ou 32 (lite).")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Passos de acúmulo de gradiente para economizar VRAM em modelos grandes.")
    parser.add_argument("--mixed-precision", action="store_true", help="Ativa precisão mista automática (AMP FP16/BF16) para acelerar e otimizar VRAM.")
    parser.add_argument("--no-ema", action="store_true", help="Desativa o rastreamento EMA dos pesos.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    samples, errors = DatasetValidator().validate(args.images, args.annotations)
    if errors:
        print("Amostras ignoradas:\n" + "\n".join(errors))
    if len(samples) < 2:
        raise SystemExit("São necessárias ao menos duas imagens locais válidas para treino e validação.")
    if args.preset == "xxlarge":
        model_config = ModelConfig.xxlarge()
    elif args.preset == "xlarge":
        model_config = ModelConfig.xlarge()
    elif args.preset == "large":
        model_config = ModelConfig.large()
    else:
        model_config = ModelConfig.lite()
    if args.resolution:
        model_config = ModelConfig.from_dict({**model_config.__dict__, "image_resolution": (args.resolution, args.resolution)})
    resolution = model_config.image_resolution[0]
    split = split_dataset(samples, validation_fraction=0.2)
    processor = DatasetPreprocessor(PreprocessingConfig((resolution, resolution), args.batch_size))
    train_samples = [processor.process(sample) for sample in split.train]
    validation_source = split.validation or split.train[:1]
    validation_samples = [processor.process(sample) for sample in validation_source]
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir, learning_rate=args.learning_rate, device=args.device)
    model = DiffusionTrainingModel(
        model_config,
        learning_rate=config.learning_rate,
        device=config.resolved_device,
        seed=config.seed,
        use_ema=not args.no_ema,
        use_amp=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(f"Parâmetros treináveis do denoiser: {count_trainable_parameters(model.denoiser):,}")
    trainer = Trainer(config, model)
    history = trainer.fit(ProcessedDataLoader(train_samples, args.batch_size, config.seed), ProcessedDataLoader(validation_samples, args.batch_size, config.seed))
    print(f"Treinamento concluído em {trainer.current_epoch} épocas; loss final: {history.train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
