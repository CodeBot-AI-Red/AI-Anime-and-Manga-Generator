"""Executa uma geração experimental local sem modelos ou APIs externas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import IterativeImageGenerator, ModelConfig


def main() -> None:
    generator = IterativeImageGenerator(ModelConfig(image_resolution=(8, 8), generation_steps=3))
    image = generator.generate("guerreira anime em uma cidade ao entardecer")
    print(f"Geração experimental concluída; tensor de imagem: {image.shape}; valores: {len(image.values)}")


if __name__ == "__main__":
    main()
