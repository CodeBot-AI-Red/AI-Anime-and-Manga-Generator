"""Codificador visual inicial, sem pesos treináveis nesta etapa."""

from __future__ import annotations

from .config import ModelConfig
from .tensors import Tensor


class ImageEncoder:
    """Converte uma imagem BCHW em um mapa latente de resolução reduzida.

    A saída ainda é preenchida com zeros: ela define o contrato de dados que
    uma futura rede convolucional/transformer treinável deverá preservar.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def forward(self, images: Tensor) -> Tensor:
        if len(images.shape) != 4:
            raise ValueError("As imagens devem usar a forma (lote, canais, altura, largura).")
        batch, channels, height, width = images.shape
        if channels != self.config.image_channels:
            raise ValueError(f"Esperados {self.config.image_channels} canais, recebidos {channels}.")
        factor = self.config.downsample_factor
        if height % factor or width % factor:
            raise ValueError("Altura e largura devem ser divisíveis pelo fator de redução.")
        return Tensor.zeros((batch, self.config.latent_channels, height // factor, width // factor))
