"""Ponto de extensão para condicionamento de texto futuro."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ModelConfig
from .tensors import Tensor


class TextConditioner:
    """Produz embeddings determinísticos simples a partir de prompts.

    Isto não é um modelo de linguagem. O vetor serve somente como interface
    estável até a integração de um tokenizer e de um encoder treinável.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.embedding_dim = config.text_embedding_dim

    def encode(self, prompts: Sequence[str]) -> Tensor:
        if not prompts:
            raise ValueError("Forneça ao menos um prompt de texto.")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Cada prompt deve ser uma string.")
        values = tuple(
            float((sum(ord(character) for character in prompt) + index) % 257) / 256
            for prompt in prompts
            for index in range(self.embedding_dim)
        )
        return Tensor(shape=(len(prompts), self.embedding_dim), values=values)
