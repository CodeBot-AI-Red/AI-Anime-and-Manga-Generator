"""Núcleo inicial, leve e extensível, para modelos de anime e mangá."""

from .config import ModelConfig
from .core import AnimeMangaCore
from .image_network import ImageEncoder
from .scene import SceneConditioner, SceneDescription
from .tensors import Tensor
from .text_conditioning import TextConditioner

__all__ = [
    "AnimeMangaCore",
    "ImageEncoder",
    "ModelConfig",
    "SceneConditioner",
    "SceneDescription",
    "Tensor",
    "TextConditioner",
]
