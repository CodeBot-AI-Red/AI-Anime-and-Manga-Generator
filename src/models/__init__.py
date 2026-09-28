"""Núcleo inicial, leve e extensível, para modelos de anime e mangá."""

from .config import ModelConfig
from .core import AnimeMangaCore
from .image_network import ImageEncoder
from .generative import IterativeImageGenerator, SmallGenerativeModel
from .latent import LatentRepresentation
from .scene import SceneConditioner, SceneDescription
from .tensors import Tensor
from .text_conditioning import TextConditioner

__all__ = [
    "AnimeMangaCore",
    "ImageEncoder",
    "IterativeImageGenerator",
    "LatentRepresentation",
    "ModelConfig",
    "SceneConditioner",
    "SceneDescription",
    "SmallGenerativeModel",
    "Tensor",
    "TextConditioner",
]
