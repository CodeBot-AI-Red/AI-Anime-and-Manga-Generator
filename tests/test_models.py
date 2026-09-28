"""Testes do contrato dimensional do núcleo inicial."""

import unittest

from src.models import AnimeMangaCore, ImageEncoder, ModelConfig, SceneConditioner, SceneDescription, Tensor, TextConditioner


class ModelModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig()

    def test_individual_modules_return_expected_shapes(self) -> None:
        image_latents = ImageEncoder(self.config).forward(Tensor.zeros((2, 3, 8, 10)))
        text_embeddings = TextConditioner(self.config).encode(["heroína anime", "cidade à noite"])
        scene_embeddings = SceneConditioner(self.config).encode(
            [SceneDescription(character_id="aiko"), SceneDescription(art_style="mangá")]
        )

        self.assertEqual(image_latents.shape, (2, 16, 4, 5))
        self.assertEqual(text_embeddings.shape, (2, 12))
        self.assertEqual(scene_embeddings.shape, (2, 12))

    def test_core_initializes_and_preserves_batch_dimensions(self) -> None:
        output = AnimeMangaCore().forward(
            Tensor.zeros((2, 3, 8, 8)),
            prompts=["guerreira anime", "painel de mangá"],
            scenes=[SceneDescription(pose="correndo"), SceneDescription(camera="close-up")],
        )

        self.assertEqual(output["image_latents"].shape, (2, 16, 4, 4))
        self.assertEqual(output["text_embeddings"].shape, (2, 12))
        self.assertEqual(output["scene_embeddings"].shape, (2, 12))

    def test_core_rejects_different_batch_sizes(self) -> None:
        with self.assertRaises(ValueError):
            AnimeMangaCore().forward(
                Tensor.zeros((2, 3, 8, 8)),
                prompts=["somente um prompt"],
                scenes=[SceneDescription(), SceneDescription()],
            )


if __name__ == "__main__":
    unittest.main()
