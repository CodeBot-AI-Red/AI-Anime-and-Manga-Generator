"""Testes do contrato dimensional do núcleo inicial."""

import unittest

from src.models import (
    AnimeMangaCore,
    ImageEncoder,
    IterativeImageGenerator,
    LatentRepresentation,
    ModelConfig,
    SceneConditioner,
    SceneDescription,
    SmallGenerativeModel,
    Tensor,
    TextConditioner,
)


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

    def test_latent_representation_preserves_expected_dimensions(self) -> None:
        representation = LatentRepresentation(self.config)
        latent = representation.encode(Tensor.zeros((2, 3, 8, 8)))
        decoded = representation.decode(latent)
        self.assertEqual(latent.shape, (2, 16, 4, 4))
        self.assertEqual(decoded.shape, (2, 3, 8, 8))

    def test_generative_model_forward_uses_text_and_step(self) -> None:
        latents = Tensor.zeros((2, 16, 4, 4))
        text = TextConditioner(self.config).encode(["heroína", "robô"])
        output = SmallGenerativeModel(self.config).forward(latents, text, generation_step=1)
        self.assertEqual(output.shape, latents.shape)
        self.assertNotEqual(output.values, latents.values)

    def test_iterative_generation_produces_an_image_from_prompt(self) -> None:
        generator = IterativeImageGenerator(ModelConfig(image_resolution=(8, 6), generation_steps=3))
        latents = generator.generate_latents(["garota anime sorrindo"])
        image = generator.generate("garota anime sorrindo")
        self.assertEqual(latents.shape, (1, 16, 3, 4))
        self.assertEqual(image.shape, (1, 3, 6, 8))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in image.values))


if __name__ == "__main__":
    unittest.main()
