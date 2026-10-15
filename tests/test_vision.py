"""Visão (encoder + crítico), regras de estúdio e novos caminhos de amostragem."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch é necessário")
class VisionTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from src.models.config import ModelConfig
        from src.training.diffusion_model import DiffusionTrainingModel
        self.torch = torch
        self.config = ModelConfig.lite(base_channels=8, time_embedding_dim=16, num_timesteps=10, generation_steps=3)
        self.model = DiffusionTrainingModel(self.config)

    def test_references_change_prediction(self) -> None:
        images, times = self.torch.randn(1, 3, 32, 32), self.torch.tensor([5])
        refs = self.torch.rand(1, 2, 3, 48, 48)
        self.model.denoiser.eval()
        plain = self.model.denoiser(images, times, ["heroína"])
        seen = self.model.denoiser(images, times, ["heroína"], reference_images=refs)
        self.assertEqual(seen.shape, images.shape)
        self.assertFalse(self.torch.allclose(plain, seen))

    def test_sample_with_negative_reference_and_init(self) -> None:
        out = self.model.sample(["a"], 3, 2.0, negative_prompts=["blur"], reference_images=self.torch.rand(1, 1, 3, 32, 32),
                                init_images=self.torch.rand(1, 3, 40, 40), strength=0.5)
        self.assertEqual(out.shape, (1, 3, 32, 32))
        self.assertTrue(0 <= out.min() and out.max() <= 1)

    def test_training_step_uses_vision(self) -> None:
        from types import SimpleNamespace
        from src.models.tensors import Tensor
        values = tuple(float(v % 7) / 7 for v in range(2 * 3 * 32 * 32))
        batch = SimpleNamespace(inputs=Tensor((2, 3, 32, 32), values), metadata=(SimpleNamespace(caption="a"), SimpleNamespace(caption="b")))
        self.assertGreater(self.model.train_step(batch), 0)

    def test_critic_finish_and_png_roundtrip(self) -> None:
        from src.studio import StudioStyleCritic, apply_studio_finish, build_prompt
        from src.studio.image_io import load_image, save_png
        image = self.torch.zeros(3, 64, 64)
        image[:, 10:50, 30:60] = self.torch.tensor([0.9, 0.4, 0.5])[:, None, None]
        report = StudioStyleCritic().analyze(image)
        self.assertTrue(any(c.key == "lineart_preto" and not c.passed for c in report.checks))
        self.assertIn("RGB(61,43,40)", report.correction_prompt)
        fixed = apply_studio_finish(image)
        self.assertGreater(fixed[0, 0, 0].item(), fixed[2, 0, 0].item())
        self.assertIn("cel shading", build_prompt("menina", "idol"))
        with TemporaryDirectory() as folder:
            path = save_png(fixed, Path(folder) / "x.png")
            self.assertEqual(tuple(load_image(path).shape), (3, 64, 64))

    def test_checkpoint_roundtrip_recreates_architecture(self) -> None:
        from src.training.diffusion_model import DiffusionTrainingModel
        with TemporaryDirectory() as folder:
            path = self.model.save_checkpoint(Path(folder) / "m.pt", 1, {})
            loaded = DiffusionTrainingModel.from_checkpoint(path)
            self.assertEqual(loaded.config, self.config)

    def test_domain_validation_and_enforcement(self) -> None:
        from src.studio import build_prompt, enforce_manga_anime_style, validate_manga_anime_domain

        valid, msg = validate_manga_anime_domain("guerreira anime em um castelo")
        self.assertTrue(valid)

        invalid, msg = validate_manga_anime_domain("a photorealistic 3d render of a real human person")
        self.assertFalse(invalid)
        self.assertIn("photorealistic", msg)

        enforced = enforce_manga_anime_style("photorealistic photo of a warrior, 3d render")
        self.assertNotIn("photorealistic", enforced.lower())
        self.assertNotIn("3d render", enforced.lower())
        self.assertIn("anime", build_prompt("guerreira").lower())

    def test_terminal_visualization_and_html_gallery(self) -> None:
        from src.studio import StudioStyleCritic, render_terminal_image, save_html_gallery

        image = self.torch.rand(3, 32, 32)
        rendered = render_terminal_image(image, max_cols=20)
        self.assertIn("\033[38;2;", rendered)
        self.assertIn("▀", rendered)

        report = StudioStyleCritic().analyze(image)
        with TemporaryDirectory() as folder:
            html_file = Path(folder) / "view.html"
            out_path = save_html_gallery(image, html_file, report, prompt="guerreira anime")
            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("Visualizador de Estúdio AI Anime & Manga", content)
            self.assertIn("guerreira anime", content)

    def test_manga_screentone_finish(self) -> None:
        from src.studio import FinishConfig, apply_studio_finish

        image = self.torch.rand(3, 32, 32)
        manga_out = apply_studio_finish(image, FinishConfig(manga_mode=True, screentone=True))
        self.assertEqual(manga_out.shape, (3, 32, 32))
        self.assertTrue(self.torch.equal(manga_out[0], manga_out[1]))

    def test_xlarge_model_config(self) -> None:
        from src.models.config import ModelConfig

        xlarge_cfg = ModelConfig.xlarge()
        self.assertEqual(xlarge_cfg.image_resolution, (512, 512))
        self.assertEqual(xlarge_cfg.generation_profile, "xlarge")


if __name__ == "__main__":
    unittest.main()
