"""Synthetic checks for the real autograd diffusion path."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed from requirements.txt in the training environment")
class DiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from src.models.config import ModelConfig
        from src.models.diffusion import DiffusionDenoiser, TimestepEmbedding
        from src.models.scheduler import NoiseScheduler
        self.torch, self.config = torch, ModelConfig(image_resolution=(32, 32), base_channels=8, time_embedding_dim=16, num_timesteps=10, generation_steps=3)
        self.model, self.embedding, self.scheduler = DiffusionDenoiser(self.config), TimestepEmbedding(16), NoiseScheduler(10)

    def test_embedding_scheduler_and_forward_shapes(self) -> None:
        images, times = self.torch.randn(2, 3, 32, 32), self.torch.tensor([0, 9])
        noise = self.torch.randn_like(images)
        noisy = self.scheduler.add_noise(images, noise, times)
        self.assertEqual(self.embedding(times).shape, (2, 16))
        self.assertEqual(noisy.shape, images.shape)
        self.assertFalse(self.torch.equal(noisy, images))
        self.assertEqual(self.model(noisy, times, ["anime boy", "manga city"]).shape, images.shape)

    def test_training_sampling_and_checkpoint(self) -> None:
        from src.models import Tensor
        from src.training.data import TrainingBatch
        from src.training.diffusion_model import DiffusionTrainingModel
        batch = TrainingBatch(Tensor.zeros((2, 3, 32, 32)), Tensor.zeros((2, 3, 32, 32)), (object(), object()))
        trained = DiffusionTrainingModel(self.config, seed=1)
        before = next(trained.denoiser.parameters()).detach().clone()
        self.assertGreaterEqual(trained.train_step(batch), 0)
        self.assertFalse(self.torch.equal(before, next(trained.denoiser.parameters())))
        self.assertEqual(tuple(trained.sample(["anime hero"], 2).shape), (1, 3, 32, 32))
        with TemporaryDirectory() as directory:
            path = trained.save_checkpoint(Path(directory) / "model.pt", 1, {"train_loss": 1.0})
            self.assertEqual(DiffusionTrainingModel(self.config).load_checkpoint(path), 1)
