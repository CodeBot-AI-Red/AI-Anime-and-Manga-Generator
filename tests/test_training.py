"""Testes do pipeline de treinamento local."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.training import GenerativeTrainingModel, ProcessedDataLoader, Trainer, TrainingConfig, mean_squared_error
from src.training.checkpoints import load_checkpoint
from src.training.mock import synthetic_samples
from src.models import Tensor


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.config = TrainingConfig(epochs=2, batch_size=2, checkpoint_dir=Path(self.temp_dir.name), seed=3)
        self.loader = ProcessedDataLoader(synthetic_samples(), self.config.batch_size, self.config.seed)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_configuration_validation_and_cpu_resolution(self) -> None:
        self.assertEqual(self.config.resolved_device, "cpu")
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)

    def test_loss_and_small_training_step(self) -> None:
        self.assertEqual(mean_squared_error(Tensor((1,), (1.0,)), Tensor((1,), (0.0,))), 1.0)
        trainer = Trainer(self.config)
        before = trainer.model.scale
        loss = trainer.train_step(next(self.loader.batches()))
        self.assertGreater(loss, 0)
        self.assertNotEqual(before, trainer.model.scale)

    def test_validation_checkpoint_and_resume(self) -> None:
        trainer = Trainer(self.config)
        history = trainer.fit(self.loader, self.loader)
        checkpoint_path = Path(self.temp_dir.name) / "epoch-2.json"
        self.assertTrue(checkpoint_path.exists())
        self.assertEqual(load_checkpoint(checkpoint_path)["epoch"], 2)
        self.assertEqual(len(history.validation_loss), 2)
        resumed = Trainer(self.config)
        resumed.resume(checkpoint_path)
        self.assertEqual(resumed.current_epoch, 2)
        self.assertEqual(resumed.model.state_dict(), trainer.model.state_dict())

    def test_generative_model_integrates_with_existing_trainer(self) -> None:
        trainer = Trainer(self.config, GenerativeTrainingModel())
        loss = trainer.train_step(next(self.loader.batches()))
        self.assertGreater(loss, 0)
        self.assertNotEqual(trainer.model.scale, 0.5)


if __name__ == "__main__":
    unittest.main()
