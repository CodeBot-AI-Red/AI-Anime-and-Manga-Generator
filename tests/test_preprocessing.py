"""Testes do pré-processamento local de imagens PNG do dataset."""

from __future__ import annotations

from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import zlib

from src.data import (
    DatasetPreprocessor,
    DatasetSample,
    ImageInfo,
    ImageMetadata,
    PreprocessingConfig,
    PreprocessingError,
)


def png_bytes(width: int, height: int, pixels: list[tuple[int, int, int]]) -> bytes:
    """Cria um PNG RGB simples para testes, sem adicionar imagens ao repositório."""
    raw = b"".join(b"\x00" + bytes(component for pixel in pixels[row * width:(row + 1) * width] for component in pixel) for row in range(height))

    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.metadata = ImageMetadata(character="Aiko", caption="Aiko sorri.")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def sample(self, name: str, width: int, height: int, pixels: list[tuple[int, int, int]]) -> DatasetSample:
        path = self.root / name
        path.write_bytes(png_bytes(width, height, pixels))
        return DatasetSample(ImageInfo(path, "png", width, height), self.metadata)

    def test_resizes_image_to_configured_resolution(self) -> None:
        sample = self.sample("resize.png", 2, 1, [(255, 0, 0), (0, 255, 0)])
        result = DatasetPreprocessor(PreprocessingConfig(resolution=(4, 2))).process(sample)
        self.assertEqual(result.tensor.shape, (3, 2, 4))
        self.assertEqual(result.tensor.values[:4], (1.0, 1.0, 0.0, 0.0))

    def test_normalizes_pixels_and_converts_to_channel_first_tensor(self) -> None:
        sample = self.sample("tensor.png", 1, 1, [(128, 64, 255)])
        tensor = DatasetPreprocessor(PreprocessingConfig(resolution=(1, 1))).process(sample).tensor
        self.assertEqual(tensor.shape, (3, 1, 1))
        self.assertEqual(tensor.values, (128 / 255, 64 / 255, 1.0))

    def test_processes_batch_and_preserves_metadata(self) -> None:
        samples = [
            self.sample("one.png", 1, 1, [(0, 0, 0)]),
            self.sample("two.png", 1, 1, [(255, 255, 255)]),
        ]
        batch = DatasetPreprocessor(PreprocessingConfig(resolution=(2, 2), batch_size=2)).process_batch(samples)
        self.assertEqual(batch.tensor.shape, (2, 3, 2, 2))
        self.assertEqual(batch.metadata, (self.metadata, self.metadata))

    def test_rejects_invalid_image_and_incompatible_dimensions(self) -> None:
        bad_path = self.root / "bad.png"
        bad_path.write_bytes(b"not a png")
        invalid = DatasetSample(ImageInfo(bad_path, "png", 1, 1), self.metadata)
        processor = DatasetPreprocessor()
        with self.assertRaises(PreprocessingError):
            processor.process(invalid)

        mismatched = self.sample("mismatch.png", 1, 1, [(0, 0, 0)])
        mismatched = DatasetSample(ImageInfo(mismatched.image.path, "png", 2, 1), self.metadata)
        with self.assertRaises(PreprocessingError):
            processor.process(mismatched)

    def test_rejects_missing_data_and_oversized_batches(self) -> None:
        processor = DatasetPreprocessor(PreprocessingConfig(batch_size=1))
        with self.assertRaises(PreprocessingError):
            processor.process_batch([])
        sample = self.sample("one.png", 1, 1, [(0, 0, 0)])
        with self.assertRaises(PreprocessingError):
            processor.process(None)  # type: ignore[arg-type]
        with self.assertRaises(PreprocessingError):
            processor.process_batch([sample, sample])


if __name__ == "__main__":
    unittest.main()
