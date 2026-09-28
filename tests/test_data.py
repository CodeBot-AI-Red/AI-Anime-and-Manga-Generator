"""Testes dos componentes locais de preparação de datasets."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data import DatasetValidator, ImageLoader, MetadataReader, MetadataValidationError, split_dataset


PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x03"


class DatasetDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "images"
        self.annotations = self.root / "annotations"
        self.images.mkdir()
        self.annotations.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_image(self, name: str) -> Path:
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_HEADER)
        return path

    def test_loader_discovers_images_and_reads_png_dimensions(self) -> None:
        image = self.write_image("chapter/page.png")
        (self.images / "notes.md").write_text("ignorado", encoding="utf-8")

        self.assertEqual(ImageLoader().discover(self.images), [image])
        info = ImageLoader().inspect(image)
        self.assertEqual((info.format, info.width, info.height), ("png", 2, 3))

    def test_reader_merges_json_metadata_and_text_caption(self) -> None:
        image = self.write_image("chapter/page.png")
        annotation_dir = self.annotations / "chapter"
        annotation_dir.mkdir()
        (annotation_dir / "page.json").write_text(
            json.dumps({"character": "Aiko", "pose": "correndo", "style": "anime"}), encoding="utf-8"
        )
        (annotation_dir / "page.txt").write_text("Heroína correndo na cidade.", encoding="utf-8")

        metadata = MetadataReader().read(image, self.images, self.annotations)
        self.assertEqual(metadata.character, "Aiko")
        self.assertEqual(metadata.caption, "Heroína correndo na cidade.")

    def test_reader_rejects_missing_or_invalid_metadata(self) -> None:
        image = self.write_image("page.png")
        with self.assertRaises(MetadataValidationError):
            MetadataReader().read(image, self.images, self.annotations)

        (self.annotations / "page.json").write_text('{"camera": 12}', encoding="utf-8")
        with self.assertRaises(MetadataValidationError):
            MetadataReader().read(image, self.images, self.annotations)

    def test_validator_reports_invalid_images_and_annotations(self) -> None:
        self.write_image("good.png")
        (self.annotations / "good.json").write_text('{"scene_description": "Rua chuvosa"}', encoding="utf-8")
        (self.images / "broken.png").write_bytes(b"not an image")
        (self.annotations / "broken.txt").write_text("imagem quebrada", encoding="utf-8")
        self.write_image("without_metadata.png")

        samples, errors = DatasetValidator().validate(self.images, self.annotations)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("Conteúdo não reconhecido" in error for error in errors))
        self.assertTrue(any("Metadados ausentes" in error for error in errors))

    def test_split_is_deterministic_and_preserves_all_samples(self) -> None:
        samples = []
        for name in ("one.png", "two.png", "three.png", "four.png"):
            self.write_image(name)
            (self.annotations / name.replace(".png", ".txt")).write_text(name, encoding="utf-8")
        samples, errors = DatasetValidator().validate(self.images, self.annotations)

        first = split_dataset(samples, validation_fraction=0.25, seed="fixed")
        second = split_dataset(samples, validation_fraction=0.25, seed="fixed")
        self.assertEqual(errors, [])
        self.assertEqual(first, second)
        self.assertEqual(len(first.train), 3)
        self.assertEqual(len(first.validation), 1)
        self.assertEqual(set(first.train + first.validation), set(samples))


if __name__ == "__main__":
    unittest.main()
