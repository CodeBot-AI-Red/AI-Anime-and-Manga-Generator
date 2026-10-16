"""Tests for anime settei sheets and cel shading pipeline."""

import unittest
import torch

from src.anime import CelShadingPipeline, SetteiSheetGenerator


class TestAnimeModules(unittest.TestCase):

    def test_cel_shading(self) -> None:
        img = torch.rand((3, 32, 32))
        out = CelShadingPipeline.apply_cel_shading(img, num_levels=3)
        self.assertEqual(out.shape, (3, 32, 32))

    def test_rim_light(self) -> None:
        img = torch.rand((3, 32, 32))
        out = CelShadingPipeline.apply_rim_light(img)
        self.assertEqual(out.shape, (3, 32, 32))

    def test_settei_sheet(self) -> None:
        img = torch.ones((3, 64, 64))
        out = SetteiSheetGenerator.create_settei_layout(img)
        self.assertEqual(out.shape[0], 3)
        self.assertGreater(out.shape[1], 64)
        self.assertEqual(out.shape[2], 64)


if __name__ == "__main__":
    unittest.main()
