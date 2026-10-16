"""Tests for manga layout, speed lines, and screentone generation."""

import unittest
import torch

from src.manga import MangaPanelLayout, ScreentoneConfig, apply_advanced_screentone, apply_speed_lines, generate_screentone_pattern


class TestMangaModules(unittest.TestCase):

    def test_apply_speed_lines(self) -> None:
        img = torch.ones((3, 64, 64))
        out = apply_speed_lines(img, line_density=0.5)
        self.assertEqual(out.shape, (3, 64, 64))
        self.assertTrue((out <= 1.0).all() and (out >= 0.0).all())

    def test_manga_panel_layout(self) -> None:
        img = torch.ones((3, 64, 64))
        out = MangaPanelLayout.draw_grid_gutters(img, rows=2, cols=2)
        self.assertEqual(out.shape, (3, 64, 64))

    def test_screentone_pattern(self) -> None:
        cfg = ScreentoneConfig(frequency=10.0, angle=45.0, dot_type="round")
        pat = generate_screentone_pattern((32, 32), cfg)
        self.assertEqual(pat.shape, (32, 32))

    def test_apply_advanced_screentone(self) -> None:
        luma = torch.linspace(0, 1, 32 * 32).view(32, 32)
        out = apply_advanced_screentone(luma)
        self.assertEqual(out.shape, (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
