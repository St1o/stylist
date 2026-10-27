import unittest

import numpy as np

from analysis.edging import edge_width, zone_features


class EdgeWidthTest(unittest.TestCase):
    def test_sharp_edge_is_narrower_than_gradual(self):
        sharp = np.concatenate([np.full(40, 40.0), np.full(40, 160.0)])
        gradual = np.concatenate([np.full(10, 40.0), np.linspace(40, 160, 60), np.full(10, 160.0)])
        self.assertLess(edge_width(sharp, 1.0), 2.0)
        self.assertAlmostEqual(edge_width(gradual, 1.0), 0.8 * 59, delta=3)

    def test_low_contrast_gives_none(self):
        self.assertIsNone(edge_width(np.full(50, 100.0), 1.0))


class ZoneFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.face = np.zeros((300, 300), np.uint8)
        self.face[100:260, 80:220] = 1  # открытая зона: ширина 140, высота 160
        self.hair = np.zeros((300, 300), np.uint8)
        self.hair[50:100, 70:230] = 1  # шапка волос над лбом толщиной 50

    def test_forehead_width_and_volume(self):
        features = zone_features(self.face, self.hair, brow_y=140, chin_y=260, scale=100.0)
        self.assertAlmostEqual(features["forehead"], 0.4, places=2)
        self.assertAlmostEqual(features["face_width"], 1.39, places=2)
        self.assertAlmostEqual(features["hair_top"], 0.5, places=2)
        self.assertAlmostEqual(features["hair_ratio"], 50 * 160 / (160 * 140), places=2)

    def test_side_hair_reaches_lower_on_one_side(self):
        self.hair[100:200, 60:80] = 1  # слева волосы спускаются до строки 200
        features = zone_features(self.face, self.hair, brow_y=140, chin_y=260, scale=100.0)
        self.assertAlmostEqual(features["side_left"], 0.6, places=1)
        self.assertLess(features["side_right"], 0.0)  # справа волосы выше бровей

    def test_empty_face_gives_none(self):
        self.assertIsNone(zone_features(np.zeros((10, 10), np.uint8), self.hair[:10, :10], 5, 9, 10.0))


if __name__ == "__main__":
    unittest.main()
