import math
import unittest

import numpy as np

from analysis.forehead import forehead_profile


class ForeheadProfileTest(unittest.TestCase):
    def test_straight_sloped_forehead(self):
        y = np.arange(500, 800, 10)
        x = 100 + (800 - y) * math.tan(math.radians(20))  # лицо смотрит влево, лоб уходит назад вверх
        result = forehead_profile(y, x)
        self.assertAlmostEqual(result["angle"], 20, delta=0.5)
        self.assertLess(result["straightness"], 0.01)
        self.assertAlmostEqual(result["curve"], 0, delta=0.5)
        self.assertAlmostEqual(result["ridge_overhang"], 0, delta=0.01)

    def test_forehead_curving_back_at_the_top_has_positive_curve(self):
        y = np.arange(500, 800, 10)
        t = (800 - y) / 300
        self.assertGreater(forehead_profile(y, 100 + 60 * t**2)["curve"], 10)  # выше лоб наклонён сильнее

    def test_steeper_lower_forehead_has_negative_curve(self):
        y = np.arange(500, 800, 10)
        t = (800 - y) / 300
        self.assertLess(forehead_profile(y, 100 + 60 * (2 * t - t**2))["curve"], -10)

    def test_ridge_overhang(self):
        y = np.arange(500, 800, 10)
        x = 100 + (800 - y) * 0.3
        x[-3:] -= 15  # надбровье выдаётся вперёд
        self.assertGreater(forehead_profile(y, x)["ridge_overhang"], 0.02)

    def test_right_facing_matches_mirrored_left(self):
        y = np.arange(500, 800, 10)
        x = 100 + (800 - y) * 0.36
        left = forehead_profile(y, x, "left")
        right = forehead_profile(y, 1000 - x, "right")
        self.assertAlmostEqual(left["angle"], right["angle"], places=6)


if __name__ == "__main__":
    unittest.main()
