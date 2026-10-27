import math
import unittest

import cv2
import numpy as np

from analysis.brow_geometry import brow_profile, gap_profile, grow_zone, lid_gap, silhouette


def brow_mask(center_y, half_width, x0=20, x1=220, size=(200, 260), flip=False):
    """Синтетическая бровь: центр y(x) и полутолщина h(x), голова слева (x0)."""
    xs = np.arange(x0, x1 + 1)
    top = np.column_stack([xs, [center_y(x) - half_width(x) for x in xs]])
    bottom = np.column_stack([xs[::-1], [center_y(x) + half_width(x) for x in xs[::-1]]])
    mask = np.zeros(size, np.uint8)
    cv2.fillPoly(mask, [np.round(np.vstack([top, bottom])).astype(np.int32)], 1)
    if flip:
        mask = mask[:, ::-1].copy()
    return mask


class BrowGeometryTest(unittest.TestCase):
    def test_tapered_brow_gets_narrower_and_tail_up(self):
        mask = brow_mask(lambda x: 100 - 0.05 * (x - 20), lambda x: 10 - 8 * (x - 20) / 200)
        profile = brow_profile(mask, head=(20, 100))
        self.assertAlmostEqual(profile.axis_angle, math.degrees(math.atan(0.05)), delta=1.0)
        self.assertAlmostEqual(profile.width[0], 19, delta=3)
        self.assertLess(profile.width[-1], profile.width[0] / 2)
        self.assertAlmostEqual(profile.arch, 0, delta=1.5)
        self.assertAlmostEqual(profile.length, 200, delta=6)

    def test_arch_is_positive_when_the_middle_is_higher(self):
        mask = brow_mask(lambda x: 100 - 20 * math.sin(math.pi * (x - 20) / 200), lambda x: 5)
        self.assertGreater(brow_profile(mask, head=(20, 100)).arch, 10)

    def test_droop_is_negative_and_side_does_not_matter(self):
        droop = lambda x: 100 + 0.08 * (x - 20)
        left_side = brow_profile(brow_mask(droop, lambda x: 5), head=(20, 100))
        self.assertLess(left_side.axis_angle, -3)
        # зеркальная бровь другой стороны: голова справа, хвост слева
        mirrored = brow_profile(brow_mask(droop, lambda x: 5, flip=True), head=(260 - 1 - 20, 100))
        self.assertAlmostEqual(mirrored.axis_angle, left_side.axis_angle, delta=0.5)

    def test_tail_turn_and_tips(self):
        straight = brow_profile(brow_mask(lambda x: 100, lambda x: 5), head=(20, 100))
        self.assertLess(abs(straight.tail_turn), 5)
        self.assertAlmostEqual(straight.head_tip[0], 20, delta=8)
        self.assertAlmostEqual(straight.tail_tip[0], 220, delta=8)
        hooked = brow_profile(brow_mask(lambda x: 100 if x < 140 else 100 + 0.8 * (x - 140), lambda x: 5), head=(20, 100))
        self.assertGreater(hooked.tail_turn, 20)
        self.assertGreater(hooked.tail_tip[1], hooked.head_tip[1] + 30)

    def test_lid_gap_is_measured_from_the_lower_edge(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[80:101, 20:180] = 1  # бровь, нижний край на строке 100
        self.assertEqual(lid_gap(mask, (100, 130)), 30.0)
        self.assertIsNone(lid_gap(mask, (10, 130)))  # над этой точкой века брови нет

    def test_zone_takes_the_halo_but_not_far_noise(self):
        darkness = np.zeros((120, 200), np.uint8)
        core = np.zeros_like(darkness)
        core[50:70, 40:160] = 1
        darkness[core > 0] = 100
        darkness[40:50, 60:140] = 50  # светлая периферия рядом с ядром
        darkness[10:20, 60:100] = 60  # шум далеко от ядра
        zone = grow_zone(core, darkness, low=30, radius=12, close=3)
        self.assertEqual(zone[45, 100], 1)
        self.assertEqual(zone[15, 80], 0)
        self.assertTrue((zone[core > 0] == 1).all())

    def test_silhouette_merges_parts_only_across_small_gaps(self):
        core = np.zeros((40, 80), np.uint8)
        core[10:30, 10:35] = 1
        core[10:30, 41:70] = 1  # щель между частями шириной 6
        self.assertEqual(cv2.connectedComponents(silhouette(core, 3))[0] - 1, 2)
        self.assertEqual(cv2.connectedComponents(silhouette(core, 9))[0] - 1, 1)

    def test_silhouette_fills_holes(self):
        core = np.ones((30, 30), np.uint8)
        core[10:20, 10:20] = 0
        self.assertEqual(int(silhouette(core, 1).sum()), 900)

    def test_gap_profile_runs_from_inner_to_outer_corner(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[60:81, 20:180] = 1  # бровь, нижний край на строке 80
        # верхнее веко слева направо опускается: 100 -> 140
        lid = np.array([[40, 100], [100, 120], [160, 140]], float)
        self.assertEqual(gap_profile(mask, lid, count=3), [20.0, 40.0, 60.0])
        # тот же глаз, но внутренний угол справа: порядок точек обратный
        self.assertEqual(gap_profile(mask, lid[::-1], count=3), [60.0, 40.0, 20.0])

    def test_too_few_pixels(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[10, 10:15] = 1
        self.assertIsNone(brow_profile(mask, head=(10, 10)))


if __name__ == "__main__":
    unittest.main()
