import unittest

import numpy as np

from server import quality
from tests.helpers import blurred, jpeg, texture


def codes(found):
    return [item["code"] for item in found]


class QualityTest(unittest.TestCase):
    def test_good_frame_has_no_hints(self):
        frame = texture()
        result = quality.measure(frame, frame)
        self.assertEqual(result.motion, 0.0)
        self.assertEqual(quality.hints(result), [])

    def test_first_frame_has_unknown_motion(self):
        self.assertIsNone(quality.measure(texture()).motion)

    def test_blur_is_detected(self):
        sharp = quality.measure(texture())
        soft = quality.measure(blurred())
        self.assertGreater(sharp.sharpness, soft.sharpness * 10)
        self.assertIn("blurry", codes(quality.hints(soft)))

    def test_dark_and_bright(self):
        dark = quality.measure(texture() // 4)
        self.assertIn("dark", codes(quality.hints(dark)))
        glare = quality.measure(np.full((480, 640), 255, dtype=np.uint8))
        self.assertIn("bright", codes(quality.hints(glare)))

    def test_shaky_hand_only_matters_when_motion_is_checked(self):
        shifted = quality.measure(np.roll(texture(), 20, axis=1), texture())
        self.assertIn("shaky", codes(quality.hints(shifted)))
        self.assertNotIn("shaky", codes(quality.hints(shifted, check_motion=False)))

    def test_prepare_shrinks_and_rejects_garbage(self):
        big = texture(shape=(1200, 1600))
        small = quality.prepare(jpeg(big))
        self.assertEqual(small.shape[1], quality.ANALYSIS_WIDTH)
        self.assertIsNone(quality.prepare(b"not an image"))

    def test_decode_full_keeps_original_size(self):
        width, height, small = quality.decode_full(jpeg(texture(shape=(1200, 1600))))
        self.assertEqual((width, height), (1600, 1200))
        self.assertEqual(small.shape[1], quality.ANALYSIS_WIDTH)
        self.assertIsNone(quality.decode_full(b"nope"))


if __name__ == "__main__":
    unittest.main()
