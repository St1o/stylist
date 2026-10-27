import unittest

from server import protocol
from server.protocol import STEADY_FRAMES, ScanSession, Step
from tests.helpers import blurred, texture


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


STEPS = (
    Step("a", "Первый", "первый шаг"),
    Step("b", "Серия", "второй шаг", frames=3, interval=0.5, check_motion=False),
)


def make(clock=None):
    return ScanSession("test", STEPS, clock=clock or FakeClock())


def steady_until_capture(session, frame=None):
    """Кормит сеанс одним и тем же хорошим кадром до команды «снимай»."""
    frame = texture() if frame is None else frame
    replies = [session.feed(frame) for _ in range(STEADY_FRAMES + 1)]
    return replies


class ScanSessionTest(unittest.TestCase):
    def test_capture_after_enough_steady_frames(self):
        flags = [reply["capture"] for reply in steady_until_capture(make())]
        # первый кадр без предыдущего: тряска неизвестна и не засчитывается
        self.assertEqual(flags, [False] * STEADY_FRAMES + [True])

    def test_bad_frame_resets_the_streak(self):
        session = make()
        for _ in range(STEADY_FRAMES):
            session.feed(texture())
        reply = session.feed(blurred())
        self.assertEqual(reply["steady"], 0)
        self.assertFalse(reply["capture"])
        self.assertIn("blurry", [tip["code"] for tip in reply["hints"]])

    def test_waits_for_upload_then_advances(self):
        session = make()
        steady_until_capture(session)
        self.assertTrue(session.feed(texture())["waiting"])
        self.assertFalse(session.accept_capture("b"))  # не тот шаг
        self.assertTrue(session.accept_capture("a"))
        self.assertEqual(session.current.key, "b")

    def test_series_closes_on_the_last_frame(self):
        session = make()
        session.skip()
        self.assertEqual(session.current.key, "b")
        self.assertTrue(session.accept_capture("b", 0))
        self.assertTrue(session.accept_capture("b", 2))
        self.assertEqual(session.current.key, "b")
        self.assertFalse(session.accept_capture("b", 3))  # вне серии
        self.assertTrue(session.accept_capture("b", 1))
        self.assertTrue(session.done)

    def test_series_step_ignores_hand_motion(self):
        session = make()
        session.skip()
        frame = texture()
        # рука в кадре: сдвиг каждый кадр, но для серии тряска не проверяется
        replies = [session.feed(frame if i % 2 else texture(1)) for i in range(STEADY_FRAMES)]
        self.assertTrue(replies[-1]["capture"])

    def test_timeout_returns_to_evaluating(self):
        clock = FakeClock()
        session = make(clock)
        steady_until_capture(session)
        clock.now = protocol.CAPTURE_TIMEOUT + 1
        reply = session.feed(texture())
        self.assertFalse(reply["waiting"])
        self.assertIsNotNone(reply["quality"])

    def test_force_capture_ignores_quality(self):
        session = make()
        session.force_capture()
        self.assertTrue(session.feed(blurred())["capture"])

    def test_skip_is_remembered_and_finishes(self):
        session = make()
        session.skip()
        session.skip()
        self.assertTrue(session.done)
        self.assertEqual(session.skipped, ["a", "b"])
        reply = session.feed(texture())
        self.assertTrue(reply["done"])
        self.assertIsNone(reply["step"])

    def test_default_protocol_has_hair_moving_steps(self):
        moving = [step for step in protocol.BROW_SCAN if step.frames > 1]
        self.assertEqual([step.key for step in moving], ["brow_left_move", "brow_right_move"])


if __name__ == "__main__":
    unittest.main()
