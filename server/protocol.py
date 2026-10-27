"""Протокол сканирования и состояние одного сеанса.

Сеанс идёт по шагам. На каждом шаге сервер смотрит на поток кадров с телефона
и, когда свет, резкость и (если нужно) неподвижность в порядке несколько кадров
подряд, просит снять полный кадр (`capture`). Шаг закрывается, когда загружены
все его кадры. Позу головы (15°) сервер пока не проверяет: модуль лица ещё не
подключён, поэтому снимки таких шагов помечаются `pose_checked: false`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .quality import FrameQuality, hints, measure

STEADY_FRAMES = 3  # подряд хороших кадров до команды «снимай»
CAPTURE_TIMEOUT = 12.0  # секунд на загрузку, дальше сервер снова оценивает поток


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    instruction: str
    frames: int = 1  # сколько полных кадров нужно (серия при движении волосков)
    interval: float = 0.0  # пауза между кадрами серии, секунд
    check_motion: bool = True  # в серии рука в кадре, поэтому тряску не проверяем


BROW_SCAN: tuple[Step, ...] = (
    Step("front", "Анфас", "Лицо целиком в кадре, смотрите прямо, голова ровно."),
    Step("yaw_left", "Поворот влево", "Медленно поверните голову влево примерно на 15 градусов и замрите."),
    Step("yaw_right", "Поворот вправо", "Медленно поверните голову вправо примерно на 15 градусов и замрите."),
    Step("brow_left_close", "Левая бровь крупно", "Ваша левая бровь крупно, 10-15 сантиметров, волоски лежат как есть."),
    Step("brow_right_close", "Правая бровь крупно", "Ваша правая бровь крупно, 10-15 сантиметров, волоски лежат как есть."),
    Step(
        "brow_left_move",
        "Левая бровь: двигаем волоски",
        "После сигнала проведите пальцем или расчёской по брови против роста, потом по росту.",
        frames=12,
        interval=0.4,
        check_motion=False,
    ),
    Step(
        "brow_right_move",
        "Правая бровь: двигаем волоски",
        "После сигнала проведите пальцем или расчёской по брови против роста, потом по росту.",
        frames=12,
        interval=0.4,
        check_motion=False,
    ),
)


class ScanSession:
    def __init__(self, scan_id: str, steps: tuple[Step, ...] = BROW_SCAN, clock: Callable[[], float] = time.monotonic):
        self.scan_id = scan_id
        self.steps = steps
        self._clock = clock
        self._index = 0
        self._steady = 0
        self._previous: np.ndarray | None = None
        self._waiting_since: float | None = None
        self._received: set[int] = set()
        self._force = False
        self.skipped: list[str] = []

    @property
    def current(self) -> Step | None:
        return self.steps[self._index] if self._index < len(self.steps) else None

    @property
    def done(self) -> bool:
        return self._index >= len(self.steps)

    def feed(self, small: np.ndarray) -> dict:
        """Один кадр потока -> ответ клиенту."""
        step = self.current
        if step is None:
            return self._reply(done=True)
        previous, self._previous = self._previous, small
        if self._waiting_since is not None:
            timeout = CAPTURE_TIMEOUT + step.frames * step.interval
            if self._clock() - self._waiting_since < timeout:
                return self._reply(waiting=True)
            self._waiting_since = None
            self._steady = 0
        quality = measure(small, previous)
        tips = hints(quality, check_motion=step.check_motion)
        steady = not tips and (quality.motion is not None or not step.check_motion)
        self._steady = self._steady + 1 if steady else 0
        capture = self._force or self._steady >= STEADY_FRAMES
        if capture:
            self._force = False
            self._steady = 0
            self._waiting_since = self._clock()
            self._received = set()
        return self._reply(quality=quality, tips=tips, capture=capture)

    def accept_capture(self, step_key: str, frame: int = 0) -> bool:
        """Загружен кадр шага. True, если кадр принят; шаг закрывается на последнем."""
        step = self.current
        if step is None or step.key != step_key or not 0 <= frame < step.frames:
            return False
        self._received.add(frame)
        if len(self._received) >= step.frames:
            self._advance()
        return True

    def skip(self) -> None:
        step = self.current
        if step is not None:
            self.skipped.append(step.key)
            self._advance()

    def force_capture(self) -> None:
        """Владелец сам решил, что кадр хорош: следующий кадр потока запускает съёмку."""
        self._force = True

    def snapshot(self) -> dict:
        return self._reply()

    def _advance(self) -> None:
        self._index += 1
        self._steady = 0
        self._waiting_since = None
        self._received = set()
        self._force = False

    def _reply(
        self,
        quality: FrameQuality | None = None,
        tips: list[dict] | None = None,
        capture: bool = False,
        waiting: bool = False,
        done: bool = False,
    ) -> dict:
        step = self.current
        return {
            "type": "feedback",
            "done": done or self.done,
            "step": None
            if step is None
            else {
                "key": step.key,
                "title": step.title,
                "instruction": step.instruction,
                "index": self._index,
                "total": len(self.steps),
                "frames": step.frames,
                "interval": step.interval,
            },
            "quality": None if quality is None else quality.as_dict(),
            "hints": tips or [],
            "steady": self._steady,
            "need": STEADY_FRAMES,
            "capture": capture,
            "waiting": waiting,
        }
