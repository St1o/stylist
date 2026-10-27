"""Оценка качества кадра для подсказок при сканировании.

Пороги - стартовые значения. Их надо подобрать по реальным кадрам iPhone
(диагностика `web/diag.html` и первые сеансы), поэтому они вынесены в константы.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ANALYSIS_WIDTH = 640  # кадры для подсказок сжимаем до этой ширины

CLIP_LEVEL = 250  # пиксель считается пересвеченным с этой яркости
DARK_BELOW = 70.0  # средняя яркость 0..255
BRIGHT_ABOVE = 200.0
CLIPPED_ABOVE = 0.04  # доля пересвеченных пикселей
BLURRY_BELOW = 60.0  # дисперсия лапласиана
SHAKY_ABOVE = 6.0  # среднее изменение яркости между соседними кадрами


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    clipped: float
    sharpness: float
    motion: float | None  # None у первого кадра

    def as_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 1),
            "clipped": round(self.clipped, 3),
            "sharpness": round(self.sharpness, 1),
            "motion": None if self.motion is None else round(self.motion, 2),
        }


def shrink(gray: np.ndarray) -> np.ndarray:
    """Уменьшает серый кадр до ANALYSIS_WIDTH по ширине (меньший не трогает)."""
    height, width = gray.shape[:2]
    if width <= ANALYSIS_WIDTH:
        return gray
    scale = ANALYSIS_WIDTH / width
    return cv2.resize(gray, (ANALYSIS_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def prepare(jpeg: bytes) -> np.ndarray | None:
    """JPEG потокового кадра -> маленький серый кадр. None, если файл не читается."""
    gray = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if gray is None else shrink(gray)


def decode_full(jpeg: bytes) -> tuple[int, int, np.ndarray] | None:
    """Полный кадр: (ширина, высота, маленький серый кадр). None, если файл не читается."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height, shrink(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def measure(small: np.ndarray, previous: np.ndarray | None = None) -> FrameQuality:
    motion = None
    if previous is not None and previous.shape == small.shape:
        motion = float(np.abs(small.astype(np.int16) - previous.astype(np.int16)).mean())
    return FrameQuality(
        brightness=float(small.mean()),
        clipped=float((small >= CLIP_LEVEL).mean()),
        sharpness=float(cv2.Laplacian(small, cv2.CV_64F).var()),
        motion=motion,
    )


def hints(quality: FrameQuality, check_motion: bool = True) -> list[dict]:
    """Что мешает снимать, по важности. Пустой список - кадр годится."""
    found = []
    if quality.brightness < DARK_BELOW:
        found.append({"code": "dark", "text": "Слишком темно: добавьте света"})
    if quality.brightness > BRIGHT_ABOVE or quality.clipped > CLIPPED_ABOVE:
        found.append({"code": "bright", "text": "Пересвет или блики: уберите прямой свет"})
    if check_motion and quality.motion is not None and quality.motion > SHAKY_ABOVE:
        found.append({"code": "shaky", "text": "Держите телефон ровнее"})
    if quality.sharpness < BLURRY_BELOW:
        found.append({"code": "blurry", "text": "Нет резкости: дождитесь фокуса или чуть отодвиньтесь"})
    return found
