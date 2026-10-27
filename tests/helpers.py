"""Общие синтетические кадры для тестов: настоящих снимков лиц в репозитории нет."""
from __future__ import annotations

import cv2
import numpy as np


def texture(seed: int = 0, shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Резкий кадр средней яркости (шум)."""
    return np.random.default_rng(seed).integers(60, 190, shape, dtype=np.uint8)


def blurred(seed: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(texture(seed), (0, 0), 8)


def jpeg(gray: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", gray)
    assert ok
    return encoded.tobytes()
