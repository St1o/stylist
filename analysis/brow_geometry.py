"""Геометрия брови по маске: осевая линия и профиль ширины вдоль брови.

Модель лица сюда не входит: на вход бинарная маска и положение головы (внутреннего
конца) брови. Длины в пикселях, делить на масштаб (расстояние между глазами) нужно снаружи.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_PIXELS = 40


@dataclass(frozen=True)
class BrowProfile:
    length: float  # длина вдоль оси, px
    axis_angle: float  # наклон оси голова -> хвост, градусы; + : хвост выше головы
    arch: float  # выпуклость осевой линии над хордой, px; + : дугой вверх
    width: np.ndarray  # толщина брови по точкам от головы к хвосту, px
    center: np.ndarray  # центр брови по тем же точкам, форма (bins, 2), x и y


def brow_profile(mask: np.ndarray, head: tuple[float, float], bins: int = 10) -> BrowProfile | None:
    """Профиль брови или None, если в маске слишком мало пикселей."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    head_xy = np.array(head, dtype=float)
    centered = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    u = vectors[:, -1]  # главная ось маски
    if np.dot(pts.mean(axis=0) - head_xy, u) < 0:
        u = -u  # ось смотрит от головы к хвосту
    v = np.array([u[1], -u[0]])
    if v[1] > 0:
        v = -v  # v смотрит вверх на снимке (y растёт вниз)
    s = (pts - head_xy) @ u
    p = (pts - head_xy) @ v
    lo, hi = np.percentile(s, [1, 99])
    edges = np.linspace(lo, hi, bins + 1)
    step = edges[1] - edges[0]
    width = np.zeros(bins)
    center = np.zeros((bins, 2))
    for k in range(bins):
        inside = (s >= edges[k]) & (s <= edges[k + 1])
        if inside.sum() == 0:
            width[k] = 0.0
            center[k] = head_xy + u * (edges[k] + step / 2)
            continue
        width[k] = inside.sum() / step  # площадь на длину: средняя толщина
        center[k] = head_xy + u * (edges[k] + step / 2) + v * p[inside].mean()
    chord = center[-1] - center[0]
    angle = math.degrees(math.atan2(-chord[1], abs(chord[0]) + 1e-9))
    normal = np.array([chord[1], -chord[0]]) / (np.linalg.norm(chord) + 1e-9)
    if normal[1] > 0:
        normal = -normal  # «вверх»
    offsets = (center - center[0]) @ normal
    middle = offsets[bins // 3 : bins - bins // 3]
    return BrowProfile(
        length=float(hi - lo),
        axis_angle=float(angle),
        arch=float(middle.mean()),
        width=width,
        center=center,
    )
