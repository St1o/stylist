"""Геометрия брови по маске: осевая линия и профиль ширины вдоль брови.

Модель лица сюда не входит: на вход бинарная маска и положение головы (внутреннего
конца) брови. Длины в пикселях, делить на масштаб (расстояние между глазами) нужно снаружи.
Центр брови считается по точкам вдоль оси, поэтому загиб хвоста дальше примерно 90° эта
геометрия не видит.
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
    head_tip: np.ndarray  # крайняя точка у носа (середина 2% пикселей по оси), x и y
    tail_tip: np.ndarray  # крайняя точка хвоста, x и y
    tail_turn: float  # на сколько хвост загибается вниз относительно тела брови, градусы


def grow_zone(core: np.ndarray, darkness: np.ndarray, low: float, radius: int, close: int) -> np.ndarray:
    """Зона брови: ядро плюс более светлые тонкие волоски рядом с ним.

    core - бинарная маска ядра, darkness - карта «тёмных тонких структур», low - порог слабее,
    чем у ядра, radius - на сколько пикселей зона может выйти за ядро, close - размер
    замыкания разрывов. Зона всегда содержит ядро и не уходит дальше radius от него.
    """
    import cv2

    reach = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    zone = ((darkness > low) & (reach > 0)) | (core > 0)
    zone = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    zone = ((zone > 0) | (core > 0)).astype(np.uint8)
    count, labels = cv2.connectedComponents(zone)
    keep = np.unique(labels[core > 0])
    keep = keep[keep > 0]
    return np.isin(labels, keep).astype(np.uint8)


def silhouette(core: np.ndarray, gap: int) -> np.ndarray:
    """Как бровь читается на расстоянии: щели уже gap пикселей замыкаются, внутренние дыры заливаются.

    Чем больше терпимая щель, тем более цельной, «блоком» читается бровь; при малой щели
    видны отдельные пучки и части.
    """
    import cv2

    closed = core.astype(np.uint8)
    if gap > 1:
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 1)
    return (((outside == 0)[1:-1, 1:-1]) | (closed > 0)).astype(np.uint8)


def lid_gap(mask: np.ndarray, lid: tuple[float, float], band: int = 3) -> float | None:
    """Расстояние в пикселях от нижнего края брови до верхнего века под точкой века (x, y).

    None, если над этой точкой века в маске брови нет.
    """
    col = int(round(lid[0]))
    rows = np.nonzero(mask[:, max(0, col - band) : col + band + 1].any(axis=1))[0]
    rows = rows[rows < lid[1]]
    return None if len(rows) == 0 else float(lid[1] - rows.max())


def gap_profile(mask: np.ndarray, lid: np.ndarray, count: int = 5) -> list[float | None]:
    """Просвет бровь-веко в count точках вдоль глаза от внутреннего угла к внешнему, в пикселях.

    lid - точки верхнего века (x, y) в порядке от внутреннего угла к внешнему. None там,
    где над точкой века в маске брови нет.
    """
    lid = np.asarray(lid, dtype=float)
    order = np.argsort(lid[:, 0])
    xs = np.linspace(lid[0, 0], lid[-1, 0], count)
    ys = np.interp(xs, lid[order, 0], lid[order, 1])
    return [lid_gap(mask, (float(x), float(y))) for x, y in zip(xs, ys)]


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

    def pointing(vector: np.ndarray) -> float:  # + : смотрит вверх, - : вниз, в осях брови
        return math.degrees(math.atan2(vector @ v, vector @ u))

    body = center[bins // 2] - center[0]
    tail = center[-1] - center[-(bins // 3) - 1]
    low, high = np.percentile(s, [2, 98])
    return BrowProfile(
        length=float(hi - lo),
        axis_angle=float(angle),
        arch=float(middle.mean()),
        width=width,
        center=center,
        head_tip=pts[s <= low].mean(axis=0),
        tail_tip=pts[s >= high].mean(axis=0),
        tail_turn=float(pointing(body) - pointing(tail)),
    )
