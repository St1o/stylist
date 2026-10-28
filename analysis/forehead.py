"""Лоб в профиле: наклон, прямота и изгиб силуэта от надбровной дуги вверх.

Вход - левая (или правая) граница лица по строкам снимка в профиль: для каждой строки y
координата x переднего края. Ничего не знает о модели лица: границу даёт сегментация кожи или
ручная разметка. Углы от вертикали кадра, поэтому наклон головы на снимке входит в результат.
"""
from __future__ import annotations

import math

import numpy as np


def forehead_profile(y: np.ndarray, x: np.ndarray, facing: str = "left") -> dict:
    """Характеристики силуэта лба.

    y, x - строки и координата переднего края; строки от линии роста к надбровью в любом порядке.
    facing - куда смотрит лицо на снимке ("left" или "right").
    Возвращает: наклон (градусы от вертикали, + : лоб уходит назад), наклон у надбровья и у верха,
    изгиб (наклон у верха минус наклон у надбровья: + : лоб заваливается назад к макушке,
    - : ниже лоб наклонён сильнее, чем выше), прямоту (наибольшее отклонение от прямой в долях
    длины лба) и выступ надбровья (на сколько край у надбровья выдаётся вперёд относительно
    продолжения линии лба, в долях длины; - : надбровье не выступает).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(-y)  # от надбровья вверх
    y, x = y[order], x[order]
    if facing == "right":
        x = -x
    length = y[0] - y[-1]
    back = x[-1] - x[0]  # на сколько верх лба сдвинут назад относительно надбровья
    third = max(2, len(y) // 3)

    def angle(sel: slice) -> float:
        dy = y[sel][0] - y[sel][-1]
        return math.degrees(math.atan2(x[sel][-1] - x[sel][0], dy)) if dy else 0.0

    slope = np.polyfit(y, x, 1)[0]  # x растёт вверх, поэтому наклон отрицательный по y
    line = np.polyval(np.polyfit(y, x, 1), y)
    # прямую берём по верхней половине лба, чтобы выступ надбровья не искажал её
    upper = slice(len(y) // 2, None)
    upper_line = np.polyfit(y[upper], x[upper], 1)
    return {
        "angle": math.degrees(math.atan(-slope)),
        "angle_bottom": angle(slice(0, third)),
        "angle_top": angle(slice(len(y) - third, None)),
        "curve": angle(slice(len(y) - third, None)) - angle(slice(0, third)),
        "straightness": float(np.abs(x - line).max() / length),
        "ridge_overhang": float((np.polyval(upper_line, y[0]) - x[0]) / length),
        "back": float(back / length),
    }
