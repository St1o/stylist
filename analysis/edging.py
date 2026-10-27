"""Окантовка по снимку: открытая зона лица и зона головы, считаются автоматически.

Маски даёт готовая модель сегментации (MediaPipe selfie_multiclass_256x256: 1 - волосы, 3 - кожа лица),
файл модели скачивается отдельно и в git не лежит. На двух снимках с обводкой владельца автомаска
совпала с его открытой зоной лица на 0.95-0.97, а с зоной головы на 0.88-0.91 (`refs/look-1/notes.md`).

Запуск: python -m analysis.edging --face-model face_landmarker.task --seg-model selfie_multiclass.tflite снимок1.jpg ...
Длины в долях расстояния между внешними углами глаз.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .brow_profile import FACE_OVAL, analyse

HAIR, FACE_SKIN = 1, 3


def edge_width(profile: np.ndarray, step: float, min_contrast: float = 8.0) -> float | None:
    """Ширина перехода 10-90% между волосами и кожей по профилю яркости.

    profile - яркость вдоль линии от волос (начало) к коже (конец), step - шаг между отсчётами.
    None, если контраст между волосами и кожей слишком мал.
    """
    profile = np.asarray(profile, dtype=float)
    count = max(2, len(profile) // 5)
    hair, skin = profile[:count].mean(), profile[-count:].mean()
    if abs(skin - hair) < min_contrast:
        return None
    level = np.maximum.accumulate((profile - hair) / (skin - hair))  # переход считаем монотонным
    low, high = np.argmax(level >= 0.1), np.argmax(level >= 0.9)
    if level[low] < 0.1 or level[high] < 0.9:
        return None
    return float((high - low) * step)


def zone_features(face: np.ndarray, hair: np.ndarray, brow_y: float, chin_y: float, scale: float) -> dict | None:
    """Признаки открытой зоны лица и зоны головы по двум бинарным маскам."""
    ys, xs = np.nonzero(face)
    if len(xs) == 0:
        return None
    left, right = xs.min(), xs.max()
    width = right - left
    centre = (np.arange(face.shape[1]) > (left + right) / 2 - 0.2 * width) & (np.arange(face.shape[1]) < (left + right) / 2 + 0.2 * width)
    centre_rows = np.nonzero(face[:, centre].any(axis=1))[0]
    top = centre_rows.min()
    # волосы головы: в полосе лица и не ниже подбородка
    head = hair.copy()
    head[int(chin_y) + 1 :] = 0
    head[:, : max(0, int(left - 0.9 * scale))] = 0
    head[:, int(right + 0.9 * scale) :] = 0
    hy, hx = np.nonzero(head)
    out = {
        "forehead": (brow_y - top) / scale,  # от бровей до верха открытой зоны
        "face_width": width / scale,
        "hair_ratio": float(head.sum() / face.sum()),
        "hair_top": (top - hy.min()) / scale if len(hy) else 0.0,
    }
    row = int(brow_y)
    brow_row = np.nonzero(face[row])[0]
    if len(brow_row):
        edge_left, edge_right = brow_row.min(), brow_row.max()
        for name, lo, hi in (("left", edge_left - 0.8 * scale, edge_left + 0.2 * scale), ("right", edge_right - 0.2 * scale, edge_right + 0.8 * scale)):
            band = head[:, max(0, int(lo)) : max(0, int(hi))]
            rows = np.nonzero(band.any(axis=1))[0]
            out[f"side_{name}"] = (rows.max() - brow_y) / scale if len(rows) else None  # насколько низко волосы по краю лица
    return out


def segment(image_bgr: np.ndarray, model_path: str) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    options = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=model_path), output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    return result.category_mask.numpy_view().copy().reshape(image_bgr.shape[:2])


def fade_band(image_bgr: np.ndarray, face: np.ndarray, brow_y: float, scale: float) -> float | None:
    """Средняя ширина перехода кожа-волосы у висков (чем шире, тем плавнее фейд), в долях масштаба."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), max(1.0, 0.01 * scale))
    reach = int(0.3 * scale)
    widths = []
    for y in range(int(brow_y - 0.3 * scale), int(brow_y - 0.05 * scale)):
        row = np.nonzero(face[y])[0]
        if len(row) == 0:
            continue
        for edge, direction in ((row.min(), 1), (row.max(), -1)):  # от волос к коже
            xs = np.arange(edge - direction * reach, edge + direction * reach, direction)
            if xs.min() < 0 or xs.max() >= lab.shape[1]:
                continue
            width = edge_width(lab[y, xs], 1.0 / scale)
            if width is not None:
                widths.append(width)
    return float(np.median(widths)) if len(widths) >= 6 else None


def edging(result: dict, categories: np.ndarray) -> dict | None:
    """Окантовка по результату analyse (выровненный кадр) и маске категорий."""
    pts, scale = result["points"], result["scale"]
    oval = pts[FACE_OVAL]
    face = (categories == FACE_SKIN).astype(np.uint8)
    face[:, : int(oval[:, 0].min() - 0.05 * scale)] = 0
    face[:, int(oval[:, 0].max() + 0.05 * scale) :] = 0
    hair = (categories == HAIR).astype(np.uint8)
    left, right = result["brows"]["right"]["profile"], result["brows"]["left"]["profile"]
    if left is None or right is None:
        return None
    brow_y = (left.center[:, 1].mean() + right.center[:, 1].mean()) / 2
    features = zone_features(face, hair, brow_y, pts[152][1], scale)
    if features is not None:
        features["fade_band"] = fade_band(result["image"], face, brow_y, scale)
    return features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Окантовка по снимкам")
    parser.add_argument("--face-model", required=True)
    parser.add_argument("--seg-model", required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print("снимок | лоб | ширина | объём волос | толщина над лбом | спуск слева | спуск справа | переход у виска")
    for path in args.images:
        image = cv2.imread(str(path))
        result = None if image is None else analyse(image, args.face_model)
        if result is None:
            print(f"{path.name} | лицо не найдено")
            continue
        features = edging(result, segment(result["image"], args.seg_model))
        if features is None:
            print(f"{path.name} | зоны не найдены")
            continue
        cell = lambda v, f="{:.2f}": "-" if v is None else f.format(v)
        print(" | ".join([path.name, cell(features["forehead"]), cell(features["face_width"]), cell(features["hair_ratio"]), cell(features["hair_top"]),
                          cell(features.get("side_left")), cell(features.get("side_right")), cell(features["fade_band"], "{:.3f}")]))


if __name__ == "__main__":
    main()
