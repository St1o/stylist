"""Профиль каждой брови по снимку: маска (свой разбор), осевая линия, ширина вдоль брови.

Запуск: python -m analysis.brow_profile --model face_landmarker.task --out папка снимок1.jpg снимок2.jpg
Модель лица (face_landmarker.task) скачивается отдельно и в git не лежит. Готовая модель лица
используется только для области брови, глаз и масштаба: её точки бровей грубые и почти симметричные.
Снимки и результаты в git не кладём.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .brow_geometry import BrowProfile, brow_profile

# индексы канонической сетки MediaPipe; «правый/левый» - у человека на снимке
EYE_RING = {
    "right": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}
BROW = {
    "right": dict(up=[70, 63, 105, 66, 107], low=[46, 53, 52, 65, 55], head=107),
    "left": dict(up=[300, 293, 334, 296, 336], low=[276, 283, 282, 295, 285], head=336),
}
OUTER_CORNER = {"right": 33, "left": 263}
INNER_CORNER = {"right": 133, "left": 362}
BINS = 10


def detect_landmarks(image_bgr: np.ndarray, model_path: str) -> np.ndarray | None:
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model_path), num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        return None
    h, w = image_bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])


def level_eyes(image_bgr: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Поворачивает снимок так, чтобы линия глаз стала горизонтальной (убирает наклон головы набок)."""
    right = (points[OUTER_CORNER["right"]] + points[INNER_CORNER["right"]]) / 2
    left = (points[OUTER_CORNER["left"]] + points[INNER_CORNER["left"]]) / 2
    roll = math.degrees(math.atan2(left[1] - right[1], left[0] - right[0]))
    matrix = cv2.getRotationMatrix2D(tuple(float(v) for v in (right + left) / 2), roll, 1.0)
    h, w = image_bgr.shape[:2]
    rotated = cv2.warpAffine(image_bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    moved = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return rotated, moved, roll


def segment_brow(image_bgr: np.ndarray, points: np.ndarray, side: str, scale: float) -> np.ndarray:
    """Маска брови: тонкие тёмные структуры в области вокруг точек бровей, без глаза."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    spec = BROW[side]
    brow_pts = points[spec["up"] + spec["low"]].astype(np.int32)
    margin = max(3, int(0.07 * scale))
    area = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(area, cv2.convexHull(brow_pts), 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    area = cv2.dilate(area, kernel)
    eye = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(eye, cv2.convexHull(points[EYE_RING[side]].astype(np.int32)), 255)
    eye = cv2.dilate(eye, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    area[eye > 0] = 0
    # чёрная шляпа выделяет тёмные структуры тоньше ядра (брови), а широкую тень виска и волосы гасит
    size = max(5, int(0.14 * scale)) | 1
    thin_dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    values = thin_dark[area > 0]
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((thin_dark > threshold) & (area > 0)).astype(np.uint8)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(0.03 * scale)) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
    # оставляем компоненту, которая лучше всего совпадает с линией точек бровей
    line = np.zeros(gray.shape, np.uint8)
    middle = ((points[spec["up"]] + points[spec["low"]]) / 2).astype(np.int32)
    cv2.polylines(line, [middle], False, 255, thickness=max(3, 2 * margin))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_score = 0, 0
    for label in range(1, count):
        score = int(((labels == label) & (line > 0)).sum())
        if score > best_score:
            best, best_score = label, score
    return (labels == best).astype(np.uint8) if best else np.zeros_like(mask)


def analyse(image_bgr: np.ndarray, model_path: str) -> dict | None:
    points = detect_landmarks(image_bgr, model_path)
    if points is None:
        return None
    level, moved, roll = level_eyes(image_bgr, points)
    scale = float(np.linalg.norm(moved[OUTER_CORNER["right"]] - moved[OUTER_CORNER["left"]]))
    out = {"roll": roll, "scale": scale, "image": level, "brows": {}}
    for side in ("right", "left"):
        mask = segment_brow(level, moved, side, scale)
        head = tuple(moved[BROW[side]["head"]])
        out["brows"][side] = {"mask": mask, "profile": brow_profile(mask, head, BINS)}
    return out


def draw(result: dict) -> np.ndarray:
    vis = result["image"].copy()
    for side, color in (("right", (0, 255, 0)), ("left", (255, 160, 0))):
        brow = result["brows"][side]
        contours, _ = cv2.findContours(brow["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, color, 1)
        profile: BrowProfile | None = brow["profile"]
        if profile is not None:
            cv2.polylines(vis, [profile.center.astype(np.int32)], False, (0, 0, 255), 1)
    return vis


def report(name: str, result: dict) -> None:
    scale = result["scale"]
    print(f"\n{name}: масштаб (между внешними углами глаз) {scale:.0f} px, наклон головы набок {result['roll']:+.1f}°")
    for side in ("right", "left"):
        p: BrowProfile | None = result["brows"][side]["profile"]
        if p is None:
            print(f"  {side}: бровь не найдена")
            continue
        widths = " ".join(f"{w / scale:.3f}" for w in p.width)
        print(f"  {side}: длина {p.length / scale:.2f}, ось {p.axis_angle:+.1f}°, дуга {p.arch / scale:+.3f}; ширина от головы к хвосту: {widths}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Профиль бровей по снимкам")
    parser.add_argument("--model", required=True, help="файл face_landmarker.task")
    parser.add_argument("--out", type=Path, required=True, help="папка для наложений")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: не читается")
            continue
        result = analyse(image, args.model)
        if result is None:
            print(f"{path.name}: лицо не найдено")
            continue
        report(path.name, result)
        cv2.imwrite(str(args.out / f"{path.stem}_brows.png"), draw(result))


if __name__ == "__main__":
    main()
