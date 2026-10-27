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

from .brow_geometry import BrowProfile, brow_profile, gap_profile, grow_zone, lid_gap

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
LID_TOP = {"right": 159, "left": 386}  # середина верхнего века
# верхнее веко от внутреннего угла глаза к внешнему
LID_ARC = {"right": [133, 173, 157, 158, 159, 160, 161, 246, 33], "left": [362, 398, 384, 385, 386, 387, 388, 466, 263]}
INNER_CORNER = {"right": 133, "left": 362}
BINS = 10
# Зона (светлая периферия вокруг ядра): порог слабее порога ядра и допустимый вылет за ядро.
# Подобраны по одной брови одного снимка (разметка владельца, IoU 0.84), на остальных не проверены.
ZONE_LOW = 0.45
ZONE_REACH = 0.03


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


def brow_darkness(image_bgr: np.ndarray, points: np.ndarray, side: str, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Карта тонких тёмных структур, область поиска брови (без глаза) и порог ядра (Оцу)."""
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
    return thin_dark, area, float(threshold)


def segment_brow(image_bgr: np.ndarray, points: np.ndarray, side: str, scale: float) -> np.ndarray:
    """Маска ядра брови: тонкие тёмные структуры в области вокруг точек бровей, без глаза."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    thin_dark, area, threshold = brow_darkness(image_bgr, points, side, scale)
    spec = BROW[side]
    margin = max(3, int(0.07 * scale))
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
    out = {"roll": roll, "scale": scale, "image": level, "points": moved, "brows": {}}
    for side in ("right", "left"):
        mask = segment_brow(level, moved, side, scale)
        head = tuple(moved[BROW[side]["head"]])
        dark, _, threshold = brow_darkness(level, moved, side, scale)
        zone = grow_zone(mask, dark, ZONE_LOW * threshold, max(2, int(ZONE_REACH * scale)), max(3, int(0.03 * scale)) | 1)
        out["brows"][side] = {
            "mask": mask,
            "zone": zone,
            "profile": brow_profile(mask, head, BINS),
            "lid_gap": lid_gap(mask, tuple(moved[LID_TOP[side]])),
            "gap_profile": gap_profile(mask, moved[LID_ARC[side]]),
        }
    return out


def draw(result: dict) -> np.ndarray:
    vis = result["image"].copy()
    for side, color in (("right", (0, 255, 0)), ("left", (255, 160, 0))):
        brow = result["brows"][side]
        zone_contours, _ = cv2.findContours(brow["zone"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, zone_contours, -1, (0, 255, 255), 1)
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
        along = result["brows"][side]["gap_profile"]
        along_text = " ".join("-" if g is None else f"{g / scale:.3f}" for g in along)
        print(f"  {side}: просвет бровь-веко вдоль глаза от внутреннего угла к внешнему: {along_text}")
        gap = result["brows"][side]["lid_gap"]
        gap_text = "нет" if gap is None else f"{gap / scale:.3f}"
        print(f"  {side}: просвет бровь-веко {gap_text}, длина {p.length / scale:.2f}, ось {p.axis_angle:+.1f}°, дуга {p.arch / scale:+.3f}; ширина от головы к хвосту: {widths}")


def balance(result: dict) -> dict | None:
    """Сравнение левой и правой бровей на экране (слева на экране - правая бровь человека)."""
    screen_left = result["brows"]["right"]["profile"]
    screen_right = result["brows"]["left"]["profile"]
    if screen_left is None or screen_right is None:
        return None
    scale, pts = result["scale"], result["points"]
    midline = (pts[INNER_CORNER["right"]][0] + pts[INNER_CORNER["left"]][0]) / 2
    out = {
        # + : слева на экране длиннее / ниже / сильнее загнут
        "length_diff": (screen_left.length - screen_right.length) / scale,
        "head_lower_left": (screen_left.head_tip[1] - screen_right.head_tip[1]) / scale,
        "gap": (screen_right.head_tip[0] - screen_left.head_tip[0]) / scale,
        "gap_shift_right": ((screen_right.head_tip[0] + screen_left.head_tip[0]) / 2 - midline) / scale,
    }
    for name, p in (("left", screen_left), ("right", screen_right)):
        out[f"tail_drop_{name}"] = (p.tail_tip[1] - p.head_tip[1]) / scale
        out[f"tail_turn_{name}"] = p.tail_turn
    return out


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
        diff = balance(result)
        if diff:
            print(
                "  экран слева/справа: хвост ниже головы {:+.3f}/{:+.3f} (загиб хвоста {:+.0f}°/{:+.0f}°); "
                "длиннее слева на {:+.3f}; голова слева ниже на {:+.3f}; промежуток между головами {:.3f}, "
                "сдвиг промежутка вправо от середины глаз {:+.3f}".format(
                    diff["tail_drop_left"], diff["tail_drop_right"], diff["tail_turn_left"], diff["tail_turn_right"],
                    diff["length_diff"], diff["head_lower_left"], diff["gap"], diff["gap_shift_right"],
                )
            )
        cv2.imwrite(str(args.out / f"{path.stem}_brows.png"), draw(result))


if __name__ == "__main__":
    main()
