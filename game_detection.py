import numpy as np
from typing import Optional, Tuple

import cv2


def find_game_window(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """العثور على نافذة اللعبة: (x, y, width, height)

    منهجية (بدون ML):
    - تحويل لرمادي + GaussianBlur
    - كشف الحواف (Canny) ثم Dilate
    - إيجاد Contours الخارجية
    - اختيار أفضل مستطيل منطقي بحسب:
        * مساحة كبيرة لكن ليست كامل الشاشة
        * مستطيلية (ContourArea / RectArea)
        * نسبة أبعاد معقولة

    Returns:
        (x, y, w, h) أو None
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    h, w = frame.shape[:2]


    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Tuple[int, int, int, int]] = None
    best_score = 0.0

    min_area = 0.10 * (w * h)
    max_area = 0.95 * (w * h)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        rect_area = float(bw * bh)
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        aspect = bw / float(bh)


        if aspect < 0.6 or aspect > 2.2:
            continue


        score = rect_area * rectangularity

        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(bw), int(bh))

    return best


def is_game_active(frame: np.ndarray) -> bool:
    """التحقق من أن اللعبة نشطة (Heuristic سريع).

    الفكرة:
    - إذا كانت نافذة اللعبة تحتوي نسبة ملحوظة من ألوان كرات زوما (ألوان مشبعة)
      غالباً اللعبة شغالة.

    Returns:
        True إذا كانت المؤشرات تدل أن اللعبة نشطة، وإلا False
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False

    # تصغير لتسريع
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if small.ndim != 3 or small.shape[2] != 3:
        return False

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)


    masks = []

    # Red (جزئين)
    masks.append(cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)))
    masks.append(cv2.inRange(hsv, (170, 80, 80), (179, 255, 255)))

    # Blue / Green / Yellow-Orange / Purple
    masks.append(cv2.inRange(hsv, (95, 80, 80), (130, 255, 255)))   # blue
    masks.append(cv2.inRange(hsv, (35, 80, 80), (85, 255, 255)))    # green
    masks.append(cv2.inRange(hsv, (15, 80, 80), (35, 255, 255)))    # yellow/orange-ish
    masks.append(cv2.inRange(hsv, (135, 60, 60), (165, 255, 255)))  # purple/magenta

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    ratio = float(np.count_nonzero(combined)) / float(combined.size)


    return ratio > 0.01
