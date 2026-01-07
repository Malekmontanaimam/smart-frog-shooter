import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import cv2



try:
    from config import MIN_BALL_RADIUS, MAX_BALL_RADIUS
except Exception:
    MIN_BALL_RADIUS, MAX_BALL_RADIUS = 10, 30


def _to_hsv_candidates(frame: np.ndarray) -> List[np.ndarray]:
    """إرجاع HSV بافتراض BGR ثم RGB.

    بعض أدوات تصوير الشاشة تعطي RGB، بينما OpenCV يتوقع BGR.
    لذلك نعالج الحالتين ونختار الأفضل لاحقاً.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return []


    hsv_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    rgb = frame[..., ::-1]
    hsv_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    return [hsv_bgr, hsv_rgb]


def _color_ranges_hsv() -> Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
    """نطاقات HSV عامة لألوان كرات زوما.

    ملاحظة: النطاقات تقريبية ومصممة لتكون 'مرنة' قدر الإمكان.
    يمكن تضييقها أو توسيعها حسب لقطات اللعبة.
    """
    return {
        # red يحتاج مجالين بسبب التفاف Hue
        "red": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (179, 255, 255))],
        "blue": [((95, 80, 80), (130, 255, 255))],
        "green": [((35, 80, 80), (85, 255, 255))],
        "yellow": [((18, 80, 80), (35, 255, 255))],
        "orange": [((10, 100, 100), (18, 255, 255))],
        "purple": [((135, 60, 60), (165, 255, 255))],
    }


def analyze_frame(frame: np.ndarray) -> Dict[str, Any]:
    """تحليل الإطار واستخراج حالة اللعبة.

    Returns مثال:
        {
          'balls': [{'color': 'red', 'position': (x,y), 'radius': r}, ...],
          'shooter_pos': (x, y) أو None,
          'effects': True/False,
        }
    """
    balls = detect_balls(frame)
    shooter_pos = detect_shooter(frame)
    effects = detect_effects(frame)

    return {
        "balls": balls,
        "shooter_pos": shooter_pos,
        "effects": effects,
    }


def detect_balls(frame: np.ndarray) -> List[Dict[str, Any]]:
    """كشف الكرات: [{'color': str, 'position': (x,y)}, ...]

    منهجية (بدون ML):
    - تحويل إلى HSV
    - Threshold لكل لون
    - عمليات Morphology لتنظيف الضجيج
    - Contours + EnclosingCircle لاستخراج مركز ونصف قطر الدائرة
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    hsv_candidates = _to_hsv_candidates(frame)
    if not hsv_candidates:
        return []

    ranges = _color_ranges_hsv()

    def detect_on_hsv(hsv: np.ndarray) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        kernel = np.ones((3, 3), np.uint8)

        for color_name, parts in ranges.items():
            mask_total = None
            for lo, hi in parts:
                m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
                mask_total = m if mask_total is None else cv2.bitwise_or(mask_total, m)

            if mask_total is None:
                continue


            mask = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=1)


            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                r = float(radius)
                if r < float(MIN_BALL_RADIUS) or r > float(MAX_BALL_RADIUS):
                    continue


                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity < 0.45:
                    continue

                results.append({
                    "color": color_name,
                    "position": (int(x), int(y)),
                    "radius": float(r),
                })

        return results


    detected_sets = [detect_on_hsv(hsv) for hsv in hsv_candidates]
    best = max(detected_sets, key=lambda lst: len(lst)) if detected_sets else []


    deduped: List[Dict[str, Any]] = []
    for b in best:
        bx, by = b["position"]
        br = b.get("radius", 0.0)
        found = False
        for d in deduped:
            dx, dy = d["position"]
            dr = d.get("radius", 0.0)
            if (bx - dx) ** 2 + (by - dy) ** 2 <= (max(br, dr) * 0.6) ** 2:
                found = True
                break
        if not found:
            deduped.append(b)

    return deduped


def detect_shooter(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """كشف موقع الضفدع (المطلق).

    حل عملي وسريع:
    - الضفدع غالباً قريب من مركز الشاشة.
    - نحاول كشف منطقة خضراء كبيرة في وسط الإطار.
    - إن فشلنا، نعيد مركز الإطار كافتراض مقبول.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    if frame.ndim != 3 or frame.shape[2] != 3:
        return None


    roi_w, roi_h = int(w * 0.45), int(h * 0.45)
    x0 = max(0, cx - roi_w // 2)
    y0 = max(0, cy - roi_h // 2)
    x1 = min(w, x0 + roi_w)
    y1 = min(h, y0 + roi_h)
    roi = frame[y0:y1, x0:x1]


    hsv_candidates = _to_hsv_candidates(roi)
    if not hsv_candidates:
        return None

    def shooter_on_hsv(hsv: np.ndarray) -> Optional[Tuple[int, int]]:

        mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None


        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 200:
            return None

        m = cv2.moments(best_cnt)
        if m.get("m00", 0) == 0:
            return None

        x = int(m["m10"] / m["m00"]) + x0
        y = int(m["m01"] / m["m00"]) + y0
        return (x, y)

    candidates = [shooter_on_hsv(hsv) for hsv in hsv_candidates]
    candidates = [c for c in candidates if c is not None]

    if candidates:
        return min(candidates, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    return None


def detect_effects(frame: np.ndarray) -> bool:
    """كشف المؤثرات البصرية (ومضات/انفجارات) بشكل سريع.

    Heuristic:
    - إذا كانت نسبة البيكسلات "شديدة السطوع" داخل الإطار كبيرة، نعتبر هناك مؤثر.
    - الهدف الأساسي: إعلام بقية النظام لتخفيف الاعتماد على هذا الإطار (إن أحببت).

    Returns:
        bool
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False

    if frame.ndim != 3 or frame.shape[2] != 3:
        return False

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)


    bright = cv2.inRange(hsv, (0, 30, 230), (179, 255, 255))
    ratio = float(np.count_nonzero(bright)) / float(bright.shape[0] * bright.shape[1])
    return ratio > 0.02
