import cv2
import numpy as np


def remove_visual_effects(frame: np.ndarray) -> np.ndarray:
    """
    إزالة المؤثرات البصرية (ضجيج – Blur خفيف)
    الهدف: تنظيف الصورة قبل التحليل
    """
    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # إزالة الضجيج باستخدام Gaussian Blur
    clean_frame = cv2.GaussianBlur(gray, (5, 5), 0)

    return clean_frame


def filter_moving_objects(frame: np.ndarray) -> np.ndarray:
    """
    تصفية الأجسام المتحركة
    الهدف: إبراز الأجسام المهمة وتقليل التشويش
    """
    # Threshold تلقائي باستخدام Otsu
    _, thresh = cv2.threshold(
        frame,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # عمليات مورفولوجية لتنظيف الصورة
    kernel = np.ones((3, 3), np.uint8)
    filtered_frame = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    return filtered_frame


def enhance_colors(frame: np.ndarray) -> np.ndarray:
    """
    تحسين الألوان والتباين
    الهدف: إبراز العناصر المهمة بصريًا
    """
    # التحويل إلى فضاء الألوان HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # فصل القنوات
    h, s, v = cv2.split(hsv)

    # تحسين قناة الإضاءة فقط
    v = cv2.equalizeHist(v)

    # دمج القنوات من جديد
    hsv_enhanced = cv2.merge((h, s, v))

    # العودة إلى BGR
    enhanced_frame = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced_frame
