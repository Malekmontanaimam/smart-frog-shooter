import cv2
import numpy as np


def remove_visual_effects(frame: np.ndarray) -> np.ndarray:


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    clean_frame = cv2.GaussianBlur(gray, (5, 5), 0)

    return clean_frame


def filter_moving_objects(frame: np.ndarray) -> np.ndarray:

    _, thresh = cv2.threshold(
        frame,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    kernel = np.ones((3, 3), np.uint8)
    filtered_frame = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    return filtered_frame


def enhance_colors(frame: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    h, s, v = cv2.split(hsv)


    v = cv2.equalizeHist(v)


    hsv_enhanced = cv2.merge((h, s, v))


    enhanced_frame = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced_frame
