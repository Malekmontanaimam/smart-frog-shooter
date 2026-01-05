import mss
import numpy as np
import cv2
import time

from config import monitor, colors

from image_preprocessing import (
    enhance_colors,
    remove_visual_effects,
    filter_moving_objects
)

def run_screen_capture():
    with mss.mss() as sct:
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
            frame = enhance_colors(frame)
            clean_frame = remove_visual_effects(frame)
            processed_frame = filter_moving_objects(clean_frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    if area > 5:
                        (x, y), radius = cv2.minEnclosingCircle(cnt)

                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
                        cv2.putText(
                            frame,
                            color_name,
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1
                        )

            time.sleep(0.01)

            cv2.imshow("Zuma", frame)
            if cv2.waitKey(1) & 0xFF == ord('0'):
                break

    cv2.destroyAllWindows()
