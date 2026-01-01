import mss        
import numpy as np  
import cv2          
import time          



monitor = {"top": 250, "left": 100, "width": 850, "height": 650}


colors = {
    "Red":    ([0, 150, 100], [10, 255, 255]),
    "Green":  ([45, 150, 100], [75, 255, 255]), 
    "Blue":   ([90, 50, 50], [145, 255, 255]), 
    "Yellow": ([25, 150, 100], [35, 255, 255]),
    "Purple": ([135, 100, 100], [155, 255, 255])
}


ROI_X, ROI_Y = 395, 350 
ROI_W, ROI_H = 10, 10

with mss.mss() as sct:
  # print("-------")
    
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        frog_roi = hsv[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        current_ball = "None"
        found_colors = []


        for name, (low, high) in colors.items():
            mask_roi = cv2.inRange(frog_roi, np.array(low), np.array(high))
            pixel_count = cv2.countNonZero(mask_roi)
            
            if pixel_count > 30: 
                found_colors.append((name, pixel_count))


        if len(found_colors) > 1:
            for color_name, count in found_colors:
                if color_name != "Yellow": 
                    current_ball = color_name
                    break
            if current_ball == "None": current_ball = "Yellow"
        elif len(found_colors) == 1:
            current_ball = found_colors[0][0]


        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circularity = 4 * np.pi * (area / (peri * peri))
                

                if 400 < area < 1500 and circularity > 0.55:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
 
                    cv2.putText(frame, color_name, (int(x)-10, int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


        time.sleep(0.01)
        cv2.imshow("Zuma AI - Final Clean View", frame)
        if cv2.waitKey(1) & 0xFF == ord('0'): break

cv2.destroyAllWindows()