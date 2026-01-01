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

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if area > 5: 
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    
                  
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
                    cv2.putText(frame, color_name, (int(x), int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        time.sleep(0.01)

        cv2.imshow("Zuma", frame)
        if cv2.waitKey(1) & 0xFF == ord('0'): break

cv2.destroyAllWindows()