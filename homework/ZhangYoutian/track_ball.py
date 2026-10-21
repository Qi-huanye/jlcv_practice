import cv2
import numpy as np

lower_green = np.array([40, 80, 80])
upper_green = np.array([75, 255, 255])

prev_x, prev_y = None, None
scale = 0.8

cap = cv2.VideoCapture("homework.mp4")
if not cap.isOpened():
    print("视频打开失败！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.75:
            (x, y), R = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            R = int(R)
            cv2.circle(frame, center, R, (0,255,0), 2)

            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                end_x = int(x + dx * scale)
                end_y = int(y + dy * scale)
                cv2.arrowedLine(frame, center, (end_x, end_y), (255,0,0), 2, tipLength=0.2)
            
            prev_x, prev_y = x, y

    cv2.imshow("result", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows(
