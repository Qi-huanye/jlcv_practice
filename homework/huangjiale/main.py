# homework/name/green_ball_tracker.py
import cv2
import numpy as np
import os
from collections import deque
from pathlib import Path

# ============ 参数配置 ============
# 视频输入路径，如果是摄像头请填 0
VIDEO_PATH = "green_ball.mp4"  
# 输出保存路径：~/桌面/cao/output_video.mp4
DESKTOP_PATH = Path.home() / "Desktop" / "cao"
OUTPUT_PATH = str(DESKTOP_PATH / "output_video.mp4")

# HSV绿色范围（根据截图，绿球是高饱和度的翠绿色）
LOWER_GREEN = np.array([40, 80, 80])
UPPER_GREEN = np.array([80, 255, 255])

MIN_AREA = 200        # 绿球的最小面积（过滤噪点）
MAX_AREA = 20000      # 绿球的最大面积（过滤大圆柱体）
ARROW_SCALE = 0.3     # 箭头长度缩放比例
MAX_SPEED_HISTORY = 5 # 速度平滑帧数
# =================================

# 确保输出文件夹存在
os.makedirs(DESKTOP_PATH, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频 {VIDEO_PATH}")
    exit()

# 获取视频属性以保存输出视频
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 保存历史质心用于计算速度
pts = deque(maxlen=MAX_SPEED_HISTORY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 颜色提取
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # 形态学去噪
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 2. 寻找轮廓并过滤出“球”
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        # 按面积从大到小排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                # 计算外接圆和圆形度（判断是不是球）
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # 圆形度 = 4 * pi * 面积 / 周长^2，越接近1越像圆
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # 如果圆形度大于0.7，认为是球（排除圆柱体）
                if circularity > 0.7:
                    center = (int(x), int(y))
                    # 画出绿球的外接圆
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                    # 标记球心
                    cv2.circle(frame, center, 3, (255, 0, 0), -1)
                    break # 找到符合条件的球就停止

    # 3. 记录质心
    if center is not None:
        pts.appendleft(center)
    else:
        pts.appendleft(None)

    # 4. 计算速度并绘制箭头
    # 需要至少两个历史点，且都没有None
    valid_pts = [p for p in pts if p is not None]
    if len(valid_pts) >= 2:
        # 使用历史最远点和当前点计算平均速度，起到平滑作用
        p_old = valid_pts[-1]
        p_new = valid_pts[0]
        
        dx = p_new[0] - p_old[0]
        dy = p_new[1] - p_old[1]
        # 计算帧间隔
        frame_gap = len(valid_pts) - 1
        
        if frame_gap > 0:
            vx = dx / frame_gap
            vy = dy / frame_gap
            speed = np.sqrt(vx**2 + vy**2) # 像素/帧

            # 速度大于阈值才画箭头（避免静止时箭头乱飘）
            if speed > 1.5:
                # 箭头终点 = 球心 + 速度向量 * 缩放系数
                end_x = int(p_new[0] + vx * ARROW_SCALE * frame_gap)
                end_y = int(p_new[1] + vy * ARROW_SCALE * frame_gap)
                
                # 绘制箭头（红色）
                cv2.arrowedLine(frame, p_new, (end_x, end_y), (0, 0, 255), 3, tipLength=0.4)
                
                # 在球上方显示速度文字
                cv2.putText(frame, f"v: {speed:.1f} px/f", (p_new[0]-40, p_new[1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 显示画面（按 ESC 退出）
    cv2.imshow("Green Ball Tracking", frame)
    
    # 保存到视频文件
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"处理完成！视频已保存至：{OUTPUT_PATH}")
