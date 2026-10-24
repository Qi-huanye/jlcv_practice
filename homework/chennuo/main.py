import csv
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "input", "homework.mp4")
OUT_DIR = os.path.join(HERE, "out")
OUT = os.path.join(OUT_DIR, "green_ball_gui.mp4")
WIN_TITLE = "Green Ball Tracking"   # 窗口标题
WIN_SIZE = (960, 540)               # 窗口初始大小（原片 1280x720，缩一半更好摆）

# ---------------- 参数 ----------------
GREEN_LO = np.array((35, 70, 35), np.uint8)     # 绿色 HSV 下限
GREEN_HI = np.array((95, 255, 255), np.uint8)   # 绿色 HSV 上限

ARROW_GAIN = 0.36        # 箭头长度 = 速度(px/s) × 0.36
ARROW_MAX = 300.0        # 箭头最大长度
ARROW_MIN_SPEED = 18.0   # 低于此速度视为静止，不画箭头

HUD_PANEL = (24, 20, 420, 246)
CHART_PANEL = (846, 500, 1258, 704)
CHART_PLOT = (884, 548, 1232, 668)


# ---------------- 颜色（BGR，直接用 OpenCV 顺序） ----------------
def rgb(r, g, b):
    return (b, g, r)


WHITE = rgb(245, 245, 245)
GREY = rgb(190, 190, 190)
DIM = rgb(150, 150, 150)
ACCENT = rgb(255, 214, 64)       # 箭头与曲线
ACCEL = rgb(255, 122, 74)        # 加速
DECEL = rgb(96, 196, 255)        # 减速
STEADY = rgb(210, 210, 210)      # 匀速
PANEL_BG = rgb(16, 18, 20)
PANEL_BORDER = rgb(120, 124, 130)


# ===========================================================================
# 窗口预览：需要带 GUI 后端的 OpenCV（桌面环境）
# ===========================================================================
def init_window():
    """创建预览窗口；OpenCV 没有 GUI 后端时给出明确提示。"""
    build = cv2.getBuildInformation()
    gui = next((ln for ln in build.splitlines() if ln.strip().startswith("GUI")), "")
    if "NONE" in gui:                    # headless 版没有窗口能力
        raise SystemExit(
            "当前 OpenCV 是 headless 版本（GUI: NONE），无法显示窗口。\n"
            "请安装带 GUI 的版本后再运行：\n"
            "    pip uninstall -y opencv-python-headless\n"
            "    pip install opencv-python"
        )
    try:
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, *WIN_SIZE)
    except cv2.error as exc:
        raise SystemExit(f"无法创建窗口（桌面环境/DISPLAY 不可用？）：{exc}")


# ===========================================================================
# 一、绿球检测：HSV 分割 + 形状过滤（区分同色绿圆柱）
# ===========================================================================
class Candidate:
    """一个绿色连通域。"""

    def __init__(self, cx, cy, area, w, h, fill, circularity):
        self.cx, self.cy = cx, cy
        self.area, self.w, self.h = area, w, h
        self.fill, self.circularity = fill, circularity

    @property
    def radius(self):
        return 0.25 * (self.w + self.h)

    @property
    def aspect(self):
        return self.w / max(self.h, 1)


def find_candidates(frame):
    """返回画面中所有绿色连通域（球和圆柱都在里面）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0.0
        fill = area / float(w * h)
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append(Candidate(m["m10"] / m["m00"], m["m01"] / m["m00"],
                             area, w, h, fill, circ))
    out.sort(key=lambda c: c.area, reverse=True)
    return out


def shape_penalty(c):
    """形状代价：球≈0，竖长的绿圆柱很大。"""
    p = 0.0
    p += 260.0 * max(0.0, 0.80 - c.aspect)
    p += 260.0 * max(0.0, c.aspect - 1.25)
    p += 700.0 * max(0.0, 0.74 - c.fill)
    p += 450.0 * max(0.0, 0.66 - c.circularity)
    return p


# ===========================================================================
# 二、跟踪：常速度预测 + 距离门限
# ===========================================================================
class BallTracker:
    def __init__(self):
        self.prev = None
        self.velocity = np.zeros(2)
        self.history = []      # 每帧 (cx, cy, r, detected)

    def update(self, cands):
        chosen = None
        if cands:
            if self.prev is None:
                pool = [c for c in cands
                        if 0.70 <= c.aspect <= 1.35 and c.fill >= 0.70 and c.area >= 6000]
                chosen = max(pool or cands, key=lambda c: c.circularity + 0.4 * c.fill)
            else:
                pred = np.array((self.prev.cx, self.prev.cy)) + self.velocity
                gate = max(90.0, 3.0 * self.prev.radius)
                best, best_cost = None, float("inf")
                for c in cands:
                    dist = float(np.hypot(c.cx - pred[0], c.cy - pred[1]))
                    cost = dist + shape_penalty(c) + 3.0 * max(0.0, dist - 40.0)
                    if dist > gate:
                        cost += 5000.0
                    if cost < best_cost:
                        best, best_cost = c, cost
                chosen = best

        if chosen is not None:
            if self.prev is not None:
                step = np.array((chosen.cx - self.prev.cx, chosen.cy - self.prev.cy))
                self.velocity = 0.6 * self.velocity + 0.4 * step
            self.prev = chosen
            self.history.append((chosen.cx, chosen.cy, chosen.radius, True))
        elif self.prev is not None:
            cx = self.prev.cx + self.velocity[0]
            cy = self.prev.cy + self.velocity[1]
            self.history.append((cx, cy, self.prev.radius, False))
        else:
            self.history.append((np.nan, np.nan, np.nan, False))


# ===========================================================================
# 三、轨迹平滑与运动学：局部二次拟合 + 中心差分
# ===========================================================================
def _quad_fit(values, lo, hi, i):
    """最小二乘二次拟合，返回 (i 处的值, i 处的斜率)。"""
    t = np.arange(lo, hi, dtype=np.float64) - i
    if hi - lo < 5:
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, values[lo:hi], rcond=None)
        return coef[0], coef[1]
    A = np.vstack([np.ones_like(t), t, t * t]).T
    coef, *_ = np.linalg.lstsq(A, values[lo:hi], rcond=None)
    return coef[0], coef[1]


def smooth_trajectory(states, fps, half_window=4):
    """对轨迹做局部二次平滑，返回 x, y, r, vx, vy（v 单位 px/s）。"""
    n = len(states)
    x = np.array([s[0] for s in states], float)
    y = np.array([s[1] for s in states], float)
    r = np.array([s[2] for s in states], float)
    for arr in (x, y, r):                      # 漏检帧线性插值
        ok = np.isfinite(arr)
        if not ok.all():
            idx = np.arange(n)
            arr[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])

    xs, ys, rs = np.empty(n), np.empty(n), np.empty(n)
    vx, vy = np.empty(n), np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        xs[i], dx = _quad_fit(x, lo, hi, i)
        ys[i], dy = _quad_fit(y, lo, hi, i)
        rs[i], _ = _quad_fit(r, lo, hi, i)
        vx[i], vy[i] = dx * fps, dy * fps

    if n >= 5:                                  # 速度轻微平滑
        k = np.array((1, 2, 3, 2, 1), float)
        k /= k.sum()
        vx = np.convolve(np.pad(vx, 2, mode="edge"), k, "valid")
        vy = np.convolve(np.pad(vy, 2, mode="edge"), k, "valid")
    return xs, ys, rs, vx, vy


# ===========================================================================
# 四、绘制：球标记、速度箭头、HUD、速度曲线（纯 OpenCV，无中文）
# ===========================================================================
def put_text(img, text, xy, scale=0.6, color=WHITE, thickness=2):
    """带黑色描边的文字（英文/数字），任何背景都清楚。"""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def put_text_right(img, text, x, y, scale, color):
    """右对齐文字（给坐标轴刻度用）。"""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    put_text(img, text, (x - tw, y), scale, color, 1)


def panel(img, box, alpha=0.62):
    x0, y0, x1, y1 = box
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), PANEL_BORDER, 1, cv2.LINE_AA)


def draw_arrow(img, start, direction, length, color=ACCENT, width=9):
    d = np.asarray(direction, float)
    d /= max(np.hypot(*d), 1e-6)
    tip = np.asarray(start, float) + d * length
    head_len = float(np.clip(0.32 * length, 16.0, 42.0))
    head_half = 0.62 * head_len
    base = tip - d * head_len
    perp = np.array((-d[1], d[0]))
    a = tuple(np.round(start).astype(int))
    b = tuple(np.round(base).astype(int))
    head = np.round(np.array([tip, base + perp * head_half,
                              base - perp * head_half])).astype(np.int32)
    cv2.line(img, a, b, (10, 10, 10), width + 6, cv2.LINE_AA)
    cv2.fillPoly(img, [head], (10, 10, 10), cv2.LINE_AA)
    cv2.line(img, a, b, color, width, cv2.LINE_AA)
    cv2.fillPoly(img, [head], color, cv2.LINE_AA)


def draw_ball_marker(img, cx, cy, r):
    c = (int(round(cx)), int(round(cy)))
    cv2.circle(img, c, int(round(r + 3)), (15, 15, 15), 6, cv2.LINE_AA)
    cv2.circle(img, c, int(round(r + 3)), WHITE, 3, cv2.LINE_AA)
    cv2.line(img, (c[0] - 9, c[1]), (c[0] + 9, c[1]), WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - 9), (c[0], c[1] + 9), WHITE, 2, cv2.LINE_AA)


def draw_chart(img, speeds, times, idx, ymax):
    x0, y0, x1, y1 = CHART_PLOT
    for frac in (0.0, 0.5, 1.0):
        y = int(y1 - frac * (y1 - y0))
        cv2.line(img, (x0, y), (x1, y), (70, 74, 80), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 94, 100), 1, cv2.LINE_AA)

    n = int(min(idx, len(speeds) - 1))
    t_span = max(times[-1], 1e-6)

    def to_px(i):
        x = x0 + (times[i] / t_span) * (x1 - x0)
        y = y1 - (min(speeds[i], ymax) / ymax) * (y1 - y0)
        return int(round(x)), int(round(y))

    step = max(1, n // 240)
    pts = [to_px(i) for i in range(0, n + 1, step)]
    area = pts + [(pts[-1][0], y1), (pts[0][0], y1)]
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(area, np.int32)], rgb(64, 76, 46))
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, ACCENT, 2, cv2.LINE_AA)

    cx, cy = to_px(n)
    cv2.line(img, (cx, y0), (cx, y1), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, (20, 20, 20), 1, cv2.LINE_AA)


def direction_name(vx, vy):
    names = ("right", "down-right", "down", "down-left",
             "left", "up-left", "up", "up-right")
    ang = np.degrees(np.arctan2(vy, vx))
    return names[int(round(ang / 45.0)) % 8]


def trend_of(speed, accel, eps=60.0):
    if speed < ARROW_MIN_SPEED:
        return "stationary", STEADY
    if accel > eps:
        return "accelerating", ACCEL
    if accel < -eps:
        return "decelerating", DECEL
    return "steady", STEADY


def render(frame, idx, cx, cy, r, vx, vy, speed, accel, speeds, times, ymax, fps):
    img = frame.copy()
    panel(img, HUD_PANEL)
    panel(img, CHART_PANEL)
    draw_chart(img, speeds, times, idx, ymax)

    # ---- 箭头：方向 = 速度方向，长度 ∝ 速度 ----
    if np.isfinite(cx) and speed >= ARROW_MIN_SPEED:
        draw_ball_marker(img, cx, cy, r)
        d = np.array((vx, vy), float) / speed
        length = min(ARROW_GAIN * speed, ARROW_MAX)
        p0 = np.array((cx + d[0] * (r + 14), cy + d[1] * (r + 14)))
        h, w = img.shape[:2]
        for axis, size in ((0, w), (1, h)):        # 贴边时缩短，保证箭头完整
            if d[axis] > 1e-6:
                length = min(length, (size - 8 - p0[axis]) / d[axis])
            elif d[axis] < -1e-6:
                length = min(length, (8 - p0[axis]) / d[axis])
        if length >= 22:
            draw_arrow(img, p0, d, length)
    elif np.isfinite(cx):
        draw_ball_marker(img, cx, cy, r)

    # ---- 文字 HUD ----
    trend, tcolor = trend_of(speed, accel)
    direction = direction_name(vx, vy) if speed >= ARROW_MIN_SPEED else "-"
    accel_txt = f"{accel:+5.0f} px/s^2" if speed >= ARROW_MIN_SPEED else "    -   "

    put_text(img, "GREEN BALL TRACKING", (44, 52), 0.85)
    put_text(img, f"frame {idx:3d} / {len(speeds) - 1:3d}    {idx / fps:5.2f} s",
             (44, 92), 0.62, GREY)
    put_text(img, f"speed  {speed:4.0f} px/s", (44, 126), 0.62)
    put_text(img, f"accel {accel_txt}", (44, 158), 0.62, tcolor)
    put_text(img, f"dir {direction:<11s} {trend}", (44, 190), 0.58)
    put_text(img, "arrow dir = motion", (44, 214), 0.52, WHITE, 1)
    put_text(img, "arrow len = speed", (44, 236), 0.52, WHITE, 1)

    # ---- 曲线图文字 ----
    x0, y0, x1, y1 = CHART_PLOT
    put_text(img, "speed vs time (px/s)", (866, 528), 0.58)
    put_text_right(img, f"{int(ymax)}", x0 - 8, y0 + 6, 0.42, DIM)
    put_text_right(img, f"{int(ymax / 2)}", x0 - 8, (y0 + y1) // 2 + 6, 0.42, DIM)
    put_text_right(img, "0", x0 - 8, y1 + 6, 0.42, DIM)
    put_text(img, "0 s", (x0, y1 + 22), 0.42, DIM, 1)
    put_text_right(img, f"{times[-1]:.0f} s", x1 + 8, y1 + 22, 0.42, DIM)
    return img


# ===========================================================================
# 五、主流程
# ===========================================================================
def main():
    if not os.path.exists(VIDEO):
        raise SystemExit(f"找不到视频：{VIDEO}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 第一遍：检测 + 跟踪
    tracker = BallTracker()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracker.update(find_candidates(frame))
    cap.release()

    states = tracker.history
    n = len(states)
    xs, ys, rs, vx, vy = smooth_trajectory(states, fps)
    speed = np.hypot(vx, vy)

    # 加速度：速度标量对时间的中心差分
    accel = np.zeros_like(speed)
    if n > 2:
        accel[1:-1] = (speed[2:] - speed[:-2]) * fps / 2.0
    accel = np.convolve(np.pad(accel, 1, mode="edge"), np.array((1, 2, 1)) / 4.0, "valid")
    accel[speed < ARROW_MIN_SPEED] = 0.0

    times = np.arange(n, dtype=np.float64) / fps
    ymax = max(400.0, np.ceil(speed.max() / 100.0) * 100.0)

    # 第二遍：画标注并写视频（cv2.VideoWriter，无需 ffmpeg）
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)   # 尽量减小压缩损失
    if not writer.isOpened():
        raise SystemExit("无法创建输出视频，请检查 OpenCV 是否支持 mp4v 编码")

    wait_ms = max(1, int(1000.0 / fps))    # 按帧率控制播放节奏
    init_window()                          # 创建预览窗口（没有 GUI 会直接报错）

    cap = cv2.VideoCapture(VIDEO)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = render(frame, i, xs[i], ys[i], rs[i], vx[i], vy[i],
                       speed[i], accel[i], speed, times, ymax, fps)
        writer.write(frame)
        if i == 150:
            cv2.imwrite(os.path.join(OUT_DIR, "gui_frame.png"), frame)
        i += 1
        cv2.imshow(WIN_TITLE, frame)       # 实时显示
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break                          # 按 q 提前退出
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 结果数据
    with open(os.path.join(OUT_DIR, "gui_speed.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "x", "y", "vx", "vy", "speed", "accel"])
        for k in range(n):
            wr.writerow([k, f"{times[k]:.4f}", f"{xs[k]:.2f}", f"{ys[k]:.2f}",
                         f"{vx[k]:.2f}", f"{vy[k]:.2f}", f"{speed[k]:.2f}",
                         f"{accel[k]:.2f}"])

    stats = {
        "frames": n, "fps": fps, "duration_s": round(n / fps, 3),
        "detected_frames": int(sum(1 for s in states if s[3])),
        "speed_px_s": {
            "mean": round(float(speed.mean()), 1),
            "median": round(float(np.median(speed)), 1),
            "p95": round(float(np.percentile(speed, 95)), 1),
            "max": round(float(speed.max()), 1),
            "max_frame": int(np.argmax(speed)),
        },
    }
    with open(os.path.join(OUT_DIR, "gui_stats.json"), "w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("标注视频：", OUT)
    print("抽帧预览：", os.path.join(OUT_DIR, "gui_frame.png"))


if __name__ == "__main__":
    main()
