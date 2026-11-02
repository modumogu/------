"""结果可视化与输出：画检测到的靶标轮廓与矩形框，并打印每帧的检测结果。"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def rect_corners(rect):
    """由 ((cx, cy), (w, h), angle) 算出矩形四个角点（像素，int32）。

    做法：先把矩形看成以中心为原点的局部坐标，按 angle 旋转，再平移到图像坐标。
    返回顺序为 左上、右上、右下、左下（按局部坐标定义）。
    """
    (cx, cy), (w, h), angle = rect
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rotate = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2],
    ], dtype=np.float32)
    return (local @ rotate.T + [cx, cy]).astype(np.int32)


def draw_detection(frame, rect, contour):
    """在帧上画靶标轮廓、拟合出的矩形、中心点和矩形参数，返回新帧（不修改输入）。"""
    canvas = frame.copy()
    cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 1)
    cv2.polylines(canvas, [rect_corners(rect)], True, (0, 255, 255), 2)

    (cx, cy), (w, h), angle = rect
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1)
    text = (f"center=({cx:6.1f},{cy:6.1f}) "
            f"size=({w:5.1f}x{h:5.1f}) angle={angle:5.1f}")
    for color, thick in (((255, 255, 255), 3), ((20, 40, 20), 1)):
        cv2.putText(canvas, text, (12, 28), FONT, 0.6, color, thick, cv2.LINE_AA)
    return canvas


def format_detection_line(index, rect):
    """终端每帧一行的检测输出：帧号、矩形中心、边长与角度。"""
    (cx, cy), (w, h), angle = rect
    return (f"[{index:03d}] center_px=({cx:6.1f}, {cy:6.1f}) "
            f"size_px=({w:5.1f}, {h:5.1f}) angle={angle:6.1f} deg")
