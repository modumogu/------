"""靶标检测：从画面中找到黑色方形靶标，返回包围它的旋转矩形。

本模块只负责“找到靶标、给出它的矩形框”，不涉及角点顺序、位姿与坐标变换。
"""

import cv2


def detect_target(frame, thresh=60, min_area=1500.0, max_area=40000.0):
    """在帧中检测贴放着的黑色方形靶标。

    输入：彩色（BGR）或灰度帧。
    输出：(rect, contour)
        rect    cv2.minAreaRect 的返回值 ((cx, cy), (w, h), angle)：
                中心与边长单位为像素，angle 为 OpenCV 的矩形角度约定；
        contour 靶标外轮廓，(N, 1, 2) int32 点集，可用于画图或自行细化角点。
    没有合格目标时返回 (None, None)。

    说明：靶标是最暗的大块区域，因此反向阈值后取外部轮廓，再按面积和
    “近似四边形”筛选；面积阈值按 640x480、靶标边长 80mm 的现场画面设定。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and area > best_area:
            best, best_area = contour, area
    if best is None:
        return None, None

    return cv2.minAreaRect(best), best
