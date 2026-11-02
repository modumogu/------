"""相机与配置文件读取：内参、畸变系数、手眼矩阵。

这里刻意没有用 cv2.FileStorage：它打不开含中文的路径，读失败时内参会静默变成
nan（不报错），等到调用 solvePnP 才抛一堆 OpenCV 内部断言，很难查。
这两份 YAML 是 OpenCV 写出来的，结构固定，直接按文本解析更稳，也方便项目放在
中文目录下运行。
"""

from pathlib import Path

import cv2
import numpy as np


def _read_lines(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def read_scalar(path, node):
    """读取 YAML 里的一个标量，例如 image_width: 640。"""
    for line in _read_lines(path):
        text = line.strip()
        if text.startswith(node + ":"):
            return float(text.split(":", 1)[1].strip())
    raise KeyError(f"{path} 里找不到 {node}")


def read_matrix(path, node):
    """读取 YAML 里名为 node 的矩阵块，返回形状为 (rows, cols) 的 float64 数组。

    块的样子（OpenCV FileStorage 写出来的格式）：

        node: !!opencv-matrix
           rows: 3
           cols: 3
           dt: d
           data: [ 1., 0., 0., ... ]
    """
    lines = _read_lines(path)
    for index, line in enumerate(lines):
        if not line.strip().startswith(node + ":"):
            continue
        rows = cols = None
        data_text = None
        for follow in lines[index + 1:]:
            text = follow.strip()
            if text.startswith("rows:"):
                rows = int(text.split(":", 1)[1])
            elif text.startswith("cols:"):
                cols = int(text.split(":", 1)[1])
            elif text.startswith("data:"):
                data_text = text.split("data:", 1)[1]
                if "]" in data_text:
                    break
            elif data_text is not None:
                data_text += text
                if "]" in text:
                    break
            elif text and not text.startswith(("dt:", "!!")):
                break  # 已经越过这一块
        if rows is None or cols is None or data_text is None:
            raise ValueError(f"{path} 里的 {node} 不是标准的 OpenCV 矩阵块")
        values = [float(v) for v in data_text.strip().strip("[]").split(",") if v.strip()]
        if len(values) != rows * cols:
            raise ValueError(f"{path} 里的 {node} 数据个数与 rows/cols 不符")
        return np.array(values, dtype=np.float64).reshape(rows, cols)
    raise KeyError(f"{path} 里找不到 {node}")


class CameraModel:
    """针孔相机 + 径向/切向畸变。内参单位与物体坐标单位保持一致（本任务统一用毫米）。"""

    def __init__(self, K, dist, width, height):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.width = int(width)
        self.height = int(height)

    @classmethod
    def from_file(cls, path):
        """从 calibration.yaml 读取内参、畸变系数与图像尺寸。"""
        K = read_matrix(path, "camera_matrix")
        dist = read_matrix(path, "dist_coeffs")
        width = int(read_scalar(path, "image_width"))
        height = int(read_scalar(path, "image_height"))
        return cls(K, dist, width, height)

    def undistort(self, frame):
        """返回去畸变后的图像（主流程不需要，供检查和加分项使用）。"""
        return cv2.undistort(frame, self.K, self.dist)


def load_handeye(path):
    """读取相机到机械臂基座的手眼矩阵 T_base_cam（4x4，平移单位毫米）。"""
    T = read_matrix(path, "T_base_cam")
    if T.shape != (4, 4):
        raise ValueError(f"{path} 里的 T_base_cam 不是 4x4")
    return T
