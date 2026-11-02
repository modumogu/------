"""第 4 章抓取定位程序入口（交接版）：读视频 -> 检测靶标 -> 写视频与结果。

运行：
    python main.py
    python main.py --video data/pose_test.mp4 --out output.avi

本版本只保留识别与输出部分：靶标位姿（相机系位置、基坐标、偏航角、重投影误差）
尚未接入，results.json 中对应字段为 null。
"""

import argparse
import json
from pathlib import Path

import cv2

from camera import CameraModel, load_handeye
from detector import detect_target
from output import draw_detection, format_detection_line

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="靶标检测与抓取点输出")
    parser.add_argument("--video", default=str(DATA / "pose_test.mp4"), help="输入视频路径")
    parser.add_argument("--calib", default=str(DATA / "calibration.yaml"), help="相机内参文件")
    parser.add_argument("--handeye", default=str(DATA / "handeye.yaml"), help="手眼矩阵文件")
    parser.add_argument("--out", default=str(ROOT / "output.avi"), help="输出视频路径")
    parser.add_argument("--json", default=str(ROOT / "results.json"), help="逐帧结果 JSON")
    parser.add_argument("--no-video", action="store_true", help="不写输出视频，只跑检测")
    return parser.parse_args()


def main():
    args = parse_args()

    # 启动自检：先把标定与手眼文件读出来。位姿部分要用到它们，放在这里能尽早
    # 暴露路径或格式问题，而不是等到算位姿时才报错。
    camera = CameraModel.from_file(args.calib)
    T_base_cam = load_handeye(args.handeye)
    print(f"-- 内参已读取：{camera.width}x{camera.height} "
          f"fx={camera.K[0, 0]:.1f} fy={camera.K[1, 1]:.1f}；"
          f"手眼矩阵 T_base_cam {T_base_cam.shape} 已读取")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (height, width)
        )
        if not writer.isOpened():
            raise SystemExit(f"无法创建输出视频：{args.out}")

    records = []
    index = 0
    lost = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rect, contour = detect_target(frame)
        if rect is None:
            lost += 1
            canvas = frame.copy()
            cv2.putText(canvas, "target lost", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        else:
            canvas = draw_detection(frame, rect, contour)
            print(format_detection_line(index, rect))
            (cx, cy), (w, h), angle = rect
            records.append({
                "frame": index,
                "rect_center_px": [float(cx), float(cy)],
                "rect_size_px": [float(w), float(h)],
                "rect_angle_deg": float(angle),
                # 位姿相关的字段：接入位姿计算后填写，单位见 data/target_spec.md
                "tvec_cam_mm": None,
                "base_mm": None,
                "yaw_deg": None,
                "reproj_px": None,
            })
        if writer is not None:
            writer.write(canvas)
        index += 1

    cap.release()
    if writer is not None:
        writer.release()
    Path(args.json).write_text(json.dumps({"frames": records, "lost": lost}, indent=1))
    print(f"-- 共 {index} 帧，丢失 {lost} 帧，结果写入 {args.json}")


if __name__ == "__main__":
    main()
