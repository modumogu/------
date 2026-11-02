# 抓取定位模块（交接版）

## 这是什么

工位相机固定在机械臂基座附近，俯视桌面。程序读取一段测试视频，在画面中找到贴放在
桌面上的黑色方形靶标，给出它的旋转矩形，把检测结果画在画面上并写入 `results.json`。

这个项目是上一届交接下来的：识别部分已经写好，但交接代码在当前环境下**跑不起来**
（见 `故障工单.md` 的 A-01）；**靶标位姿（相机系位置、机械臂基坐标、偏航角、
重投影误差）也还没有接入**，需要补上。三处已知的工程问题见 `故障工单.md`。

## 目录

```text
任务项目/
├── README.md            # 本文件
├── 故障工单.md           # 接手时确认存在的三条问题
├── requirements.txt
├── main.py              # 入口：读视频 -> 检测 -> 画图 -> 写视频与 JSON
├── camera.py            # 读内参、畸变、手眼矩阵；提供去畸变
├── detector.py          # 靶标检测：由帧得到旋转矩形与轮廓
├── output.py            # 画检测结果、终端输出格式
├── data/
│   ├── pose_test.mp4    # 测试视频，640x480，30fps，240 帧
│   ├── calibration.mp4  # 标定板视频（加分项用）
│   ├── calibration.yaml # 相机内参 + 畸变系数
│   ├── handeye.yaml     # 相机到基座的手眼矩阵 T_base_cam
│   ├── target_spec.md   # 靶标与坐标系定义、单位约定
│   └── frames/          # 几张静帧，便于单帧调试
└── expected/
    ├── 终端输出样例.txt  # 两帧的参考位姿输出
    ├── 关键帧示意.png    # 完成后的画面示意
    └── 边缘帧示意.png    # 靶标在画面边缘时的正确结果
```

## 环境

```text
pip install -r requirements.txt
```

需要 Python 3.9 以上、OpenCV 4.x。默认用 CPU，不需要 GPU。

## 数据读取

`camera.py` 已经把两份配置读好了，直接调用即可：

```python
from camera import CameraModel, load_handeye

camera = CameraModel.from_file("data/calibration.yaml")   # K、dist、图像尺寸
T_base_cam = load_handeye("data/handeye.yaml")            # 相机坐标系 -> 基座，4x4，毫米
```

这两份文件是 OpenCV 写出的 YAML。读取时**不要用 `cv2.FileStorage`**：它打不开含中文的
路径，读失败时内参会静默变成 `nan`，直到后面调用 OpenCV 才报错，非常难查。

## 运行

```text
python main.py
```

默认读取 `data/pose_test.mp4`，写 `output.avi` 和 `results.json`。

常用参数：

```text
python main.py --video data/pose_test.mp4 --out output.avi
python main.py --no-video          # 只跑检测，不写视频
```

单帧调试可直接用 `data/frames/` 里的静帧写一个小脚本调用 `detector.py`。

## 输出格式

终端每帧一行（当前只输出检测结果）：

```text
[000] center_px=( 320.5, 151.4) size_px=( 96.1,  95.2) angle= 88.5 deg
```

`results.json` 的字段：

```text
frames[]  frame           帧号
          rect_center_px  矩形中心（像素）
          rect_size_px    矩形两条边长（像素）
          rect_angle_deg  矩形角度（OpenCV minAreaRect 约定）
          tvec_cam_mm     靶标中心在相机坐标系下的位置（毫米）——待补
          base_mm         靶标中心（抓取点）在基坐标系下的位置（毫米）——待补
          yaw_deg         靶标绕基坐标系 z 轴的偏航角（度）——待补
          reproj_px       重投影误差（像素）——待补
lost      整段视频中丢失靶标的帧数
```

坐标系、单位和靶标尺寸的定义见 `data/target_spec.md`；位姿部分的输出要求见任务书。
