import cv2 
import numpy as np
from output import rect_corners
from camera import CameraModel , load_handeye

def position(rect,frame):
    """
    1.标定objp和imgp
    2.求解位姿
    3.画坐标系
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    objectpoints=np.array([
    [-40.0,40.0,0], 
    [40.0,40.0,0],
    [40.0,-40.0,0],
    [-40.0,-40.0,0],
    ],np.float32)
    (cx, cy), (w, h), angle = rect
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rotate = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2],
    ], dtype=np.float32)
    imgp_demo = (local @ rotate.T + [cx, cy]).astype(np.float32)
    imgpoints = cv2.cornerSubPix(gray, imgp_demo, (11, 11), (-1, -1), criteria)
    #这里是用于把最小的x+y那个点放在第一个位置，保证顺时针顺序，解决了坐标轴变向的问题
    num = int(np.argmin(imgpoints[:, 0] + imgpoints[:, 1]))
    imgpoints = np.roll(imgpoints, -num, axis=0)#把最小的按顺时针往前移到第一保证0123都为顺时针
    camera=CameraModel.from_file(r"任务项目\data\calibration.yaml")
    K=camera.K
    D=camera.dist
    ok, rvec, tvec = cv2.solvePnP(objectpoints, imgpoints, K, D,useExtrinsicGuess=False, flags=cv2.SOLVEPNP_IPPE)
    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 100)

    T_base_cam=load_handeye(r"任务项目\data\handeye.yaml")
    #AI帮我拼的矩阵
    R_ct, _ = cv2.Rodrigues(rvec)      # 轴角 -> 3×3 旋转
    T_cam_target = np.eye(4)
    T_cam_target[:3, :3] = R_ct        # 左上 3×3：旋转
    T_cam_target[:3, 3] = tvec.ravel() # 右上一列：平移
    # 最后一行保持 [0, 0, 0, 1]
    #开始变换参考系
    T_base_target = T_base_cam @ T_cam_target
    base = (T_base_cam @ np.append(tvec.ravel(), 1.0))[:3]
    R = T_base_cam[:3, :3] @ cv2.Rodrigues(rvec)[0]
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    proj, _ = cv2.projectPoints(objectpoints, rvec, tvec, K, D)
    diff = imgpoints - proj.reshape(-1, 2)
    reproj = np.linalg.norm(diff, axis=1).mean()
    return {
        "frame1": frame,
        "T_base_target": T_base_target,
        "base_mm": base.ravel().tolist(),
        "tvec_cam_mm": tvec.ravel().tolist(),
        "yaw_deg": float(yaw),
        "reproj_px": float(reproj)  
    }