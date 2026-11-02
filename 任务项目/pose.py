import cv2 
import numpy as np
from output import rect_corners
from camera import CameraModel , load_handeye

def position(rect,frame):
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
    camera=CameraModel.from_file(r"任务项目\data\calibration.yaml")
    K=camera.K
    D=camera.dist
    ok, rvec, tvec = cv2.solvePnP(objectpoints, imgpoints, K, D,useExtrinsicGuess=False, flags=cv2.SOLVEPNP_IPPE)
    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 100)
    return {
        "frame1": frame,
    }