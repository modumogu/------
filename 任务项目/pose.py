import cv2 
import numpy as np
from output import rect_corners
from camera import CameraModel , load_handeye

def position(rect,frame):
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
    imgpoints = (local @ rotate.T + [cx, cy]).astype(np.float32)
    camera=CameraModel.from_file(r"data\calibration.yaml")
    K=camera.K
    D=camera.dist
    ok, rvec, tvec = cv2.solvePnP(objectpoints, imgpoints, K, D,useExtrinsicGuess=False, flags=cv2.SOLVEPNP_IPPE)