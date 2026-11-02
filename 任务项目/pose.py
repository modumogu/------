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
    imagepoints=rect_corners(rect)
    