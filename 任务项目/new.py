import numpy as np
import cv2 
from camera import CameraModel

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
base=CameraModel.from_file(r"任务项目\data\calibration.yaml")
#定义objp 形成左上角为【0，0，0】的棋盘格坐标系
#先建立一个6*9的空表
#再用循环把每个点的坐标写进去，单位mm
objp = np.zeros((6*9, 3), np.float32)
k = 0
for y in range(6):
    for x in range(9):
        objp[k] = (x * 25.0, y * 25.0, 0)
        k += 1

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(r"任务项目\data\calibration.mp4")
lost=0
yes=0
if not cap.isOpened():
    raise SystemExit("打不开视频")

while True:
    ok, img = cap.read()
    if not ok: 
             # 读到结尾（或出错）时 ok 为 False
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret==True :
        yes+=1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        #把点给写出来
        for k, pt in enumerate(corners2):
            x, y = int(pt[0]), int(pt[1])
            cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1) 
            cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
        lost+=1   
print("有效帧:", yes,"无效帧",lost)
#进行求解内参和畸变系数
rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)


print("RMS:", rms )
print("K:\n", K)
print("内参差异",K-base.K)
print("D:", D)
print("畸变差异",D-base.dist)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()