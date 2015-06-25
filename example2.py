import pySLAM
import cv2
import numpy as np
from time import time,sleep
import os


pySLAM.init_threads()

K = np.float32(np.load('/home/pupil/slam_data/000/camera_matrix.npy'))
dist_coef = np.float32(np.load('/home/pupil/slam_data/000/dist_coefs.npy'))

# compute undistort transformation for debug view    
fieldcam_res = (640, 480)
newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(K, dist_coef, fieldcam_res, 0, fieldcam_res)
map1, map2 = cv2.initUndistortRectifyMap(K, dist_coef, np.identity(3), newCameraMatrix, fieldcam_res, cv2.CV_16SC2)

K = K.T

# cap = cv2.VideoCapture('/home/pupil/slam_data/000/world.mkv')
cap = cv2.VideoCapture(2)
# cap.set(3,640)
cap.set(4,480)
s,img = cap.read()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.remap(img, map1, map2,cv2.INTER_LINEAR) 

print img.shape

system = pySLAM.Slam_Context(img.shape[1],img.shape[0], K.flatten())
system.setVisualization()
system.init(img,0,0)

ts = time()
for x in range(1,2000):
    s,img = cap.read()
    if not s:
    	break
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.remap(img, map1, map2,cv2.INTER_LINEAR) 
    system.track_frame(img,x,x/30.,False)
 
system.finalize()
sleep(1)

print 'Done'