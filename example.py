import pySLAM
import cv2
import numpy as np
from time import time,sleep
import os


pySLAM.init_threads()
undistorter = pySLAM.Slam_Undistorter('/home/pupil/Downloads/LSD_room/cameraCalibration.cfg')
images = '/home/pupil/Downloads/LSD_room/images'
files = [os.path.join(images,f) for f in os.listdir(images)]
files.sort()

K = np.zeros((3,3), dtype=np.float32)
K[0,0] = 254.326950
K[1,1] = 375.934387
K[2,0] = 266.881897
K[2,1] = 231.099091
K[2,2] = 1.0

f = files.pop(0)
img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img = undistorter.undistort(img)

system = pySLAM.Slam_Context(img.shape[1],img.shape[0], K.flatten())
system.setVisualization()

system.init(img,0,0)
ts = time()
x = 0
for f in files:
    x = x+1

    img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = undistorter.undistort(img)
    system.track_frame(img,x,x/30.,False)
    if x == 900:
    	# system.unsetVisualization()
    	break

system.finalize()
sleep(1) #need this for the system to finish.
del system
print 'Done'

