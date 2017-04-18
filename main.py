from utils import *
from poly import *
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

#utils.cameraCalibration()
#image = mpimg.imread('./test_images/straight_lines2.jpg')
image = cv2.imread('./test_images/test6.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cameraUndostort(image)
#plt.imshow(image)
#plt.show()

image = pipeline(image,pip_s_thresh,pip_sx_thresh)
#plt.imshow(image,cmap='gray')
#plt.show()

binary_warped = warp(image)
plt.imshow(binary_warped,cmap='gray')
plt.show()
lanePloyfit.polyfit1(binary_warped)
lanePloyfit.polyfit2(binary_warped)
