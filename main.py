from utils import *
from poly import *
from cameraUtils import *
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip


def process_image(image):
    global lP
    image = cameraUndistort(image) # Undistort the image
    image = pipeline(image) # combination binary image
    image = warp(image) # warp image
    a= lP.lanePolyfitPipeline(image)
    print(a)

    return image

lP=lanePloyfit()
image = mpimg.imread('./test_images/straight_lines1.jpg')
res=process_image(image)
#imagePlot(image,res)





'''#utils.cameraCalibration()
#image = mpimg.imread('./test_images/straight_lines2.jpg')
image = cv2.imread('./test_images/test6.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[:,:,::-1]
image = cameraUndostort(image)
#plt.imshow(image)
#plt.show()

image = pipeline(image,pip_s_thresh,pip_sx_thresh)
#plt.imshow(image,cmap='gray')
#plt.show()

binary_warped = warp(image)
#plt.imshow(binary_warped,cmap='gray')
#plt.show()
lp =lanePloyfit()
lp.lanePolyfitStart(binary_warped)
lp.lanePolyfitPipeline(binary_warped)
'''
