from utils import *
from poly import *
from cameraUtils import *
from line import *
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip


def process_image(OrgImage):
    global lP
    global Lane
    image = cameraUndistort(OrgImage) # Undistort the image
    image = pipeline(image) # combination binary image
    image = Lane.warp(image) # warp image
    lP.lanePolyfitPipeline(image)
    #print(lP.left_fit,lP.right_fit)
    image=Lane.reslutImage(OrgImage,lP.left_fit,lP.right_fit)
    return image

lP=LanePloyfit()
Lane = Line()
image = mpimg.imread('./test_images/test4.jpg')
res=process_image(image)
imagePlot(image,res)

mp4_output = 'res.mp4'
clip1 = VideoFileClip("project_video.mp4")
res_clip = clip1.fl_image(process_image)
res_clip.write_videofile(mp4_output, audio=False)




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
