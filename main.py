import threshold
import transform
import camera
import utils
import datafield as df

import line
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip


def process_image(OrgImage):
    global lP
    global Lane
    # 1. Camera calibration and 2. Distortion correction
    image = camera.undistort(OrgImage)
    # 3. Color/gradient threshold, combination binary image
    image = threshold.pipeline(image)
    # 4.Perspective transform warp image
    warpedImage = transform.warp(image)
    #5. Detect lane lines
    line.laneDetect(warpedImage)
    #6. Determine the lane curvature
    image = line.reslutImage(OrgImage,warpedImage)
    #7. OutPut the result Image

    return image

image = mpimg.imread('./test_images/test4.jpg')
res=process_image(image)
utils.imagePlot(image,res)
image = mpimg.imread('./test_images/test5.jpg')
res=process_image(image)
utils.imagePlot(image,res)

'''mp4_output = 'res.mp4'
clip1 = VideoFileClip("project_video.mp4")
res_clip = clip1.fl_image(process_image)
res_clip.write_videofile(mp4_output, audio=False)'''




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
