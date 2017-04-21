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


def process_image(image):
    orgImage = image.copy()
    # 1. Camera calibration and 2. Distortion correction
    image = camera.undistort(image)
    # 3. Color/gradient threshold, combination binary image
    image = threshold.pipeline(image)
    # 4.Perspective transform warp image
    warpedImage = transform.warp(image)
    # 5. Detect lane lines
    # 6. Determine the lane curvature
    line.laneDetect(warpedImage)
    # 7. OutPut the result Image
    image = line.reslutImage(orgImage,warpedImage)
    return image

'''image = mpimg.imread('./test_images/straight_lines1.jpg')
print(image.shape[0],image.shape[1])
res=process_image(image)
utils.imagePlot(image,res)
image = mpimg.imread('./test_images/test1.jpg')
res=process_image(image)
utils.imagePlot(image,res)
image = mpimg.imread('./test_images/test4.jpg')
res=process_image(image)
utils.imagePlot(image,res)'''

mp4_output = 'res.mp4'
clip1 = VideoFileClip("project_video.mp4")
res_clip = clip1.fl_image(process_image)
res_clip.write_videofile(mp4_output, audio=False)
print(df.cnt)
