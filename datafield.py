# this file contained the hyperparameter
import numpy as np
import transform
import line
import cv2


_leftLine = line.Line()
_rightLine = line.Line()

#camera paramter
pip_s_thresh  = (150, 255)
pip_sx_thresh = (20, 100)
imageSize = [720,1280]
#transform perspective parameters

xtr,ytr = 690,450   # x,y topRight
xbr,ybr = 1112,719  # x,y bottomRight
xbl,ybl = 223,719   # x,y bottomLeft
xtl,ytl = 596,450   # x,y topLeft

xtr_dst,ytr_dst = 960,0  # x,y topRight Destination
xbr_dst,ybr_dst = 960,720  # x,y bottomRight Destination
xbl_dst,ybl_dst = 320,720   # x,y bottomLeft Destination
xtl_dst,ytl_dst = 320,0   # x,y topLeft Destination

src = np.float32(
    [[xtr,ytr],
     [xbr,ybr],
     [xbl,ybl],
     [xtl,ytl]])
dst = np.float32(
    [[xtr_dst,ytr_dst],
     [xbr_dst,ybr_dst],
     [xbl_dst,ybl_dst],
     [xtl_dst,ytl_dst]])
M = cv2.getPerspectiveTransform(src,dst)# transfer Matrix
Minv = cv2.getPerspectiveTransform(dst,src)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/650 # meters per pixel in x dimension

simpleDetectMargin = 50

cnt =0
