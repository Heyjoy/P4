# this file contained the hyperparameter
import numpy as np

xtr,ytr = 690,450   # x,y topRight
xbr,ybr = 1112,719  # x,y bottomRight
xbl,ybl = 223,719   # x,y bottomLeft
xtl,ytl = 596,450   # x,y topLeft

xtr_dst,ytr_dst = 960,0  # x,y topRight Destination
xbr_dst,ybr_dst = 960,720  # x,y bottomRight Destination
xbl_dst,ybl_dst = 320,720   # x,y bottomLeft Destination
xtl_dst,ytl_dst = 320,0   # x,y topLeft Destination

#camera paramter
ImgSize =(1280,720)

pip_s_thresh  = (150, 255)
pip_sx_thresh = (20, 100)
