# Define a class to receive the characteristics of each line detection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datafield import *

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # transfer M
        self.M = None
        self.Minv = None
        self.warped = None

    #def radiusOfCurvature()
    def warp(self,img):
        img_size=(img.shape[1],img.shape[0])
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
        self.M = cv2.getPerspectiveTransform(src,dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src)
        self.warped = cv2.warpPerspective(img,self.M,img_size,flags=cv2.INTER_LINEAR)
        return self.warped
    def reslutImage(self,OrgImage,left_fit,right_fit):
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, 719, num=720)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (OrgImage.shape[1], OrgImage.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(OrgImage, 1, newwarp, 0.3, 0)
        return result
