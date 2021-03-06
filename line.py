# Define a class to receive the characteristics of each line detection
import numpy as np
import cv2
import matplotlib.pyplot as plt
import datafield as df
import transform
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
    def calCurvature(self):
        fit_cr = np.polyfit(self.ally*df.ym_per_pix , self.allx*df.xm_per_pix, 2)
        y_eval = np.max(self.ally)
        radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*df.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        #print ("{} m".format(self.radius_of_curvature))
        return radius_of_curvature
    def calDistance(self):
        line_base_pos = int(self.bestx -df.imageSize[1]/2)*df.xm_per_pix
        return line_base_pos
leftLine = Line()
rightLine = Line()

def santiyCheck(left_fit,right_fit):
    checkState = True
    # Checking that they have similar curvature
    leftLine.diffs = np.absolute(np.around((leftLine.best_fit - left_fit),1))
    rightLine.diffs = np.absolute(np.around((rightLine.best_fit - right_fit),1))
    if(leftLine.diffs[2] > 50 or rightLine.diffs[2] >50):
        checkState = False
    # Checking that they are separated by approximately the right distance horizontally
    # Checking that they are roughly parallel
    return checkState

def laneDetect(binary_warped):
    # Santiy Check
    # if pass do simple
    # if no pass, Santiy Check skip this frame and counting up
    if(leftLine.detected and rightLine.detected):
        #print("simple Detect process")
        simpleDetect(binary_warped)
    else:
        #print("was no detected lines before, do histogram detect method")
        histogramDetect(binary_warped)


    leftLine.radius_of_curvature = leftLine.calCurvature()
    rightLine.radius_of_curvature = rightLine.calCurvature()

# in lane Detect followed Lane paramters update.
# Line.allx,Line.ally,Line.current_fit, Line.detected,best_fit,fitx
def simpleDetect(binary_warped):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = df.simpleDetectMargin

    left_fit = leftLine.best_fit # read prev. value
    right_fit = rightLine.best_fit  # read prev. value
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # santiyCheck, only update the result when pass the check
    if(santiyCheck(left_fit,right_fit)):
        leftLine.allx = nonzerox[left_lane_inds]
        leftLine.ally = nonzeroy[left_lane_inds]
        rightLine.allx = nonzerox[right_lane_inds]
        rightLine.ally = nonzeroy[right_lane_inds]
        leftLine.current_fit = left_fit
        rightLine.current_fit = right_fit
        leftLine.best_fit = (leftLine.best_fit + left_fit)/2
        rightLine.best_fit = (rightLine.best_fit+ right_fit)/2

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        leftLine.recent_xfitted = leftLine.current_fit[0]*ploty**2 + leftLine.current_fit[1]*ploty + leftLine.current_fit[2]
        rightLine.recent_xfitted = rightLine.current_fit[0]*ploty**2 + rightLine.current_fit[1]*ploty + rightLine.current_fit[2]
        leftLine.bestx = np.mean(leftLine.recent_xfitted)
        rightLine.bestx = np.mean(rightLine.recent_xfitted)

    #else: # not save the caculation reslut this time. do noting.
        #leftLine.detected = False
        #rightLine.detected = False
        # return


def histogramDetect(binary_warped):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 4
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftLine.allx = nonzerox[left_lane_inds]
    leftLine.ally = nonzeroy[left_lane_inds]
    rightLine.allx = nonzerox[right_lane_inds]
    rightLine.ally = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    leftLine.current_fit = np.polyfit(leftLine.ally, leftLine.allx, 2)
    rightLine.current_fit = np.polyfit(rightLine.ally, rightLine.allx, 2)
    # update best_fit
    leftLine.best_fit = leftLine.current_fit
    rightLine.best_fit = rightLine.current_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftLine.recent_xfitted = leftLine.current_fit[0]*ploty**2 + leftLine.current_fit[1]*ploty + leftLine.current_fit[2]
    rightLine.recent_xfitted = rightLine.current_fit[0]*ploty**2 + rightLine.current_fit[1]*ploty + rightLine.current_fit[2]
    leftLine.bestx = np.mean(leftLine.recent_xfitted)
    rightLine.bestx = np.mean(rightLine.recent_xfitted)

    #print(leftLine.bestx)

    leftLine.detected = True
    rightLine.detected = True


def reslutImage(OrgImage,warpedImage):
    warp_zero = np.zeros_like(warpedImage).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, 719, num=720)
    left_fit = leftLine.current_fit
    right_fit = rightLine.current_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    #newwarp = cv2.warpPerspective(color_warp, df.Minv, (OrgImage.shape[1], OrgImage.shape[0]))
    newwarp = transform.unwarp(color_warp,OrgImage)

    # Combine the result with the original image
    result = cv2.addWeighted(OrgImage, 1, newwarp, 0.3, 0)
    center_of_lane = int((leftLine.bestx + rightLine.bestx)/2)
    center_of_vechical = int(OrgImage.shape[1]/2)
    avgRadiusCurvature = (leftLine.radius_of_curvature +rightLine.radius_of_curvature)/2
    offset = leftLine.calDistance()+ rightLine.calDistance()
    cv2.putText(result,'Radius of curvature is {0:.2f} m'.format(avgRadiusCurvature),(20, 50),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),thickness=2)
    cv2.line(result,(center_of_lane,650),(center_of_lane,719), (255,0,0),2)
    cv2.line(result,(center_of_vechical,600),(center_of_vechical,719), (0,0,255),2)
    cv2.putText(result,'offset to center {0:.2f} m'.format(offset),(20, 80),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),thickness=2)
    cv2.putText(result,'left: {}'.format(leftLine.diffs),(20, 110),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),thickness=2)
    cv2.putText(result,'right:{}'.format(rightLine.diffs),(20, 145),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),thickness=2)
    if (leftLine.diffs[2] > 50 or rightLine.diffs[2] >50):
        cv2.putText(result,'look!',(20, 175),cv2.FONT_HERSHEY_DUPLEX,1,(255, 0, 0),thickness=2)
        #df.cnt = df.cnt +1

    return result
