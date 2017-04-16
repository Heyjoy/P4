# this file contained the hyperparameter
import numpy as np
gBlurKernelSize=  5
cannyLowThreshold= 50
cannyHighThreshold= 150
# region_of_interest , mask parameter
topLeft =(420,330)
topRight = (550,330)
bottomLeft =(137,539)
bottomRight = (920,539)
maskVertices = np.array([[topLeft,topRight,bottomRight,bottomLeft]],dtype = np.int32)

# houghLines parameter
hl_rho = 2
hl_theta = np.pi/180
hl_threshold = 1
hl_minLineLen = 15
hl_maxLineGap = 3

imgIndex = 0
