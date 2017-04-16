import utils
import matplotlib.image as mpimg
#utils.cameraCalibration()
image = mpimg.imread('./test_images/straight_lines2.jpg')
utils.cameraUndostort(image)
