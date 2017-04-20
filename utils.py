import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# for plot

def imagePlot(srcImage,reslutImage,srcColor=None,resColor ='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(srcImage,cmap=srcColor)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(reslutImage, cmap= resColor)
    ax2.set_title('Result Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def threeImagePlot(image1,image2,image3,color='gray'):
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(image1,cmap=color)
    ax1.set_title('image1', fontsize=50)
    ax2.imshow(image2,cmap=color)
    ax2.set_title('image2', fontsize=50)
    ax3.imshow(image3,cmap=color)
    ax3.set_title('image3', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
