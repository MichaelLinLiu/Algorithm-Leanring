# M: This program is a practice for solving border issues when convolve the image by using filters.

import cv2
import numpy as np


def mBorder(kernel_size,img):
    # M: create an empty uint8 type array
    # M: the size of padded image should be (kernel size - 1) / 2 + img width + (kernel size - 1) / 2
    paddedImage = np.zeros([kernel_size - 1 + img.shape[0], kernel_size - 1 + img.shape[1]], dtype=np.uint8)
    width = paddedImage.shape[1]
    height = paddedImage.shape[0]
    print(paddedImage.shape, 'height:',height,'width:', width, "pad:",int((kernel_size+1)/2))

    # M: pad the four corners of the padded image
    # M: corner1: top-left-corner
    for h in range(0, int((kernel_size+1)/2) ):
        for w in range(0, int((kernel_size+1)/2) ):
            paddedImage[h,w] = img[0][0]

    # M: corner2: top-right-corner
    for h in range(0, int((kernel_size+1)/2) ):
        for w in range(width - int((kernel_size+1)/2), width):
            paddedImage[h,w] = img[0][-1]

    # # M: corner3: down-left-corner
    for h in range( height - int((kernel_size+1)/2),  height):
        for w in range( 0, int((kernel_size+1)/2) ):
            paddedImage[h,w] = img[-1][0]

    # M: corner4: down-right-corner
    for h in range( height - int((kernel_size+1)/2), height  ):
        for w in range( width - int((kernel_size+1)/2), width ):
            paddedImage[h,w] = img[-1][-1]

    # M: border1: top border
    for h in range(0, int((kernel_size-1)/2) ):
        for w in range( int((kernel_size+1)/2), width - int((kernel_size+1)/2)):
            paddedImage[h,w] = img[0][w - int((kernel_size-1)/2)]

    # M: border2: down border
    for h in range(height - int((kernel_size-1)/2), height):
        for w in range( int((kernel_size+1)/2), width - int((kernel_size+1)/2)):
            paddedImage[h,w] = img[-1][w - int((kernel_size-1)/2)]

    # M: border3: left border
    for h in range( int((kernel_size+1)/2), height - int((kernel_size+1)/2)  ):
        for w in range(0, int((kernel_size-1)/2)):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][0]

    # M: border4: right border.
    for h in range( int((kernel_size+1)/2), height - int((kernel_size+1)/2)  ):
        for w in range(width -int((kernel_size-1)/2), width):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][-1]

    # M: internal
    for h in range( int((kernel_size-1)/2), height - int((kernel_size-1)/2)  ):
        for w in range(int((kernel_size-1)/2), width - int((kernel_size - 1) / 2)):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][w - int((kernel_size-1)/2)]

    return paddedImage


def mPadCropper(kernel_size, paddedImage):
    croppedImage = np.zeros([paddedImage.shape[0] - kernel_size + 1, paddedImage.shape[1] - kernel_size + 1], dtype=np.uint8)
    width = paddedImage.shape[1]
    height = paddedImage.shape[0]
    for h in range(int((kernel_size - 1) / 2), height - int((kernel_size - 1) / 2)):
        for w in range(int((kernel_size - 1) / 2), width - int((kernel_size - 1) / 2)):
            croppedImage[h - int((kernel_size - 1) / 2), w - int((kernel_size - 1) / 2)] = paddedImage[h][w]
    return croppedImage


# M: test the algorithm
fileName = '../Images_Contrast_Test/test2.png'
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
kernelSize1 = 53
paddedImage = mBorder(kernelSize1,img1)
# croppedImage = mPadCropper(kernelSize1,paddedImage)
cv2.imshow('test', paddedImage)
cv2.waitKey(0)
