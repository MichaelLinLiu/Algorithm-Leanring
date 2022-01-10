# M: This program is a practise for Gaussian filter based on the tutorial from
# http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python
# -from-scratch/
import cv2
import numpy as np
import time


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


# M: size is the kernel size
def gaussian_kernel_1D(size):
    # M: generate a linear space
    kernel_1D = np.linspace(-(size // 2), size // 2, size)

    # M: calculate the value of sigma
    sigma = size / np.sqrt(8 * np.log(2))

    # M: apply Gaussian formula to the the kernel
    for i in range(size):
        vector1 = 1 / (np.sqrt(2 * np.pi) * sigma)
        vector2 = np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))
        kernel_1D[i] = vector1 * vector2
    # print("kernel:",kernel_1D)
    return kernel_1D


# M: the convolution algorithm is the most naive one by using two 1-D Gaussian kernel filters which has O(m**2 * 2*n) complexity.
def convolution(img, kernel):
    # M: get the total weight of the kernel
    totalWeights = 0
    for i in kernel:
        totalWeights = totalWeights + i

    kernel_size = len(kernel)
    kernel_x = kernel
    kernel_y = np.transpose(kernel)

    # M: convolve image by using two 1-D Gaussian filters
    padW = int((kernel_size-1) / 2)
    for h in range(0, img.shape[0]):
        for w in range(0, img.shape[1] - kernel_size + 1):
            img[h, w + padW] = np.sum(kernel_x * img[h, w:w + kernel_size]) / totalWeights

    padH = int((kernel_size - 1) / 2)
    for w in range(0, img.shape[1]):
        for h in range(0, img.shape[0] - kernel_size + 1):
            img[h + padH, w] = np.sum(kernel_y * img[h: h + kernel_size, w]) / totalWeights

    return img


def mGaussianBlur(image, kernel_size):
    paddedImage = mBorder(kernel_size,image)
    kernel = gaussian_kernel_1D(kernel_size)
    convolvedImage = convolution(paddedImage, kernel)
    croppedImage = mPadCropper(kernel_size, convolvedImage)
    return croppedImage


# M: test the algorithm.
fileName = '../Images_Contrast_Test/test2.png'
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
size = 7
startTime = time.perf_counter()
adjustedImage1 = mGaussianBlur(img1, size)
# adjustedImage2 = cv2.GaussianBlur(img,(size,size),cv2.BORDER_ISOLATED)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('opencvGau',adjustedImage1)
cv2.waitKey(0)

# print('opencv', cv2.getGaussianKernel(size,0))
# cv2.imshow('mGau vs CV Gau, Std:{}'.format(size), np.hstack((adjustedImage1, adjustedImage2)))
# cv2.waitKey(0)

