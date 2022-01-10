# M: This program is a practise for 2D-Gaussian Convolution
import numpy as np
import cv2


def gaussian_formula(x, sigma):
    return np.exp(-0.5 * x / sigma ** 2)


def gaussian_kernel_2D(halfKernelWidth):
    kernel_size = halfKernelWidth * 2 + 1
    sigma = kernel_size / np.sqrt(8 * np.log(2))
    kernel_2D:np.array().astype("uint8") = []
    for x in range(-halfKernelWidth, halfKernelWidth+1):
        row = []
        for y in range(-halfKernelWidth,halfKernelWidth+1):
            weightSpatial = gaussian_formula(x**2+y**2,sigma)
            row.append(round(weightSpatial,4))
        kernel_2D.append(row)
    return kernel_2D


def convolution2DKernel(img,kernel_2D):
    totalWeights = 0
    for i in kernel_2D:
        for j in i:
            totalWeights = totalWeights + j

    halfWidth = int((len(kernel_2D) - 1) / 2)

    for h in range(halfWidth, img.shape[0] - halfWidth):
        for w in range(halfWidth, img.shape[1] - halfWidth):
            roi = img[h-halfWidth : h+halfWidth+1, w-halfWidth : w+halfWidth+1]
            img[h,w] = np.sum(kernel_2D * roi) / totalWeights

    return img


# M: test the algorithm
fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
halfKernelWidth = 1
kernel2D = gaussian_kernel_2D(halfKernelWidth)
adjustedImage1 = convolution2DKernel(img1, kernel2D)
cv2.imshow('mGaus ', adjustedImage1)
cv2.waitKey(0)