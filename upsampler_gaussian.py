import numpy as np
import cv2


def zero_interpolator(image):
    scale_rate = 1
    height = image.shape[0]
    width = image.shape[1]
    result_height = height * 2
    result_width = width * 2
    result = np.zeros((result_height, result_width), dtype=np.uint8)
    for h in range(0, height, scale_rate):
        for w in range(0, width, scale_rate):
            result[h*2,w*2] = image[h,w]
    return result


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


def convolution2DKernel(img, kernel_2D):
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


fileName = "../Images_Contrast_Test/test6_1.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
up_image = zero_interpolator(img1)
halfKernelWidth = 1
kernel2D = gaussian_kernel_2D(halfKernelWidth)
adjustedImage1 = convolution2DKernel(up_image, kernel2D)
for i in range(adjustedImage1.shape[0]):
    for j in range(adjustedImage1.shape[1]) :
        if int(adjustedImage1[i, j]) * 4 > 255:
            adjustedImage1[i, j] = 255
        else:
            adjustedImage1[i,j] = adjustedImage1[i,j] * 4

adjustedImage1 = convolution2DKernel(adjustedImage1, kernel2D)
cv2.imshow('Gaussian Up-sampling', adjustedImage1)
cv2.imshow('ori',img1)
cv2.waitKey(0)

