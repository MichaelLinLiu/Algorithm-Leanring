# M: the most naive solution of Bilateral Filter
import numpy as np
import cv2
import time


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


def convolution2DKernel(img, kernel_2D, sigmaRange = 10):
    # M: create a kernel to store total weight which is spatial weight combined with range weight.
    weightCombinedKernel = np.zeros((len(kernel_2D), len(kernel_2D)))

    # M: set the size for the convenient sake
    halfWidth = int((len(kernel_2D) - 1) / 2)

    # M: loop through the image(not deal with the border yet)
    for h in range(0, img.shape[0] - halfWidth - halfWidth):
        for w in range(0, img.shape[1] - halfWidth - halfWidth):
            # M: roi is the area to be convolved
            roi = img[h: h+halfWidth+halfWidth+1, w: w+halfWidth+halfWidth+1]

            # M: accumulate the combined weight for normalisation
            weightAccumulated = 0

            # M: loop the roi to calculate range weight respectively.
            # M: combine each spatial weight and range weight as the combined weight
            for i in range(0, len(roi)):
                for j in range(0, len(roi[0])):
                    p = float(roi[0][0])
                    q = float(roi[i][j])
                    weightRange = gaussian_formula((p - q) ** 2, sigmaRange)
                    weightSpatial = kernel_2D[i][j]
                    weightCombined = weightSpatial * weightRange
                    weightCombinedKernel[i][j] = weightCombined
                    weightAccumulated = weightAccumulated + weightCombined

            img[h, w] = np.sum(weightCombinedKernel * roi) / weightAccumulated

    return img


# M: test the algorithm
startTime = time.perf_counter()
fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
halfKernelWidth = 2
kernel2D = gaussian_kernel_2D(halfKernelWidth)
adjustedImage1 = convolution2DKernel(img1, kernel2D)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('mBil vs original, Std:{}'.format(halfKernelWidth), np.hstack((adjustedImage1, img)))
cv2.waitKey(0)