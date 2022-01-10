# M: This program is a practise to combine the exponential terms of Gaussian function
import numpy as np
import cv2
import time


def convolution2DKernel(flashed_image, no_flashed_image, halfWidth, sigmaRange = 10):
    result = flashed_image.copy()

    fullWidth = halfKernelWidth * 2 + 1

    sigmaSpatial = fullWidth / np.sqrt(8 * np.log(2))

    # M: create a kernel to store total weight which is spatial weight combined with range weight.
    weightCombinedKernel = np.zeros((fullWidth, fullWidth))

    lut_range = -0.5 * np.arange(256)**2 / sigmaRange**2

    # M: loop through the image(not deal with the border yet)
    for h in range(flashed_image.shape[0] - halfWidth - halfWidth):
        for w in range(flashed_image.shape[1] - halfWidth - halfWidth):
            weightAccumulated = 0
            roi = no_flashed_image[h: h + halfWidth + halfWidth + 1, w: w + halfWidth + halfWidth + 1]
            roi2 = flashed_image[h: h + halfWidth + halfWidth + 1, w: w + halfWidth + halfWidth + 1]
            for i in range(len(roi)):
                for j in range(len(roi[0])):
                    p = float(roi2[0][0])
                    q = float(roi2[i][j])
                    vector1 = lut_range[np.abs(p-q).astype(int)]
                    vector2 = -0.5 * ((i-halfWidth) ** 2 + (j-halfWidth) ** 2) / sigmaSpatial ** 2
                    weightCombined = np.exp(vector2+vector1)
                    weightCombinedKernel[i][j] = weightCombined
                    weightAccumulated = weightAccumulated + weightCombined

            result[h, w] = np.sum(weightCombinedKernel * roi) / weightAccumulated

    return result


fileName1 = "../Images_General_Test/no_flashed_image.jpg"
fileName2 = "../Images_General_Test/flashed_image.jpg"
img1 = cv2.imread(fileName1, 1)
img2 = cv2.imread(fileName2, 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
startTime = time.perf_counter()
halfKernelWidth = 3
adjustedImage1 = convolution2DKernel(img2, img1, halfKernelWidth)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('mBil vs noFlash vs flash, Std:{}'.format(halfKernelWidth), np.hstack((adjustedImage1, img1, img2)))
cv2.waitKey(0)