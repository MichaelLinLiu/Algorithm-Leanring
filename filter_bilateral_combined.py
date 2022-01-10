# M: This program is a practise to combine the exponential terms of Gaussian function
import numpy as np
import cv2
import time

# M: use lookup table to pre-store the range weight from 0 to 255
# lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2))

def convolution2DKernel(img, halfWidth, sigmaRange = 10):

    fullWidth = halfKernelWidth * 2 + 1

    sigmaSpatial = fullWidth / np.sqrt(8 * np.log(2))

    # M: create a kernel to store total weight which is spatial weight combined with range weight.
    weightCombinedKernel = np.zeros((fullWidth, fullWidth))

    # M: loop through the image(not deal with the border yet)
    for h in range(img.shape[0] - halfWidth - halfWidth):
        for w in range(img.shape[1] - halfWidth - halfWidth):
            weightAccumulated = 0
            roi = img[h: h+halfWidth+halfWidth+1, w: w+halfWidth+halfWidth+1]

            for i in range(len(roi)):
                for j in range(len(roi[0])):
                    p = float(roi[0][0])
                    q = float(roi[i][j])
                    vector1 = -0.5 * (p - q) ** 2 / sigmaRange ** 2
                    vector2 = -0.5 * ((i-halfWidth) ** 2 + (j-halfWidth) ** 2) / sigmaSpatial ** 2
                    weightCombined = np.exp(vector1 + vector2)
                    weightCombinedKernel[i][j] = weightCombined
                    weightAccumulated = weightAccumulated + weightCombined

            img[h, w] = np.sum(weightCombinedKernel * roi) / weightAccumulated

    return img


# M: test the algorithm
startTime = time.perf_counter()
fileName = "../Images_General_Test/no_flashed_image.jpg"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
halfKernelWidth = 4
adjustedImage1 = convolution2DKernel(img1, halfKernelWidth)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('mBil vs original, Std:{}'.format(halfKernelWidth), np.hstack((adjustedImage1, img)))
cv2.waitKey(0)