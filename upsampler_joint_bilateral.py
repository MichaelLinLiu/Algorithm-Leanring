# M: This program is a practise to combine the exponential terms of Gaussian function
import numpy as np
import cv2
import time


def convolution2DKernel(guidance_image, upsampled_lowRes_image, halfWidth, sigmaRange = 10):
    result = guidance_image.copy()

    fullWidth = halfKernelWidth * 2 + 1

    sigmaSpatial = fullWidth / np.sqrt(8 * np.log(2))

    # M: create a kernel to store total weight which is spatial weight combined with range weight.
    weightCombinedKernel = np.zeros((fullWidth, fullWidth))

    lut_range = -0.5 * np.arange(256)**2 / sigmaRange**2

    # M: loop through the image(not deal with the border yet)
    for h in range(guidance_image.shape[0] - halfWidth - halfWidth):
        for w in range(guidance_image.shape[1] - halfWidth - halfWidth):
            weightAccumulated = 0
            roi = upsampled_lowRes_image[h: h + halfWidth + halfWidth + 1, w: w + halfWidth + halfWidth + 1]
            roi2 = guidance_image[h: h + halfWidth + halfWidth + 1, w: w + halfWidth + halfWidth + 1]
            for i in range(len(roi)):
                for j in range(len(roi[0])):
                    p = float(roi2[0][0])
                    q = float(roi2[i][j])
                    vector1 = int(np.exp(lut_range[np.abs(p-q).astype(int)]))
                    vector2 = np.exp(-0.5 * ((i-halfWidth) ** 2 + (j-halfWidth) ** 2) / sigmaSpatial ** 2)
                    weightCombined = vector1 * vector2
                    weightCombinedKernel[i][j] = weightCombined
                    weightAccumulated = weightAccumulated + weightCombined

            result[h, w] = np.sum(weightCombinedKernel * roi) / weightAccumulated

    return result


fileName2 = "../Images_Contrast_Test/test2.png"
guidance_img = cv2.imread(fileName2, 1)
guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2GRAY)

fileName1 = "../Images_Contrast_Test/test2.png"
low_res_img = cv2.imread(fileName1, 1)
low_res_img = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2GRAY)

sampling_rate = 0.25
h, w = low_res_img.shape
down_width = int(w * sampling_rate)
down_height = int(h * sampling_rate)
down_points = (down_width, down_height)
downImage = cv2.resize(low_res_img, down_points, interpolation= cv2.INTER_LINEAR)
upImage = cv2.resize(downImage, (w, h), interpolation= cv2.INTER_LINEAR)
startTime = time.perf_counter()
halfKernelWidth = 2
adjustedImage1 = convolution2DKernel(guidance_img, upImage, halfKernelWidth)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('JBL vs original, Std:{}'.format(halfKernelWidth), adjustedImage1)
cv2.imshow('downIm',downImage)
cv2.imshow('upIm',upImage)
cv2.waitKey(0)
