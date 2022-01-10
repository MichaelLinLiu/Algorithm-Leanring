# M: This program is a practise to separate bilateral filter
import numpy as np
import cv2
import time


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
    return kernel_1D


def convolution2DKernel(img, halfWidth, kernel, sigmaRange=10):
    height = img.shape[0]
    width = img.shape[1] - halfWidth - halfWidth
    result = np.zeros((height, width), dtype=np.uint8)
    height2 = img.shape[0] - halfWidth - halfWidth
    result2 = np.zeros((height2, width), dtype=np.uint8)

    lut_range = np.exp(-1.0 * np.arange(256) ** 2 / (2 * sigmaRange ** 2))

    for h in range(img.shape[0]):
        for w in range(img.shape[1]-halfWidth-halfWidth):
            roi = img[h, w: w+halfWidth+halfWidth+1]
            rangeKernel = []
            for i in roi:
                p = float(roi[0])
                q = float(i)
                rangeWeight = lut_range[np.abs(p-q).astype(int)]
                rangeKernel.append(rangeWeight)
            weightCombined = rangeKernel * kernel
            result[h,w] = np.sum(weightCombined * roi) / np.sum(weightCombined)

    for w in range(result.shape[1]):
        for h in range(result.shape[0]-halfWidth-halfWidth):
            roi = result[h: h+halfWidth+halfWidth+1, w]
            rangeKernel = []
            for i in roi:
                p = float(roi[0])
                q = float(i)
                rangeWeight = lut_range[np.abs(p-q).astype(int)]
                rangeKernel.append(rangeWeight)
            weightCombined = rangeKernel * kernel
            result2[h,w] = np.sum(weightCombined * roi) / np.sum(weightCombined)

    return result2


# M: test the algorithm
startTime = time.perf_counter()
fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
halfKernelWidth = 6
kernel = gaussian_kernel_1D(halfKernelWidth * 2 + 1)
adjustedImage1 = convolution2DKernel(img1, halfKernelWidth, kernel)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('Separable Bilateral Filter', adjustedImage1)
cv2.waitKey(0)