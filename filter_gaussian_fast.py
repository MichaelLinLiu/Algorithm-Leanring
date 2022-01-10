# M: This program is a practise for box filter and the approximation of Gaussian by using box filters based on the tutorial from

import cv2
import numpy as np
import time


def mIntegralImage(img):
    # M: create an empty uint8 type array
    integralImage: np.array().astype("uint8") = []
    for h in range(img.shape[0]+1):
        row = []
        for w in range(img.shape[1]+1):
            row.append(0)
        integralImage.append(row)

    # M: cruise through img to get the integral image
    for h in range(0, img.shape[0]):
        for w in range(0, img.shape[1]):
            integralImage[h+1][w+1] = integralImage[h+1][w] + integralImage[h][w+1] - integralImage[h][w] + img[h][w]

    return integralImage


def boxesForGauss(size, n:int=3):
    # M: calculate the widths of two box filters according to the sigma respectively.
    sigma = size / np.sqrt(8 * np.log(2))
    idealW = int(np.sqrt(12 * sigma ** 2 / n + 1))
    if idealW % 2 == 0:
        w1 = idealW - 1
        w2 = idealW + 1
    else:
        w1 = idealW
        w2 = idealW + 2
    rawM = (12*sigma**2 - n*w1**2 - 4*n*w1 - 3*n) / (-4*w1 - 4)
    m: int = int(np.round(rawM))
    print('idealW:',idealW,'\n','raw m:',rawM,'\n','    m:', m,' w1:',w1, '\n', '  n-m:', n-m,' w2:',w2)

    # M: filter m times with the size of w1 kernel
    kernel_Box1 = []
    for h in range(w1):
        for w in range(w1):
            kernel_Box1.append(1)

    # M: filter n - m times with the size of w2 kernel
    kernel_Box2 = []
    for h in range(w2):
        for w in range(w2):
            kernel_Box2.append(1)

    return m, len(kernel_Box1), n - m, len(kernel_Box2)


def convolution(img, kernelsArray, integralImg):
    times_kernel1 = kernelsArray[0]
    times_kernel2 = kernelsArray[2]
    weights_kernel1 = kernelsArray[1]
    weights_kernel2 = kernelsArray[3]
    pad_newImg1 = int((np.sqrt(weights_kernel1) - 1) / 2)
    pad_Integral1 = int((np.sqrt(weights_kernel1) - 1))
    pad_newImg2 = int((np.sqrt(weights_kernel2) - 1) / 2)
    pad_Integral2 = int((np.sqrt(weights_kernel2) - 1))

    for t in range(times_kernel1):
        for h in range(0, img.shape[0] - pad_Integral1):
            for w in range(0, img.shape[1] - pad_Integral1):
                D = integralImg[h + pad_Integral1 + 1][w + pad_Integral1 + 1]
                C = integralImg[h + pad_Integral1 + 1][w]
                B = integralImg[h][pad_Integral1 + w + 1]
                A = integralImg[h][w]
                img[h + pad_newImg1, w + pad_newImg1] = (D - C - B + A) / weights_kernel1

    for t in range(times_kernel2):
        for h in range(0, img.shape[0] - pad_Integral2):
            for w in range(0, img.shape[1] - pad_Integral2):
                D = integralImg[h + pad_Integral2 +1][w + pad_Integral2 +1]
                C = integralImg[h + pad_Integral2 +1][w]
                B = integralImg[h][pad_Integral2 + w +1]
                A = integralImg[h][w]
                img[h + pad_newImg2, w + pad_newImg2] = (D - C - B + A) / weights_kernel2

    return img


def mGaussianBlur(image, kernel_size):
    kernels = boxesForGauss(kernel_size)
    integralImage = mIntegralImage(image)
    return convolution(image, kernels, integralImage)


# M: test the algorithm
fileName = '../Images_Contrast_Test/test2.png'
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
size = 13
startTime = time.perf_counter()
adjustedImage1 = mGaussianBlur(img1, size)
endTime = time.perf_counter()
print("M: Finished training and the process took", f"{endTime - startTime:0.4f} Seconds.")
adjustedImage2 = cv2.GaussianBlur(img,(size,size),cv2.BORDER_ISOLATED)
endTime2 = time.perf_counter()
print("M: Finished training and the process took", f"{endTime2 - endTime:0.4f} Seconds.")
cv2.imshow('mGau vs CV Gau, Std:{}'.format(size), np.hstack((adjustedImage1, adjustedImage2)))
cv2.waitKey(0)




